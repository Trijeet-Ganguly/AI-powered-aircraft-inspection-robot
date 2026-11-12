"""Synthetic aircraft-skin imagery: brushed-metal texture + corrosion / crack / dent.

Used for two things:
  1. Generating a training set so you can run the whole pipeline on day one.
  2. Building the simulated fuselage panel for `main.py --mode sim`.

Replace this with real photos of your mock panel as soon as you have them
(see README). A CNN trained only on synthetic data will NOT transfer to real metal.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["clean", "corrosion", "crack", "dent"]  # alphabetical = ImageFolder order


def metal_texture(h, w, rng):
    fine = cv2.GaussianBlur(rng.normal(size=(h, w)).astype(np.float32), (0, 0), 1.0)
    fine /= fine.std() + 1e-6
    streak = cv2.GaussianBlur(rng.normal(size=(h, w)).astype(np.float32), (0, 0), sigmaX=20, sigmaY=0.8)
    streak /= streak.std() + 1e-6
    light = cv2.resize(rng.normal(size=(3, 3)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    gray = 165 + 5 * fine + 4 * streak + 6 * light
    img = np.dstack([gray * 1.04, gray, gray * 0.96])  # slight cool tint, BGR
    return np.clip(img, 0, 255).astype(np.uint8)


def _box(img, cx, cy, r):
    h, w = img.shape[:2]
    return max(0, int(cx - r)), max(0, int(cy - r)), min(w, int(cx + r) + 1), min(h, int(cy + r) + 1)


def draw_corrosion(img, cx, cy, r, rng):
    x0, y0, x1, y1 = _box(img, cx, cy, r)
    if x1 <= x0 or y1 <= y0:
        return
    ph, pw = y1 - y0, x1 - x0
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r
    blobs = cv2.resize(rng.normal(size=(7, 7)).astype(np.float32), (pw, ph), interpolation=cv2.INTER_CUBIC)
    grain = rng.normal(size=(ph, pw)).astype(np.float32)
    field = 0.9 * blobs + 0.25 * grain + 0.9 - 1.6 * d
    alpha = np.clip(field * 2.5, 0, 1)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 1.0)[..., None] * 0.9
    tone = np.clip(0.75 + 0.75 * cv2.GaussianBlur(grain, (0, 0), 1.2), 0.4, 1.3)[..., None]
    rust = np.array([35, 80, 150], np.float32) * tone  # rust colour, BGR
    patch = img[y0:y1, x0:x1].astype(np.float32)
    img[y0:y1, x0:x1] = np.clip(patch * (1 - alpha) + rust * alpha, 0, 255).astype(np.uint8)


def draw_crack(img, cx, cy, r, rng):
    def walk(x, y, ang, length, step=3.0):
        pts = [(x, y)]
        for _ in range(max(2, int(length / step))):
            ang += rng.normal(0, 0.2)
            x += step * np.cos(ang)
            y += step * np.sin(ang)
            pts.append((x, y))
        return np.array(pts, np.float32)

    ang = rng.uniform(0, np.pi)
    main = walk(cx - r * np.cos(ang), cy - r * np.sin(ang), ang, 2 * r)
    paths = [main]
    low = len(main) // 4
    for _ in range(int(rng.integers(0, 3))):
        i = int(rng.integers(low, max(low + 1, 3 * len(main) // 4)))
        branch_ang = ang + rng.choice([-1, 1]) * rng.uniform(0.5, 1.0)
        paths.append(walk(main[i, 0], main[i, 1], branch_ang, r * rng.uniform(0.5, 1.0)))
    thick = int(rng.integers(1, 3))
    for p in paths:
        pts = p.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts + 1], False, (205, 205, 205), 1, cv2.LINE_AA)  # lit edge
        cv2.polylines(img, [pts], False, (25, 25, 28), thick, cv2.LINE_AA)     # dark fissure


def draw_dent(img, cx, cy, r, rng):
    R = r * 1.3
    x0, y0, x1, y1 = _box(img, cx, cy, R)
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx, dy = (xx - cx) / r, (yy - cy) / r
    blob = np.exp(-(dx ** 2 + dy ** 2) * 1.6)
    ang = rng.uniform(0, 2 * np.pi)
    shade = (np.cos(ang) * dx + np.sin(ang) * dy) * blob * rng.uniform(80, 110) - blob * 6
    patch = img[y0:y1, x0:x1].astype(np.float32)
    img[y0:y1, x0:x1] = np.clip(patch + shade[..., None], 0, 255).astype(np.uint8)


DRAW = {"corrosion": draw_corrosion, "crack": draw_crack, "dent": draw_dent}


def make_tile(cls, size, rng):
    img = metal_texture(size, size, rng)
    if cls != "clean":
        cx = size / 2 + rng.uniform(-0.2, 0.2) * size
        cy = size / 2 + rng.uniform(-0.2, 0.2) * size
        r = size * rng.uniform(0.22, 0.40)
        DRAW[cls](img, cx, cy, r, rng)
    return cv2.convertScaleAbs(img, alpha=rng.uniform(0.9, 1.1), beta=rng.uniform(-10, 10))


def make_panel(w_px, h_px, n_defects, rng, r_range=(28, 50), margin=70, min_gap=220):
    """Big panel with randomly placed defects. Returns (image, ground_truth_list)."""
    img = metal_texture(h_px, w_px, rng)
    truth, tries = [], 0
    while len(truth) < n_defects and tries < 1000:
        tries += 1
        cls = str(rng.choice(CLASSES[1:]))
        r = rng.uniform(*r_range)
        cx = rng.uniform(margin, w_px - margin)
        cy = rng.uniform(margin, h_px - margin)
        if any(np.hypot(cx - t["cx"], cy - t["cy"]) < min_gap for t in truth):
            continue
        DRAW[cls](img, cx, cy, r, rng)
        truth.append({"label": cls, "cx": float(cx), "cy": float(cy), "r": float(r)})
    return img, truth


def make_dataset(root, n_per_class=800, size=128, seed=0):
    rng = np.random.default_rng(seed)
    root = Path(root)
    for split, n in (("train", n_per_class), ("val", max(1, n_per_class // 4))):
        for cls in CLASSES:
            out = root / split / cls
            out.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                cv2.imwrite(str(out / f"{cls}_{i:04d}.png"), make_tile(cls, size, rng))
    print(f"Dataset written to {root.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate a synthetic defect dataset")
    ap.add_argument("--out", default="data")
    ap.add_argument("--n", type=int, default=800, help="training images per class")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    make_dataset(a.out, a.n, a.size, a.seed)

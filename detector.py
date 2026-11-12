"""CNN defect classifier + sliding-window detector.

The CNN classifies fixed-size tiles (clean / corrosion / crack / dent).
`DefectDetector.detect()` slides that classifier over a camera frame and
returns non-clean tiles above a confidence threshold, after a simple
distance-based non-max suppression.
"""
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

IMG_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], np.float32)


def build_model(num_classes, pretrained=True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    m = models.resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def _positions(n, tile, stride):
    if n < tile:
        raise ValueError(f"frame dimension {n}px is smaller than tile {tile}px")
    pos = list(range(0, n - tile + 1, stride))
    if pos[-1] != n - tile:
        pos.append(n - tile)
    return pos


class DefectDetector:
    def __init__(self, ckpt_path, conf_thresh=0.85, stride=64, device=None, batch=64):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.classes = ckpt["classes"]
        self.tile = int(ckpt.get("img_size", 128))
        self.stride = stride
        self.conf_thresh = conf_thresh
        self.batch = batch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(len(self.classes), pretrained=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval().to(self.device)

    @torch.no_grad()
    def _classify(self, tiles_bgr):
        arr = np.stack(tiles_bgr)[..., ::-1].astype(np.float32) / 255.0  # BGR -> RGB
        arr = (arr - IMG_MEAN) / IMG_STD
        x = torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 3, 1, 2)))
        probs = []
        for i in range(0, len(x), self.batch):
            logits = self.model(x[i:i + self.batch].to(self.device))
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(probs)

    def detect(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        t = self.tile
        tiles, origins = [], []
        for y in _positions(h, t, self.stride):
            for x in _positions(w, t, self.stride):
                tiles.append(frame_bgr[y:y + t, x:x + t])
                origins.append((x, y))
        probs = self._classify(tiles)

        cands = []
        for (x, y), p in zip(origins, probs):
            k = int(p.argmax())
            label = self.classes[k]
            if label != "clean" and p[k] >= self.conf_thresh:
                cands.append({"label": label, "conf": float(p[k]),
                              "cx": x + t / 2, "cy": y + t / 2,
                              "box": (x, y, x + t, y + t)})
        return self._nms(cands, min_dist=0.75 * t)

    @staticmethod
    def _nms(dets, min_dist):
        kept = []
        for d in sorted(dets, key=lambda d: -d["conf"]):
            if all(np.hypot(d["cx"] - k["cx"], d["cy"] - k["cy"]) >= min_dist for k in kept):
                kept.append(d)
        return kept

"""Builds the defect map: stitches frames into a panel mosaic, merges duplicate
detections from overlapping frames, and exports an annotated image, CSV and JSON."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np

COLORS = {"corrosion": (0, 140, 255), "crack": (0, 0, 255), "dent": (255, 160, 0)}  # BGR
# Illustrative weights only. Real severity comes from the Structural Repair Manual limits.
SEVERITY_W = {"crack": 3.0, "corrosion": 2.0, "dent": 1.0}


def frame_px_to_mm(px, py, frame_w, frame_h, cx_mm, cy_mm, fov_w_mm, fov_h_mm):
    """Convert a pixel in a frame (camera centred at cx_mm, cy_mm) to panel mm."""
    return (cx_mm + (px - frame_w / 2) * fov_w_mm / frame_w,
            cy_mm + (py - frame_h / 2) * fov_h_mm / frame_h)


class DefectMap:
    def __init__(self, width_mm, height_mm, px_per_mm=4.0, merge_mm=20.0, zone_mm=100.0, marker_mm=14.0):
        self.W, self.H, self.ppm = width_mm, height_mm, px_per_mm
        self.merge_mm, self.zone, self.marker_mm = merge_mm, zone_mm, marker_mm
        self.mosaic = np.full((int(round(height_mm * px_per_mm)), int(round(width_mm * px_per_mm)), 3),
                              40, np.uint8)
        self.defects = []

    def add_frame(self, frame, cx_mm, cy_mm, fov_w_mm, fov_h_mm):
        pw, ph = int(round(fov_w_mm * self.ppm)), int(round(fov_h_mm * self.ppm))
        if frame.shape[1] != pw or frame.shape[0] != ph:
            frame = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
        x0 = int(round((cx_mm - fov_w_mm / 2) * self.ppm))
        y0 = int(round((cy_mm - fov_h_mm / 2) * self.ppm))
        H, W = self.mosaic.shape[:2]
        sx, sy = max(0, -x0), max(0, -y0)
        dx0, dy0 = max(0, x0), max(0, y0)
        dx1, dy1 = min(W, x0 + pw), min(H, y0 + ph)
        if dx1 <= dx0 or dy1 <= dy0:
            return
        self.mosaic[dy0:dy1, dx0:dx1] = frame[sy:sy + (dy1 - dy0), sx:sx + (dx1 - dx0)]

    def add_detection(self, label, conf, x_mm, y_mm):
        for d in self.defects:
            if d["label"] == label and np.hypot(d["x_mm"] - x_mm, d["y_mm"] - y_mm) <= self.merge_mm:
                w_old, w_new = d["_wsum"], conf
                d["x_mm"] = (d["x_mm"] * w_old + x_mm * w_new) / (w_old + w_new)
                d["y_mm"] = (d["y_mm"] * w_old + y_mm * w_new) / (w_old + w_new)
                d["_wsum"] += conf
                d["conf"] = max(d["conf"], float(conf))
                d["hits"] += 1
                return
        self.defects.append({"label": label, "conf": float(conf), "x_mm": float(x_mm),
                             "y_mm": float(y_mm), "hits": 1, "_wsum": float(conf)})

    def _zone(self, x, y):
        return f"{chr(65 + min(int(x // self.zone), 25))}{int(y // self.zone) + 1}"

    def results(self):
        out = []
        for i, d in enumerate(sorted(self.defects, key=lambda d: (d["y_mm"], d["x_mm"])), 1):
            out.append({"id": i, "label": d["label"], "conf": round(d["conf"], 3),
                        "x_mm": round(d["x_mm"], 1), "y_mm": round(d["y_mm"], 1),
                        "zone": self._zone(d["x_mm"], d["y_mm"]), "hits": d["hits"],
                        "severity": round(SEVERITY_W.get(d["label"], 1.0) * d["conf"], 2)})
        return out

    def render(self):
        base = self.mosaic.copy()
        over = base.copy()
        res = self.results()
        r = max(4, int(self.marker_mm * self.ppm))
        for d in res:
            p = (int(d["x_mm"] * self.ppm), int(d["y_mm"] * self.ppm))
            cv2.circle(over, p, r, COLORS.get(d["label"], (255, 255, 255)), -1)
        out = cv2.addWeighted(over, 0.35, base, 0.65, 0)

        step = int(self.zone * self.ppm)  # zone grid
        for x in range(step, out.shape[1], step):
            cv2.line(out, (x, 0), (x, out.shape[0] - 1), (90, 90, 90), 1)
        for y in range(step, out.shape[0], step):
            cv2.line(out, (0, y), (out.shape[1] - 1, y), (90, 90, 90), 1)

        for d in res:
            c = COLORS.get(d["label"], (255, 255, 255))
            p = (int(d["x_mm"] * self.ppm), int(d["y_mm"] * self.ppm))
            cv2.circle(out, p, r, c, 2, cv2.LINE_AA)
            cv2.putText(out, f'{d["id"]} {d["label"]} {d["conf"]:.2f}', (p[0] + r + 4, p[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(out, f'{d["id"]} {d["label"]} {d["conf"]:.2f}', (p[0] + r + 4, p[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 1, cv2.LINE_AA)
        return out

    def save(self, out_dir):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        res = self.results()
        cv2.imwrite(str(out_dir / "mosaic.png"), self.mosaic)
        cv2.imwrite(str(out_dir / "defect_map.png"), self.render())
        with open(out_dir / "defects.csv", "w", newline="") as f:
            cols = ["id", "label", "conf", "x_mm", "y_mm", "zone", "hits", "severity"]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(res)
        counts = {}
        for d in res:
            counts[d["label"]] = counts.get(d["label"], 0) + 1
        report = {"panel_mm": [self.W, self.H], "defect_count": len(res), "counts": counts,
                  "total_severity": round(sum(d["severity"] for d in res), 2), "defects": res}
        with open(out_dir / "report.json", "w") as f:
            json.dump(report, f, indent=2)
        return report

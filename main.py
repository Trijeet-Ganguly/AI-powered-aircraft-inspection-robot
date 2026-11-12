"""Run an inspection scan: raster-scan the panel, detect defects, build the defect map.

  Simulation : python main.py --mode sim
  Hardware   : python main.py --mode hardware --port /dev/ttyUSB0 --fov-w 128 --fov-h 128
"""
import argparse
import time

import cv2
import numpy as np

from detector import DefectDetector
from mapper import DefectMap, frame_px_to_mm


def _axis(length, fov, step):
    if length <= fov:
        return [length / 2]
    pts = list(np.arange(fov / 2, length - fov / 2 + 1e-6, step))
    if pts[-1] < length - fov / 2 - 1e-6:
        pts.append(length - fov / 2)
    return [float(p) for p in pts]


def plan_raster(panel_w, panel_h, fov_w, fov_h, overlap=0.2):
    """Serpentine waypoints (camera centres, mm) covering the whole panel."""
    xs = _axis(panel_w, fov_w, fov_w * (1 - overlap))
    ys = _axis(panel_h, fov_h, fov_h * (1 - overlap))
    path = []
    for i, y in enumerate(ys):
        row = xs if i % 2 == 0 else xs[::-1]
        path += [(x, y) for x in row]
    return path


def evaluate(truth, results, tol_mm=20.0):
    """Sim only: match detections to ground truth (same class, within tol_mm)."""
    used, tp = set(), 0
    for t in truth:
        for i, d in enumerate(results):
            if i not in used and d["label"] == t["label"] and \
                    np.hypot(d["x_mm"] - t["x_mm"], d["y_mm"] - t["y_mm"]) <= tol_mm:
                used.add(i)
                tp += 1
                break
    fp, fn = len(results) - tp, len(truth) - tp
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    print(f"\nGround-truth check: TP {tp}  FP {fp}  FN {fn}  precision {prec:.2f}  recall {rec:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sim", "hardware"], default="sim")
    ap.add_argument("--model", default="models/defect_cnn.pt")
    ap.add_argument("--panel-w", type=float, default=600, help="panel width, mm")
    ap.add_argument("--panel-h", type=float, default=400, help="panel height, mm")
    ap.add_argument("--fov-w", type=float, default=128, help="camera field of view width, mm")
    ap.add_argument("--fov-h", type=float, default=128, help="camera field of view height, mm")
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--ppm", type=float, default=4.0, help="pixels per mm used for the model (keep = training scale)")
    ap.add_argument("--conf", type=float, default=0.85, help="min confidence to report a defect")
    ap.add_argument("--n-defects", type=int, default=8, help="sim only")
    ap.add_argument("--seed", type=int, default=0, help="sim only")
    ap.add_argument("--port", default="/dev/ttyUSB0", help="hardware only (e.g. COM3 on Windows)")
    ap.add_argument("--cam", type=int, default=0, help="hardware only")
    ap.add_argument("--show", action="store_true", help="live preview window")
    ap.add_argument("--out", default="runs/latest")
    a = ap.parse_args()

    if a.panel_w < a.fov_w or a.panel_h < a.fov_h:
        raise SystemExit("panel must be at least as large as the camera field of view")

    det = DefectDetector(a.model, conf_thresh=a.conf)
    dmap = DefectMap(a.panel_w, a.panel_h, px_per_mm=a.ppm)

    if a.mode == "sim":
        from robot import SimRobot
        robot = SimRobot(a.panel_w, a.panel_h, (a.fov_w, a.fov_h), a.ppm, a.n_defects, a.seed)
    else:
        from robot import HardwareRobot
        robot = HardwareRobot(a.port, a.cam)

    fw, fh = int(round(a.fov_w * a.ppm)), int(round(a.fov_h * a.ppm))
    path = plan_raster(a.panel_w, a.panel_h, a.fov_w, a.fov_h, a.overlap)
    print(f"Scanning {len(path)} positions over {a.panel_w:.0f}x{a.panel_h:.0f} mm panel...")
    t0 = time.time()
    try:
        for i, (x, y) in enumerate(path, 1):
            robot.move_to(x, y)
            frame = robot.grab_frame()
            if frame.shape[1] != fw or frame.shape[0] != fh:  # normalise scale for the CNN
                frame = cv2.resize(frame, (fw, fh), interpolation=cv2.INTER_AREA)
            dmap.add_frame(frame, x, y, a.fov_w, a.fov_h)
            dets = det.detect(frame)
            for d in dets:
                xm, ym = frame_px_to_mm(d["cx"], d["cy"], fw, fh, x, y, a.fov_w, a.fov_h)
                dmap.add_detection(d["label"], d["conf"], xm, ym)
            print(f"  [{i:>3}/{len(path)}] at ({x:6.1f},{y:6.1f}) mm -> {len(dets)} detection(s)")
            if a.show:
                vis = frame.copy()
                for d in dets:
                    x0, y0, x1, y1 = d["box"]
                    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 2)
                    cv2.putText(vis, f'{d["label"]} {d["conf"]:.2f}', (x0, y0 + 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.imshow("inspection", vis)
                cv2.waitKey(1)
    finally:
        robot.close()
        if a.show:
            cv2.destroyAllWindows()

    report = dmap.save(a.out)
    print(f"\nScan done in {time.time() - t0:.1f}s. {report['defect_count']} defect(s): {report['counts']}")
    for d in report["defects"]:
        print(f"  #{d['id']:<2} {d['label']:<10} conf {d['conf']:.2f}  zone {d['zone']}  "
              f"({d['x_mm']:.0f}, {d['y_mm']:.0f}) mm")
    print(f"Outputs in {a.out}/  (defect_map.png, mosaic.png, defects.csv, report.json)")

    if a.mode == "sim":
        evaluate(robot.truth, report["defects"])


if __name__ == "__main__":
    main()

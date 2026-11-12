# AI Aircraft Inspection Robot

A gantry/crawler scans a mock fuselage panel in a serpentine raster, a CNN classifies camera tiles as
**clean / corrosion / crack / dent**, and the results are merged into a **defect map** (annotated image,
CSV and JSON report with panel zones).

```
camera frame -> sliding-window CNN -> detections (px) -> panel coordinates (mm)
             -> merge duplicates across overlapping frames -> defect map + report
```

## Files

| File | Purpose |
|---|---|
| `synthetic_data.py` | Generates synthetic metal images with defects (training data + simulated panel) |
| `train_cnn.py` | Fine-tunes ResNet18 on `data/train` and `data/val` |
| `detector.py` | Model definition + sliding-window detector with NMS |
| `mapper.py` | Mosaic stitching, defect merging, overlay, CSV/JSON export |
| `robot.py` | `SimRobot` (no hardware) and `HardwareRobot` (Arduino gantry + USB camera) |
| `main.py` | Raster path planning and the full scan |
| `arduino_gantry/` | Stepper firmware (`G x y` / `H` serial protocol) |

## Quick start (simulation, no hardware)

```bash
pip install -r requirements.txt
python synthetic_data.py --out data --n 800     # make a demo dataset
python train_cnn.py --data data --epochs 8      # ~minutes on CPU, faster on GPU
python main.py --mode sim                       # scan a simulated panel
```

Outputs land in `runs/latest/`: `defect_map.png`, `mosaic.png`, `defects.csv`, `report.json`.
In sim mode the script also compares detections with the known ground truth (precision / recall).

## Using real data (important)

A model trained only on synthetic images will not work on real metal. For a real demo:

1. Photograph a mock panel (aluminium sheet) with real dents, scratches/cracks and corrosion
   (e.g. salt-water and vinegar on bare aluminium), plus clean areas, under the same lighting and
   camera height you will use during the scan.
2. Crop 128x128 px tiles at your working scale (4 px per mm) into
   `data/train/<class>/` and `data/val/<class>/`. Aim for a few hundred per class.
3. Optionally mix in public datasets (e.g. NEU surface defect, Kaggle corrosion datasets) for pretraining.
4. Re-run `train_cnn.py`, check the confusion matrix, then tune `--conf`.

## Hardware (gantry version)

- Arduino Uno + CNC Shield + 2x A4988/DRV8825, 2 NEMA17 steppers on a 2-axis belt gantry, USB camera
  pointing straight down at the panel with a ring light.
- Flash `arduino_gantry/arduino_gantry.ino` (needs the AccelStepper library). Check `STEPS_PER_MM`.
- Move the carriage by hand so the camera centre is over the panel's top-left corner, then run:

```bash
python main.py --mode hardware --port /dev/ttyUSB0 --panel-w 600 --panel-h 400 --fov-w 128 --fov-h 128
```

Measure the real camera field of view (mm) on the panel and pass it as `--fov-w/--fov-h`. Frames are
resized to 4 px/mm so the CNN sees the same scale it was trained on.

For a crawler or drone instead of a gantry, keep `main.py`, `detector.py` and `mapper.py`, and write a
new class in `robot.py` with `move_to`, `grab_frame` and `close`. A drone or crawler needs a position
estimate (visual markers/AprilTags, or odometry) feeding `move_to`.

## Limitations

- Classification is tile-based, so defect size is only approximate. Upgrade path: YOLOv8 or a
  segmentation model (U-Net) for real boxes and areas.
- Severity weights in `mapper.py` are illustrative. Real inspection decisions follow the aircraft's
  Structural Repair Manual limits, never this score.
- Specular reflections, rivets, seams and paint edges will cause false positives unless they appear as
  `clean` examples in your training data.

## Ideas to extend

- Grad-CAM heatmaps to show what the CNN is looking at.
- Repeat scans over time and diff the defect maps to track growth.
- ROS 2 node wrapper (publish detections as a topic, RViz overlay).
- Export a PDF inspection report.

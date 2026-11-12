"""Robot interfaces. Both expose the same three methods:
    move_to(x_mm, y_mm)  - put the camera centre over that point of the panel
    grab_frame()         - return a BGR image
    close()
"""
import time

import cv2
import numpy as np

from synthetic_data import make_panel


class SimRobot:
    """Simulated gantry over a synthetic fuselage panel (no hardware needed)."""

    def __init__(self, panel_w_mm, panel_h_mm, fov_mm, px_per_mm=4.0, n_defects=8, seed=0):
        self.rng = np.random.default_rng(seed)
        self.ppm, self.fov = px_per_mm, fov_mm
        self.panel, truth_px = make_panel(int(panel_w_mm * px_per_mm), int(panel_h_mm * px_per_mm),
                                          n_defects, self.rng)
        self.truth = [{"label": t["label"], "x_mm": t["cx"] / px_per_mm, "y_mm": t["cy"] / px_per_mm}
                      for t in truth_px]
        self.pos = (fov_mm[0] / 2, fov_mm[1] / 2)

    def move_to(self, x, y):
        self.pos = (x, y)

    def grab_frame(self):
        fw, fh = int(round(self.fov[0] * self.ppm)), int(round(self.fov[1] * self.ppm))
        H, W = self.panel.shape[:2]
        x0 = int(np.clip(round(self.pos[0] * self.ppm - fw / 2), 0, W - fw))
        y0 = int(np.clip(round(self.pos[1] * self.ppm - fh / 2), 0, H - fh))
        frame = self.panel[y0:y0 + fh, x0:x0 + fw].astype(np.float32)
        frame += self.rng.normal(0, 3, frame.shape)  # camera noise
        return np.clip(frame, 0, 255).astype(np.uint8)

    def close(self):
        pass


class HardwareRobot:
    """Arduino gantry over serial (see arduino_gantry/) + USB camera."""

    def __init__(self, port, cam_index=0, baud=115200, cam_size=(1280, 720)):
        import serial  # pyserial
        self.ser = serial.Serial(port, baud, timeout=1)
        self._wait_for("READY", 10)
        self.ser.write(b"H\n")  # current carriage position (camera over panel corner) becomes 0,0
        self._wait_for("OK", 5)
        self.cap = cv2.VideoCapture(cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_size[1])
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {cam_index}")

    def _wait_for(self, token, timeout):
        t0 = time.time()
        while time.time() - t0 < timeout:
            line = self.ser.readline().decode(errors="ignore").strip()
            if line.startswith(token):
                return
        raise TimeoutError(f"no '{token}' from controller within {timeout}s")

    def move_to(self, x, y):
        self.ser.write(f"G {x:.1f} {y:.1f}\n".encode())
        self._wait_for("OK", 60)
        time.sleep(0.3)  # let vibration settle before imaging

    def grab_frame(self):
        for _ in range(3):  # flush stale buffered frames
            self.cap.read()
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        return frame

    def close(self):
        self.cap.release()
        self.ser.close()

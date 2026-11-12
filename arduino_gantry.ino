// 2-axis stepper gantry controller for the inspection robot.
// Hardware: Arduino Uno + CNC Shield v3 + 2x A4988/DRV8825 (X and Y axes).
// Library:  AccelStepper (install from Arduino Library Manager).
//
// Serial protocol, 115200 baud, newline terminated:
//   "G x y"  move camera centre to absolute (x, y) in mm  -> replies "OK"
//   "H"      declare current position as (0, 0)           -> replies "OK"
// On boot it prints "READY".
//
// No limit switches: move the carriage by hand so the camera centre is over the
// panel's top-left corner, then send "H". Panel coordinates in main.py are
// measured from that corner. Add limit switches before leaving it unattended.

#include <AccelStepper.h>

const float STEPS_PER_MM = 80.0;  // GT2 belt, 20T pulley, 1/16 microstepping: 200*16/40
const float MAX_SPEED = 3000.0;   // steps/s (AccelStepper tops out near 4000 on an Uno)
const float ACCEL = 1500.0;       // steps/s^2
const int ENABLE_PIN = 8;         // CNC shield: LOW = drivers enabled

AccelStepper stepX(AccelStepper::DRIVER, 2, 5);  // STEP, DIR
AccelStepper stepY(AccelStepper::DRIVER, 3, 6);

void setup() {
  Serial.begin(115200);
  pinMode(ENABLE_PIN, OUTPUT);
  digitalWrite(ENABLE_PIN, LOW);
  for (AccelStepper* s : {&stepX, &stepY}) {
    s->setMaxSpeed(MAX_SPEED);
    s->setAcceleration(ACCEL);
  }
  Serial.println("READY");
}

void moveTo(float xmm, float ymm) {
  stepX.moveTo((long)(xmm * STEPS_PER_MM));
  stepY.moveTo((long)(ymm * STEPS_PER_MM));
  while (stepX.distanceToGo() != 0 || stepY.distanceToGo() != 0) {
    stepX.run();
    stepY.run();
  }
}

void loop() {
  if (!Serial.available()) return;
  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() == 0) return;

  if (line[0] == 'G') {
    String rest = line.substring(1);
    rest.trim();
    int sp = rest.indexOf(' ');
    if (sp < 0) { Serial.println("ERR"); return; }
    float x = rest.substring(0, sp).toFloat();
    float y = rest.substring(sp + 1).toFloat();
    moveTo(x, y);
    Serial.println("OK");
  } else if (line[0] == 'H') {
    stepX.setCurrentPosition(0);
    stepY.setCurrentPosition(0);
    Serial.println("OK");
  } else {
    Serial.println("ERR");
  }
}

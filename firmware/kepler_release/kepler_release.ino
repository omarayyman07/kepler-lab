// Kepler releaser: servo arm + electromagnet, driven over USB serial.
//
// Mechanics: the servo shaft sits on the pendulum's pivot axis, a few cm
// BEHIND the swing plane. A light arm (wooden dowel / ruler) of the same
// length as the pendulum carries a small 5V electromagnet at its tip. The bob
// is steel (or has a steel washer glued to its back). The magnet holds the bob
// from behind, the arm carries it to the requested angle, the magnet lets go,
// and the bob swings freely in front of the arm.
//
// Wiring: servo signal -> SERVO_PIN (servo powered from 5V supply, common GND),
//         magnet via logic-level MOSFET or relay module on MAGNET_PIN
//         (+ flyback diode across the coil if you drive it with a MOSFET).
//
// Protocol (115200 baud, newline-terminated):
//   ARM <deg>  magnet on, arm to vertical, grab the bob, sweep to <deg>  -> READY
//   DROP       magnet off                                                  -> DROPPED
//   HOME       magnet off, arm back to vertical                            -> READY

#if defined(ESP32)
  #include <ESP32Servo.h>
  const int SERVO_PIN = 13;
  const int MAGNET_PIN = 26;
#else
  #include <Servo.h>
  const int SERVO_PIN = 9;
  const int MAGNET_PIN = 7;
#endif

// Calibrate these two: the servo command that points the arm straight down,
// and +1 or -1 so that positive pendulum angles go the way you expect.
const int SERVO_VERTICAL = 90;
const int DIRECTION = 1;
const float MAX_ANGLE = 75.0;
const int SWEEP_MS_PER_DEG = 25;   // slow sweep so the bob does not swing off

Servo arm;
float current = 0.0;

void moveTo(float deg) {
  deg = constrain(deg, -MAX_ANGLE, MAX_ANGLE);
  int steps = (int)abs(deg - current);
  float dir = (deg > current) ? 1.0 : -1.0;
  for (int i = 0; i < steps; i++) {
    current += dir;
    arm.write(SERVO_VERTICAL + DIRECTION * (int)round(current));
    delay(SWEEP_MS_PER_DEG);
  }
  current = deg;
  arm.write(SERVO_VERTICAL + DIRECTION * (int)round(current));
  delay(200);
}

void setup() {
  Serial.begin(115200);
  pinMode(MAGNET_PIN, OUTPUT);
  digitalWrite(MAGNET_PIN, LOW);
  arm.attach(SERVO_PIN);
  arm.write(SERVO_VERTICAL);
  Serial.println("READY");
}

void loop() {
  if (!Serial.available()) return;
  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.startsWith("ARM")) {
    float deg = line.substring(3).toFloat();
    if (abs(deg) > MAX_ANGLE) { Serial.println("ERR angle out of range"); return; }
    digitalWrite(MAGNET_PIN, HIGH);
    moveTo(0);
    delay(2500);            // a bob still swinging is caught as it passes the magnet
    moveTo(deg);
    Serial.println("READY");
  } else if (line == "DROP") {
    digitalWrite(MAGNET_PIN, LOW);
    Serial.println("DROPPED");
  } else if (line == "HOME") {
    digitalWrite(MAGNET_PIN, LOW);
    moveTo(0);
    Serial.println("READY");
  } else if (line.length()) {
    Serial.println("ERR unknown command");
  }
}

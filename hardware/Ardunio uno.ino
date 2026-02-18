#include <AFMotor.h>
#include <Servo.h>

// Motors
AF_DCMotor leftMotor(1);
AF_DCMotor rightMotor(2);

// Servo
Servo scanner;
const int SERVO_PIN = 9;

const int SERVO_LEFT = 50;
const int SERVO_CENTER = 90;
const int SERVO_RIGHT = 130;

int motorSpeed = 180;

// ----------------- MOTOR FUNCTIONS -----------------
void stopMotors() {
  leftMotor.run(RELEASE);
  rightMotor.run(RELEASE);
}

void forward() {
  scanner.write(SERVO_CENTER);
  leftMotor.setSpeed(motorSpeed);
  rightMotor.setSpeed(motorSpeed);
  leftMotor.run(FORWARD);
  rightMotor.run(FORWARD);
}

void backward() {
  scanner.write(SERVO_CENTER);
  leftMotor.setSpeed(motorSpeed);
  rightMotor.setSpeed(motorSpeed);
  leftMotor.run(BACKWARD);
  rightMotor.run(BACKWARD);
}

void turnLeft() {
  scanner.write(SERVO_LEFT);
  leftMotor.setSpeed(motorSpeed - 60);
  rightMotor.setSpeed(motorSpeed);
  leftMotor.run(FORWARD);
  rightMotor.run(FORWARD);
}

void turnRight() {
  scanner.write(SERVO_RIGHT);
  leftMotor.setSpeed(motorSpeed);
  rightMotor.setSpeed(motorSpeed - 60);
  leftMotor.run(FORWARD);
  rightMotor.run(FORWARD);
}

// ----------------- SERVO SCAN FUNCTION -----------------
void scanLeftRight() {
  stopMotors();

  // Look Left
  scanner.write(SERVO_LEFT);
  delay(600);

  // Look Right
  scanner.write(SERVO_RIGHT);
  delay(600);

  // Return Center
  scanner.write(SERVO_CENTER);
  delay(400);
}

// ----------------- SETUP -----------------
void setup() {
  Serial.begin(9600);

  scanner.attach(SERVO_PIN);
  scanner.write(SERVO_CENTER);

  leftMotor.setSpeed(motorSpeed);
  rightMotor.setSpeed(motorSpeed);

  stopMotors();
  Serial.println("Robot Ready");
}

// ----------------- LOOP -----------------
void loop() {

  if (Serial.available()) {
    char cmd = Serial.read();

    switch (cmd) {
      case 'F': forward(); break;
      case 'B': backward(); break;
      case 'L': turnLeft(); break;
      case 'R': turnRight(); break;
      case 'S': stopMotors(); break;
      case 'X': scanLeftRight(); break;
    }
  }
}

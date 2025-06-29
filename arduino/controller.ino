// Arduino Sketch: controller.ino
// This sketch receives steering angle, speed, and vehicle status from Python
// via serial communication and can control motors/servos accordingly.

// Define pins for motors/servos (example pins)
const int STEERING_SERVO_PIN = 9; // Example: Servo for steering
const int MOTOR_PWM_PIN = 10;     // Example: PWM pin for motor speed
const int MOTOR_DIR_PIN1 = 11;    // Example: Motor direction pin 1
const int MOTOR_DIR_PIN2 = 12;    // Example: Motor direction pin 2

// Include Servo library if using servos
#include <Servo.h>
Servo steeringServo;

// Variables to store received data
int receivedAngle = 90; // Default center angle for servo
int receivedSpeed = 0;  // Default speed
int receivedStatus = 0; // 0: Düz, 1: Viraj, 2: Dur

void setup() {
  Serial.begin(9600); // Initialize serial communication at 9600 baud
  while (!Serial); // Wait for serial port to connect. Needed for native USB port only

  steeringServo.attach(STEERING_SERVO_PIN);
  steeringServo.write(receivedAngle); // Set initial steering to center

  pinMode(MOTOR_PWM_PIN, OUTPUT);
  pinMode(MOTOR_DIR_PIN1, OUTPUT);
  pinMode(MOTOR_DIR_PIN2, OUTPUT);

  Serial.println("Arduino Controller Ready");
}

void loop() {
  if (Serial.available() > 0) {
    String message = Serial.readStringUntil('\n'); // Read the incoming message until newline
    parseMessage(message);
  }

  // Apply control based on received data
  applyControl();

  // Optional: Send status back to Python (e.g., "ACK\n")
  // Serial.println("ACK");
}

void parseMessage(String msg) {
  // Expected format: "A<angle>S<speed>V<status_code>"
  // Example: "A90S50V0"

  int angleIndex = msg.indexOf('A');
  int speedIndex = msg.indexOf('S');
  int statusIndex = msg.indexOf('V');

  if (angleIndex != -1 && speedIndex != -1 && statusIndex != -1) {
    String angleStr = msg.substring(angleIndex + 1, speedIndex);
    String speedStr = msg.substring(speedIndex + 1, statusIndex);
    String statusStr = msg.substring(statusIndex + 1);

    receivedAngle = angleStr.toInt();
    receivedSpeed = speedStr.toInt();
    receivedStatus = statusStr.toInt();

    // Constrain values to valid ranges
    receivedAngle = constrain(receivedAngle, 0, 180); // Servo angle 0-180
    receivedSpeed = constrain(receivedSpeed, 0, 255); // PWM speed 0-255
    receivedStatus = constrain(receivedStatus, 0, 2); // Status codes 0, 1, 2

    Serial.print("Received - Angle: "); Serial.print(receivedAngle);
    Serial.print(", Speed: "); Serial.print(receivedSpeed);
    Serial.print(", Status: "); Serial.println(receivedStatus);
  } else {
    Serial.print("Invalid message format: ");
    Serial.println(msg);
  }
}

void applyControl() {
  // Control Steering Servo
  steeringServo.write(receivedAngle);

  // Control Motor Speed and Direction
  if (receivedStatus == 2) { // Status "Dur" (Stop)
    digitalWrite(MOTOR_DIR_PIN1, LOW);
    digitalWrite(MOTOR_DIR_PIN2, LOW);
    analogWrite(MOTOR_PWM_PIN, 0); // Stop motor
  } else { // "Düz" or "Viraj" (Move)
    // Example: Forward direction
    digitalWrite(MOTOR_DIR_PIN1, HIGH);
    digitalWrite(MOTOR_DIR_PIN2, LOW);
    analogWrite(MOTOR_PWM_PIN, receivedSpeed); // Set motor speed
  }

  // Add more complex control logic here based on receivedStatus
  // e.g., for "Viraj" status, you might adjust speed differently or
  // implement more nuanced steering.
}
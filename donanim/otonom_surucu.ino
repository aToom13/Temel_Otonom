/*
 * Temel Otonom Arac - Arduino Motor ve Servo Kontrol Yazilimi
 * Cift protokol destegi:
 *   1) ASCII Format: 'H<hiz>A<aci>\n' (Orn: H110A-15.0\n -> Hiz: 110, Aci: -15.0 deg)
 *   2) Kat4 / Robotaksi Formati: '[f/r/l]<aci>v<hiz>\n' (Orn: r015v110\n -> Saga 15 deg, Hiz: 110)
 */

#include <Servo.h>

// Pin Tanimlamalari
const int SERVO_PIN = 9;
const int MOTOR_PWM_PIN = 5;
const int MOTOR_DIR_PIN = 6;

// Sinirlar ve Merkez Degerleri
const int SERVO_MERKEZ = 90;   // Duz gidis acisi (derece)
const int SERVO_MIN = 55;      // Maksimum sola donus acisi
const int SERVO_MAX = 125;     // Maksimum saga donus acisi

Servo direksiyonServosu;
String gelenVeri = "";

void setup() {
  Serial.begin(115200);
  
  direksiyonServosu.attach(SERVO_PIN);
  direksiyonServosu.write(SERVO_MERKEZ);

  pinMode(MOTOR_PWM_PIN, OUTPUT);
  pinMode(MOTOR_DIR_PIN, OUTPUT);

  // Baslangicta motoru durdur
  analogWrite(MOTOR_PWM_PIN, 0);
  digitalWrite(MOTOR_DIR_PIN, LOW);
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n') {
      gelenVeri.trim();
      if (gelenVeri.length() > 0) {
        komutIsle(gelenVeri);
      }
      gelenVeri = "";
    } else if (c != '\r') {
      gelenVeri += c;
    }
  }
}

void komutIsle(String komut) {
  int hedefHiz = 0;
  float hedefAci = 0.0;

  // Protokol 1: H<hiz>A<aci>
  int hIndex = komut.indexOf('H');
  int aIndex = komut.indexOf('A');

  if (hIndex != -1 && aIndex != -1 && aIndex > hIndex) {
    hedefHiz = komut.substring(hIndex + 1, aIndex).toInt();
    hedefAci = komut.substring(aIndex + 1).toFloat();
  }
  // Protokol 2: [f/r/l]<aci>v<hiz> (Kat4 Robotaksi Formati)
  else if ((komut.charAt(0) == 'f' || komut.charAt(0) == 'r' || komut.charAt(0) == 'l') && komut.indexOf('v') != -1) {
    char yon = komut.charAt(0);
    int vIndex = komut.indexOf('v');
    int aciVal = komut.substring(1, vIndex).toInt();
    hedefHiz = komut.substring(vIndex + 1).toInt();

    if (yon == 'r') {
      hedefAci = (float)aciVal;
    } else if (yon == 'l') {
      hedefAci = -(float)aciVal;
    } else {
      hedefAci = 0.0;
    }
  } else {
    return; // Bilinmeyen format
  }

  // Direksiyon servosu acisini sinirla ve uygula
  int servoAci = constrain(SERVO_MERKEZ + (int)hedefAci, SERVO_MIN, SERVO_MAX);
  direksiyonServosu.write(servoAci);

  // DC Motor PWM surusu
  int motorPWM = constrain(hedefHiz, 0, 255);
  digitalWrite(MOTOR_DIR_PIN, HIGH);
  analogWrite(MOTOR_PWM_PIN, motorPWM);
}

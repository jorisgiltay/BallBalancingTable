/*
 * Simple BNO055 IMU for Ball Balancing
 * 
 * Features:
 * - No calibration requirements - just use raw sensor data
 * - 100Hz output rate for real-time control
 * - Focus on pitch/roll for table control
 * - Much simpler and more reliable
 */

#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>

Adafruit_BNO055 bno = Adafruit_BNO055(55);

unsigned long lastTime = 0;
bool initialized = false;

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("INIT: Starting simple BNO055 for ball balancing...");

  initialized = bno.begin();
  if (!initialized) {
    Serial.println("ERROR: BNO055 not detected. Check wiring!");
    while (1) {
      delay(1000);
      Serial.println("ERROR: BNO055 initialization failed");
    }
  }

  bno.setExtCrystalUse(true);
  Serial.println("INIT: BNO055 initialized successfully");
  Serial.println("INFO: Using uncalibrated mode - perfect for ball balancing!");
  
  lastTime = millis();
  Serial.println("READY: System ready for ball balancing control");
}

void loop() {
  unsigned long now = millis();
  
  // Output IMU data at 100Hz for ball balancing control
  if (now - lastTime >= 10) {
    lastTime = now;

    imu::Vector<3> euler = bno.getVector(Adafruit_BNO055::VECTOR_EULER);

    // Output format: heading,pitch,roll
    // For ball balancing, you mainly care about pitch and roll
    if (isnan(euler.x()) || isnan(euler.y()) || isnan(euler.z())) {
      Serial.println("DATA: 0.00,0.00,0.00");
    } else {
      Serial.print("DATA: ");
      Serial.print(euler.x(), 2);  // Heading (probably not needed)
      Serial.print(",");
      Serial.print(euler.y(), 2);  // Pitch (table tilt forward/back)
      Serial.print(",");
      Serial.println(euler.z(), 2); // Roll (table tilt left/right)
    }
  }
}

/*
  Enhanced BNO055 Arduino Code for Embedded IMU Calibration
  
  This version outputs both Euler angles and raw gyroscope data
  to support advanced calibration of embedded table IMUs.
*/

#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>

// Create BNO055 instance
Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28);

void setup() {
  Serial.begin(115200);
  Serial.println("Enhanced BNO055 for Embedded Table IMU");
  
  // Initialize BNO055
  if (!bno.begin()) {
    Serial.println("ERROR: No BNO055 detected. Check wiring!");
    while (1);
  }
  
  delay(1000);
  
  // Use external crystal for better accuracy
  bno.setExtCrystalUse(true);
  
  Serial.println("READY: Enhanced BNO055 initialized");
  Serial.println("INFO: Outputs both Euler angles and gyroscope data");
}

void loop() {
  // Get Euler angles (for normal operation)
  sensors_event_t orientationData;
  bno.getEvent(&orientationData, Adafruit_BNO055::VECTOR_EULER);
  
  // Get gyroscope data (for calibration)
  sensors_event_t gyroData;
  bno.getEvent(&gyroData, Adafruit_BNO055::VECTOR_GYROSCOPE);
  
  // Check if data is valid
  if (orientationData.orientation.x >= 0 || orientationData.orientation.x < 0) {
    // Output Euler angles (normal format)
    Serial.print("DATA: ");
    Serial.print(orientationData.orientation.x, 1);  // Heading
    Serial.print(",");
    Serial.print(orientationData.orientation.y, 1);  // Pitch  
    Serial.print(",");
    Serial.println(orientationData.orientation.z, 1); // Roll
    
    // Output gyroscope data (for calibration)
    Serial.print("GYRO: ");
    Serial.print(gyroData.gyro.x, 3);  // X-axis gyro (rad/s)
    Serial.print(",");
    Serial.print(gyroData.gyro.y, 3);  // Y-axis gyro (rad/s) 
    Serial.print(",");
    Serial.println(gyroData.gyro.z, 3); // Z-axis gyro (rad/s)
  } else {
    Serial.println("WARNING: Invalid sensor data");
  }
  
  delay(10); // 100Hz output rate
}

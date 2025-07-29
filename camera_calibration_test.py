"""
Camera Calibration Test - Safe Version

This script tests the table calibration process safely.
It will detect your table/mousemat and save the calibration data.

Usage: python camera_calibration_test.py
"""

import cv2
import numpy as np
import time
import os
import json
from typing import Optional, Tuple
import pyrealsense2 as rs


class CameraCalibrationTest:
    """
    Safe calibration test for table detection
    """
    
    def __init__(self):
        self.pipeline = None
        self.config = None
        self.table_corners_pixels = None
        # Mousemat dimensions (18cm x 22cm) 
        self.table_width = 0.22   # 22cm width
        self.table_height = 0.18  # 18cm height
        
        # Output directory for calibration data
        self.output_dir = "calibration_data"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"📁 Created calibration directory: {self.output_dir}")
    
    def initialize_camera(self) -> bool:
        """Initialize RealSense camera"""
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # Configure streams
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            
            # Start streaming
            profile = self.pipeline.start(self.config)
            
            print("✅ Camera initialized for calibration")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize camera: {e}")
            return False
    
    def detect_table_corners(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect table corners (same as preview script)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_corners = None
        best_area = 0
        
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            area = cv2.contourArea(contour)
            if len(approx) == 4 and area > 10000:
                if area > best_area:
                    best_area = area
                    corners = approx.reshape(4, 2).astype(np.float32)
                    best_corners = self._sort_corners(corners)
        
        return best_corners
    
    def _sort_corners(self, corners: np.ndarray) -> np.ndarray:
        """Sort corners consistently"""
        center = np.mean(corners, axis=0)
        
        def angle_from_center(point):
            return np.arctan2(point[1] - center[1], point[0] - center[0])
        
        sorted_corners = sorted(corners, key=angle_from_center)
        return np.array(sorted_corners, dtype=np.float32)
    
    def run_calibration(self, num_samples: int = 10) -> bool:
        """Run table calibration process"""
        if not self.pipeline:
            print("❌ Camera not initialized")
            return False
        
        print(f"\n🎯 Starting Table Calibration")
        print("=" * 40)
        print(f"📊 Will take {num_samples} samples")
        print("📋 Make sure your table/mousemat is clearly visible")
        print("🚫 Remove any balls or objects from the surface")
        print("\nPress ENTER when ready...")
        input()
        
        corner_samples = []
        images_saved = []
        
        try:
            for i in range(num_samples):
                print(f"\n📸 Capturing sample {i+1}/{num_samples}...")
                
                # Get frame with timeout
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    print(f"   ⚠️ Frame {i+1} failed - no color data")
                    continue
                
                # Convert to numpy array
                color_image = np.asanyarray(color_frame.get_data())
                
                # Detect corners
                corners = self.detect_table_corners(color_image)
                
                if corners is not None:
                    corner_samples.append(corners)
                    
                    # Save the frame for reference
                    timestamp = int(time.time() * 1000)
                    frame_filename = f"{self.output_dir}/calibration_frame_{i+1}_{timestamp}.jpg"
                    
                    # Draw corners on image for visualization
                    debug_image = color_image.copy()
                    corners_int = corners.astype(np.int32)
                    cv2.polylines(debug_image, [corners_int], True, (0, 255, 0), 3)
                    
                    # Number the corners
                    for j, corner in enumerate(corners):
                        x, y = int(corner[0]), int(corner[1])
                        cv2.circle(debug_image, (x, y), 8, (0, 255, 255), -1)
                        cv2.putText(debug_image, f"{j+1}", (x-10, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    cv2.imwrite(frame_filename, debug_image)
                    images_saved.append(frame_filename)
                    
                    print(f"   ✅ Sample {i+1} captured successfully")
                    print(f"   📁 Saved: {frame_filename}")
                    
                    # Show corner coordinates
                    for j, corner in enumerate(corners):
                        print(f"      Corner {j+1}: ({corner[0]:.1f}, {corner[1]:.1f})")
                
                else:
                    print(f"   ❌ Sample {i+1} failed - no table detected")
                    
                    # Save failed frame for debugging
                    timestamp = int(time.time() * 1000)
                    failed_filename = f"{self.output_dir}/failed_frame_{i+1}_{timestamp}.jpg"
                    cv2.imwrite(failed_filename, color_image)
                    print(f"   📁 Saved failed frame: {failed_filename}")
                
                # Small delay between samples
                time.sleep(0.5)
            
            # Process results
            if len(corner_samples) >= num_samples // 2:
                # Average the corner positions
                self.table_corners_pixels = np.mean(corner_samples, axis=0)
                
                print(f"\n✅ Calibration Successful!")
                print(f"📊 Used {len(corner_samples)}/{num_samples} valid samples")
                print("📐 Average corner positions:")
                
                for i, corner in enumerate(self.table_corners_pixels):
                    print(f"   Corner {i+1}: ({corner[0]:.2f}, {corner[1]:.2f})")
                
                # Save calibration data
                self.save_calibration_data(images_saved, len(corner_samples))
                
                return True
            else:
                print(f"\n❌ Calibration Failed!")
                print(f"📊 Only {len(corner_samples)}/{num_samples} valid samples")
                print("💡 Try:")
                print("   - Better lighting")
                print("   - Clearer table edges")
                print("   - Remove background clutter")
                
                return False
                
        except Exception as e:
            print(f"❌ Calibration error: {e}")
            return False
    
    def save_calibration_data(self, images_saved: list, valid_samples: int):
        """Save calibration results to file"""
        timestamp = int(time.time())
        
        calibration_data = {
            "timestamp": timestamp,
            "table_width_meters": self.table_width,   # 22cm
            "table_height_meters": self.table_height, # 18cm
            "valid_samples": valid_samples,
            "corner_pixels": self.table_corners_pixels.tolist(),
            "images_saved": images_saved,
            "camera_resolution": [640, 480]
        }
        
        # Save as JSON
        json_filename = f"{self.output_dir}/calibration_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        print(f"💾 Calibration data saved: {json_filename}")
        
        # Save as numpy array for easy loading
        npy_filename = f"{self.output_dir}/table_corners_{timestamp}.npy"
        np.save(npy_filename, self.table_corners_pixels)
        print(f"💾 Corner array saved: {npy_filename}")
    
    def cleanup(self):
        """Clean up camera resources"""
        if self.pipeline:
            try:
                self.pipeline.stop()
                print("🛑 Camera stopped")
            except:
                pass


def main():
    """Main calibration test"""
    print("🎯 Camera Calibration Test")
    print("=========================")
    print("This will calibrate your camera to detect table corners.")
    print("Make sure your table (or mousemat for testing) is visible.\n")
    
    calibrator = CameraCalibrationTest()
    
    try:
        if calibrator.initialize_camera():
            print("📋 Calibration options:")
            print("  1. Quick test (5 samples)")
            print("  2. Standard calibration (10 samples)")
            print("  3. High precision (20 samples)")
            
            choice = input("\nEnter choice (1-3): ").strip()
            
            if choice == "1":
                samples = 5
            elif choice == "2":
                samples = 10
            elif choice == "3":
                samples = 20
            else:
                print("Invalid choice, using 10 samples")
                samples = 10
            
            success = calibrator.run_calibration(samples)
            
            if success:
                print("\n🎉 Calibration completed successfully!")
                print("📁 Check the 'calibration_data' folder for results")
                print("🚀 You can now use the full camera interface")
            else:
                print("\n❌ Calibration failed")
                print("📸 Check saved frames to see what went wrong")
        
        else:
            print("❌ Failed to initialize camera")
    
    except KeyboardInterrupt:
        print("\n🛑 Calibration cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    
    finally:
        calibrator.cleanup()


if __name__ == "__main__":
    main()

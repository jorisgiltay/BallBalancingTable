"""
Camera Calibration Test - Color Marker Version

This script tests the color marker calibration process.
Much simpler and more reliable than ArUco markers!

Setup Requirements:
- 35x35cm wooden base plate
- 4x4cm colored markers (Red, Green, Blue, Yellow) placed at corners
- Print markers using: python generate_color_markers.py

Usage: python camera_calibration_color_test.py
"""

import cv2
import numpy as np
import time
import os
import json
from typing import Optional, Tuple
import pyrealsense2 as rs


class ColorCalibrationTest:
    """
    Color marker calibration test for static base plate
    """
    
    def __init__(self):
        self.pipeline = None
        self.config = None
        self.table_corners_pixels = None
        
        # Base plate and table dimensions
        self.base_plate_size = 0.35  # 35cm x 35cm base plate
        self.table_size = 0.25       # 25cm x 25cm tilting table
        self.marker_size = 0.04      # 4cm x 4cm colored markers
        
        # Color ranges for marker detection (HSV)
        self.color_ranges = {
            0: {"name": "Red", "lower": np.array([0, 120, 70]), "upper": np.array([10, 255, 255])},      # Red
            1: {"name": "Green", "lower": np.array([35, 120, 70]), "upper": np.array([85, 255, 255])},   # Green
            2: {"name": "Blue", "lower": np.array([100, 120, 70]), "upper": np.array([130, 255, 255])},  # Blue
            3: {"name": "Yellow", "lower": np.array([20, 120, 70]), "upper": np.array([30, 255, 255])}   # Yellow
        }
        
        # Alternative red range for red markers (red wraps around in HSV)
        self.red_alt_range = {"lower": np.array([170, 120, 70]), "upper": np.array([180, 255, 255])}
        
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
    
    def detect_colored_markers(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect colored markers on static base plate
        
        Args:
            image: Input color image
            
        Returns:
            Array of 4 corner points in pixel coordinates, or None if not found
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Apply Gaussian blur to reduce noise
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
        
        marker_centers = {}
        debug_dir = "debug_frames"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        
        # Save original HSV for debugging
        cv2.imwrite(f"{debug_dir}/hsv_image.jpg", hsv)
        
        for marker_id, color_info in self.color_ranges.items():
            color_name = color_info["name"]
            
            # Create mask for this color
            mask = cv2.inRange(hsv, color_info["lower"], color_info["upper"])
            
            # Special handling for red (which wraps around in HSV)
            if marker_id == 0:  # Red marker
                mask_alt = cv2.inRange(hsv, self.red_alt_range["lower"], self.red_alt_range["upper"])
                mask = cv2.bitwise_or(mask, mask_alt)
            
            # Clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Save debug mask
            cv2.imwrite(f"{debug_dir}/mask_{color_name.lower()}.jpg", mask)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest contour (should be our marker)
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                # Check if the area is reasonable for a 4cm marker
                if area > 100:  # Minimum area threshold
                    # Calculate the center of the contour
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        center_x = int(M["m10"] / M["m00"])
                        center_y = int(M["m01"] / M["m00"])
                        marker_centers[marker_id] = [center_x, center_y]
                        
                        print(f"   ✅ Found {color_name} marker at ({center_x}, {center_y})")
                    else:
                        print(f"   ⚠️ {color_name} marker found but center calculation failed")
                else:
                    print(f"   ⚠️ {color_name} marker too small (area: {area})")
            else:
                print(f"   ❌ No {color_name} marker detected")
        
        # Check if we have all 4 expected markers
        if len(marker_centers) != 4:
            missing = []
            for i in range(4):
                if i not in marker_centers:
                    missing.append(self.color_ranges[i]["name"])
            print(f"⚠️ Missing colored markers: {missing}")
            print(f"💡 Debug images saved to '{debug_dir}/' - check color masks")
            return None
        
        # Arrange corners in consistent order: [0, 1, 2, 3] -> [Red, Green, Blue, Yellow]
        corner_points = np.array([
            marker_centers[0],  # Red (Top-left)
            marker_centers[1],  # Green (Top-right)  
            marker_centers[3],  # Yellow (Bottom-left)
            marker_centers[2]   # Blue (Bottom-right)
        ], dtype=np.float32)
        
        print(f"✅ Detected all 4 colored markers!")
        return corner_points
    
    def preview_detection(self):
        """Live preview to help position camera and markers"""
        if not self.pipeline:
            print("❌ Camera not initialized")
            return
        
        print("\n👁️ Live Color Detection Preview")
        print("=" * 30)
        print("📋 Position your camera to see all 4 colored markers")
        print("🌈 Red=Top-Left, Green=Top-Right, Yellow=Bottom-Left, Blue=Bottom-Right")
        print("💡 Green rectangles = detected markers")
        print("❌ Press ESC or 'q' to exit preview")
        print("\nStarting preview...")
        
        try:
            while True:
                # Get frame
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    continue
                
                color_image = np.asanyarray(color_frame.get_data())
                
                # Flip image 180 degrees for ball balancing table setup
                color_image = cv2.rotate(color_image, cv2.ROTATE_180)
                
                preview_image = color_image.copy()
                
                # Try to detect markers
                corners = self.detect_colored_markers(color_image)
                
                if corners is not None:
                    # Draw detected corners
                    corners_int = corners.astype(np.int32)
                    cv2.polylines(preview_image, [corners_int], True, (0, 255, 0), 3)
                    
                    # Number and color the corners
                    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]  # BGR
                    names = ["Red", "Green", "Blue", "Yellow"]
                    
                    for i, corner in enumerate(corners):
                        x, y = int(corner[0]), int(corner[1])
                        cv2.circle(preview_image, (x, y), 12, colors[i], -1)
                        cv2.circle(preview_image, (x, y), 12, (255, 255, 255), 2)
                        cv2.putText(preview_image, names[i], (x-30, y-20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Status text
                    cv2.putText(preview_image, "✓ All 4 colored markers detected", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    # Status text
                    cv2.putText(preview_image, "✗ Markers not detected", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(preview_image, "Check marker colors and lighting", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Show the frame
                cv2.imshow('Color Detection Preview', preview_image)
                
                # Check for exit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                    
        except Exception as e:
            print(f"❌ Preview error: {e}")
        
        finally:
            cv2.destroyAllWindows()
            print("📺 Preview window closed")

    def run_calibration(self, num_samples: int = 10) -> bool:
        """Run table calibration process"""
        if not self.pipeline:
            print("❌ Camera not initialized")
            return False
        
        print(f"\n🎨 Starting Color Marker Calibration")
        print("=" * 40)
        print(f"📊 Will take {num_samples} samples")
        print("📋 Make sure all 4 colored markers are clearly visible")
        print("🌈 Red=Top-Left, Green=Top-Right, Yellow=Bottom-Left, Blue=Bottom-Right")
        print("🏗️ Ensure 35x35cm base plate with 4x4cm colored markers is in view")
        print("🚫 Remove any balls or objects that might block the markers")
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
                
                # Flip image 180 degrees for ball balancing table setup
                color_image = cv2.rotate(color_image, cv2.ROTATE_180)
                
                # Detect corners
                corners = self.detect_colored_markers(color_image)
                
                if corners is not None:
                    corner_samples.append(corners)
                    
                    # Save the frame for reference
                    timestamp = int(time.time() * 1000)
                    frame_filename = f"{self.output_dir}/calibration_frame_{i+1}_{timestamp}.jpg"
                    
                    # Draw corners on image for visualization
                    debug_image = color_image.copy()
                    corners_int = corners.astype(np.int32)
                    cv2.polylines(debug_image, [corners_int], True, (0, 255, 0), 3)
                    
                    # Number and color the corners
                    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]  # BGR
                    names = ["Red", "Green", "Blue", "Yellow"]
                    
                    for j, corner in enumerate(corners):
                        x, y = int(corner[0]), int(corner[1])
                        cv2.circle(debug_image, (x, y), 12, colors[j], -1)
                        cv2.circle(debug_image, (x, y), 12, (255, 255, 255), 2)
                        cv2.putText(debug_image, f"{j+1}-{names[j]}", (x-40, y-20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    cv2.imwrite(frame_filename, debug_image)
                    images_saved.append(frame_filename)
                    
                    print(f"   ✅ Sample {i+1} captured successfully")
                    print(f"   📁 Saved: {frame_filename}")
                    
                    # Show corner coordinates
                    for j, corner in enumerate(corners):
                        print(f"      {names[j]}: ({corner[0]:.1f}, {corner[1]:.1f})")
                
                else:
                    print(f"   ❌ Sample {i+1} failed - not all markers detected")
                    
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
                
                names = ["Red", "Green", "Blue", "Yellow"]
                for i, corner in enumerate(self.table_corners_pixels):
                    print(f"   {names[i]}: ({corner[0]:.2f}, {corner[1]:.2f})")
                
                # Save calibration data
                self.save_calibration_data(images_saved, len(corner_samples))
                
                return True
            else:
                print(f"\n❌ Calibration Failed!")
                print(f"📊 Only {len(corner_samples)}/{num_samples} valid samples")
                print("💡 Try:")
                print("   - Better lighting")
                print("   - More saturated colored markers")
                print("   - Remove similar colored objects from background")
                
                return False
                
        except Exception as e:
            print(f"❌ Calibration error: {e}")
            return False
    
    def save_calibration_data(self, images_saved: list, valid_samples: int):
        """Save color calibration results to file"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        calibration_data = {
            "timestamp": timestamp,
            "calibration_type": "color_markers",
            "marker_colors": ["Red", "Green", "Blue", "Yellow"],
            "marker_ids": [0, 1, 2, 3],
            "base_plate_size_cm": 35,      # 35x35cm base plate
            "table_size_cm": 25,           # 25x25cm tilting table
            "marker_size_cm": 4,           # 4x4cm colored markers
            "valid_samples": valid_samples,
            "corner_pixels": self.table_corners_pixels.tolist(),
            "images_saved": images_saved,
            "camera_resolution": [640, 480],
            "image_orientation": "flipped_180_degrees",
            "color_ranges": {str(k): {"name": v["name"], "lower": v["lower"].tolist(), "upper": v["upper"].tolist()} 
                           for k, v in self.color_ranges.items()}
        }
        
        # Save as JSON
        json_filename = f"{self.output_dir}/color_calibration_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        print(f"💾 Color calibration data saved: {json_filename}")
        
        # Save as numpy array for easy loading
        npy_filename = f"{self.output_dir}/color_corners_{timestamp}.npy"
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
    print("🎨 Color Marker Calibration Test")
    print("===============================")
    print("This will calibrate your camera to detect colored markers.")
    print("Make sure your base plate with 4 colored markers is visible.\n")
    
    calibrator = ColorCalibrationTest()
    
    try:
        if calibrator.initialize_camera():
            print("📋 Options:")
            print("  p. Preview detection (check marker positioning)")
            print("  1. Quick test (5 samples)")
            print("  2. Standard calibration (10 samples)")
            print("  3. High precision (20 samples)")
            
            choice = input("\nEnter choice (p/1-3): ").strip().lower()
            
            if choice == "p":
                calibrator.preview_detection()
                return
            elif choice == "1":
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
                print("📸 Check saved frames and debug_frames/ to see what went wrong")
        
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

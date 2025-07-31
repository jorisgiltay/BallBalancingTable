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
        
        # All blue markers - detect by position instead of color!
        # Wider blue range to handle different lighting conditions (blinds up/down, etc.)
        self.blue_range = {
            "name": "Blue", 
            "lower": np.array([90, 50, 50]),    # Much wider range for lighting changes
            "upper": np.array([130, 255, 255])
        }
        
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
        Smart blue marker detection - infer position from pixel coordinates
        All markers are blue, but we know the expected layout!
        
        Args:
            image: Input color image
            
        Returns:
            Array of 4 corner points in pixel coordinates, or None if not found
        """
        # Adaptive preprocessing based on image brightness
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        
        if avg_brightness < 80:  # Dark image (blinds closed)
            enhanced = cv2.convertScaleAbs(image, alpha=1.3, beta=25)
        elif avg_brightness > 150:  # Bright image (blinds open)
            enhanced = cv2.convertScaleAbs(image, alpha=0.9, beta=5)
        else:  # Normal lighting
            enhanced = cv2.convertScaleAbs(image, alpha=1.1, beta=15)
        
        # Convert to HSV
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        
        # Light blur to reduce noise
        hsv = cv2.GaussianBlur(hsv, (3, 3), 0)
        
        debug_dir = "debug_frames"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        
        cv2.imwrite(f"{debug_dir}/enhanced.jpg", enhanced)
        cv2.imwrite(f"{debug_dir}/hsv.jpg", hsv)
        
        # Detect ALL blue markers
        mask = cv2.inRange(hsv, self.blue_range["lower"], self.blue_range["upper"])
        
        # Simple cleanup
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        cv2.imwrite(f"{debug_dir}/mask_blue_all.jpg", mask)
        
        # Find all blue contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        blue_markers = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 75:  # Lower threshold since we need all 4
                # Get center
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                    blue_markers.append([center_x, center_y, area])
                    print(f"   🔵 Blue marker: ({center_x}, {center_y}) area: {area:.0f}")
        
        if len(blue_markers) != 4:
            print(f"   ⚠️ Found {len(blue_markers)} blue markers, need exactly 4")
            return None
        
        # Sort markers by position to assign corners
        # Expected layout: Red=Top-Left, Green=Top-Right, Yellow=Bottom-Left, Blue=Bottom-Right
        markers = np.array(blue_markers)
        
        # Sort by Y coordinate first (top vs bottom)
        top_markers = markers[markers[:, 1] < np.median(markers[:, 1])]
        bottom_markers = markers[markers[:, 1] >= np.median(markers[:, 1])]
        
        # Sort top markers by X (left to right) -> Red(TL), Green(TR)
        top_sorted = top_markers[np.argsort(top_markers[:, 0])]
        red_marker = top_sorted[0][:2]    # Top-left
        green_marker = top_sorted[1][:2]  # Top-right
        
        # Sort bottom markers by X (left to right) -> Yellow(BL), Blue(BR)
        bottom_sorted = bottom_markers[np.argsort(bottom_markers[:, 0])]
        yellow_marker = bottom_sorted[0][:2]  # Bottom-left
        blue_marker = bottom_sorted[1][:2]    # Bottom-right
        
        print(f"   📍 Assigned positions:")
        print(f"      Red (TL): ({red_marker[0]:.0f}, {red_marker[1]:.0f})")
        print(f"      Green (TR): ({green_marker[0]:.0f}, {green_marker[1]:.0f})")
        print(f"      Yellow (BL): ({yellow_marker[0]:.0f}, {yellow_marker[1]:.0f})")
        print(f"      Blue (BR): ({blue_marker[0]:.0f}, {blue_marker[1]:.0f})")
        
        # Arrange corners: [Red, Green, Yellow, Blue] -> [TL, TR, BL, BR]
        corner_points = np.array([
            red_marker,    # Red (Top-left)
            green_marker,  # Green (Top-right)  
            yellow_marker, # Yellow (Bottom-left)
            blue_marker    # Blue (Bottom-right)
        ], dtype=np.float32)
        
        print(f"✅ Found all 4 blue markers and assigned positions!")
        return corner_points
    
    def preview_detection(self):
        """Live preview to help position camera and markers"""
        if not self.pipeline:
            print("❌ Camera not initialized")
            return
        
        print("\n👁️ Live Blue Marker Detection Preview")
        print("=" * 40)
        print("📋 Position your camera to see all 4 BLUE markers")
        print("🔵 All markers should be BLUE - position determines identity:")
        print("   Top-Left=Red, Top-Right=Green, Bottom-Left=Yellow, Bottom-Right=Blue")
        print("💡 Green rectangles = detected markers with assigned positions")
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
                    # Draw detected corners as a proper rectangle
                    # Reorder corners for proper rectangle drawing: [TL, TR, BR, BL]
                    # Current order is [TL, TR, BL, BR], need to swap BL and BR
                    corners_rect = np.array([
                        corners[0],  # TL (Top-Left)
                        corners[1],  # TR (Top-Right)  
                        corners[3],  # BR (Bottom-Right)
                        corners[2]   # BL (Bottom-Left)
                    ])
                    corners_int = corners_rect.astype(np.int32)
                    cv2.polylines(preview_image, [corners_int], True, (0, 255, 0), 3)
                    
                    # Position-based labels and colors
                    colors = [(255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255)]  # All white circles
                    position_names = ["TL", "TR", "BL", "BR"]  # Simple position labels
                    
                    for i, corner in enumerate(corners):
                        x, y = int(corner[0]), int(corner[1])
                        cv2.circle(preview_image, (x, y), 12, colors[i], -1)
                        cv2.circle(preview_image, (x, y), 12, (0, 255, 0), 2)  # Green border
                        cv2.putText(preview_image, position_names[i], (x-15, y-20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Status text
                    cv2.putText(preview_image, "✓ All 4 blue markers detected", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    # Status text
                    cv2.putText(preview_image, "✗ Markers not detected", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(preview_image, "Check blue markers and lighting", (10, 60), 
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
        
        print(f"\n🔵 Starting Blue Marker Calibration")
        print("=" * 40)
        print(f"📊 Will take {num_samples} samples")
        print("📋 Make sure all 4 BLUE markers are clearly visible")
        print("🔵 Position determines identity: TL=Red, TR=Green, BL=Yellow, BR=Blue")
        print("🏗️ Ensure 35x35cm base plate with 4x4cm BLUE markers is in view")
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
                    
                    # Position-based labels for saved images
                    colors = [(255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255)]  # All white circles
                    position_names = ["TL", "TR", "BL", "BR"]
                    
                    for j, corner in enumerate(corners):
                        x, y = int(corner[0]), int(corner[1])
                        cv2.circle(debug_image, (x, y), 12, colors[j], -1)
                        cv2.circle(debug_image, (x, y), 12, (0, 255, 0), 2)
                        cv2.putText(debug_image, f"{j+1}-{position_names[j]}", (x-25, y-20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    cv2.imwrite(frame_filename, debug_image)
                    images_saved.append(frame_filename)
                    
                    print(f"   ✅ Sample {i+1} captured successfully")
                    print(f"   📁 Saved: {frame_filename}")
                    
                    # Show corner coordinates
                    position_names = ["TL", "TR", "BL", "BR"]
                    for j, corner in enumerate(corners):
                        print(f"      {position_names[j]}: ({corner[0]:.1f}, {corner[1]:.1f})")
                
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
                
                position_names = ["TL", "TR", "BL", "BR"]
                for i, corner in enumerate(self.table_corners_pixels):
                    print(f"   {position_names[i]}: ({corner[0]:.2f}, {corner[1]:.2f})")
                
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
            "calibration_type": "blue_markers_position_based",
            "marker_colors": ["Blue", "Blue", "Blue", "Blue"],  # All blue!
            "marker_positions": ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"],
            "marker_ids": [0, 1, 2, 3],
            "base_plate_size_cm": 35,      # 35x35cm base plate
            "table_size_cm": 25,           # 25x25cm tilting table
            "marker_size_cm": 4,           # 4x4cm colored markers
            "valid_samples": valid_samples,
            "corner_pixels": self.table_corners_pixels.tolist(),
            "images_saved": images_saved,
            "camera_resolution": [640, 480],
            "image_orientation": "flipped_180_degrees",
            "color_ranges": {"blue_only": {"name": self.blue_range["name"], 
                                          "lower": self.blue_range["lower"].tolist(), 
                                          "upper": self.blue_range["upper"].tolist()}}
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
    print("🔵 Blue Marker Position-Based Calibration")
    print("=========================================")
    print("This will calibrate using 4 BLUE markers - position determines identity!")
    print("Make sure your base plate with 4 blue markers is visible.\n")
    
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

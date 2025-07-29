"""
Ball Detection Test

This script tests ball detection on your calibrated table.
Place a white ping pong ball on your table/mousemat and see if it gets detected.

Usage: python ball_detection_test.py
"""

import cv2
import numpy as np
import time
import os
import json
from typing import Optional, Tuple
import pyrealsense2 as rs


class BallDetectionTest:
    """Test ball detection with calibrated camera"""
    
    def __init__(self):
        self.pipeline = None
        self.config = None
        self.table_corners_pixels = None
        # Mousemat dimensions (18cm x 22cm)
        self.table_width = 0.22   # 22cm width
        self.table_height = 0.18  # 18cm height
        
        # Ball detector settings - More flexible for stationary detection
        self.lower_white = np.array([0, 0, 180])    # Lower brightness threshold
        self.upper_white = np.array([180, 50, 255]) # Higher saturation tolerance
        self.min_contour_area = 50                  # Smaller minimum area
        self.max_contour_area = 8000               # Larger maximum area
        self.circularity_threshold = 0.6           # More lenient circularity
        
        # Zoom/crop settings
        self.use_crop = True  # Enable cropping to zoom in on table
        self.crop_margin = 50  # Extra pixels around table corners
        self.show_crop_view = False  # Show separate crop window
        self.show_debug_mask = False  # Show HSV mask for debugging
        
        # Load calibration data
        self.load_latest_calibration()
    
    def load_latest_calibration(self):
        """Load the most recent calibration data"""
        calib_dir = "calibration_data"
        if not os.path.exists(calib_dir):
            print("❌ No calibration data found. Run camera_calibration_test.py first.")
            return False
        
        # Find latest calibration file
        json_files = [f for f in os.listdir(calib_dir) if f.startswith("calibration_") and f.endswith(".json")]
        
        if not json_files:
            print("❌ No calibration files found.")
            return False
        
        # Sort by timestamp (filename contains timestamp)
        latest_file = sorted(json_files)[-1]
        json_path = os.path.join(calib_dir, latest_file)
        
        try:
            with open(json_path, 'r') as f:
                calib_data = json.load(f)
            
            self.table_corners_pixels = np.array(calib_data["corner_pixels"], dtype=np.float32)
            
            # Flip calibration corners to match flipped image orientation
            # When image is rotated 180°, coordinates transform: (x,y) -> (640-x, 480-y)
            self.table_corners_pixels[:, 0] = 640 - self.table_corners_pixels[:, 0]  # Flip X
            self.table_corners_pixels[:, 1] = 480 - self.table_corners_pixels[:, 1]  # Flip Y
            
            print(f"✅ Loaded calibration from: {latest_file}")
            print(f"📐 Table corners loaded and flipped: {self.table_corners_pixels.shape}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load calibration: {e}")
            return False
    
    def initialize_camera(self) -> bool:
        """Initialize camera"""
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            
            self.pipeline.start(self.config)
            print("✅ Camera initialized for ball detection")
            return True
            
        except Exception as e:
            print(f"❌ Camera initialization failed: {e}")
            return False
    
    def get_crop_bounds(self) -> tuple:
        """Calculate crop bounds to zoom in on table area"""
        if self.table_corners_pixels is None:
            return None
        
        # Find bounding box of table corners
        min_x = int(np.min(self.table_corners_pixels[:, 0]) - self.crop_margin)
        max_x = int(np.max(self.table_corners_pixels[:, 0]) + self.crop_margin)
        min_y = int(np.min(self.table_corners_pixels[:, 1]) - self.crop_margin)
        max_y = int(np.max(self.table_corners_pixels[:, 1]) + self.crop_margin)
        
        # Ensure bounds are within image
        min_x = max(0, min_x)
        max_x = min(640, max_x)
        min_y = max(0, min_y)
        max_y = min(480, max_y)
        
        return (min_x, min_y, max_x, max_y)
    
    def crop_image(self, image: np.ndarray) -> tuple:
        """Crop image to focus on table area"""
        if not self.use_crop:
            return image, (0, 0)
        
        crop_bounds = self.get_crop_bounds()
        if crop_bounds is None:
            return image, (0, 0)
        
        min_x, min_y, max_x, max_y = crop_bounds
        cropped = image[min_y:max_y, min_x:max_x]
        
        return cropped, (min_x, min_y)
    
    def adjust_coordinates_for_crop(self, pixel_x: float, pixel_y: float, crop_offset: tuple) -> tuple:
        """Adjust pixel coordinates to account for cropping"""
        offset_x, offset_y = crop_offset
        return pixel_x + offset_x, pixel_y + offset_y
    
    def detect_ball(self, color_frame: np.ndarray, crop_offset: tuple = (0, 0)) -> Optional[Tuple[float, float]]:
        # Convert to HSV
        hsv = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for white ball
        mask = cv2.inRange(hsv, self.lower_white, self.upper_white)
        
        # Store original mask for debugging
        self.debug_mask = mask.copy()
        
        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Filter by area and circularity
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_contour_area < area < self.max_contour_area:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > self.circularity_threshold:
                        valid_contours.append((contour, area))
        
        if not valid_contours:
            return None
        
        # Use largest valid contour
        best_contour = max(valid_contours, key=lambda x: x[1])[0]
        
        # Get centroid
        M = cv2.moments(best_contour)
        if M["m00"] == 0:
            return None
        
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        
        # Adjust coordinates for crop offset
        cx_adjusted, cy_adjusted = self.adjust_coordinates_for_crop(cx, cy, crop_offset)
        
        return (cx_adjusted, cy_adjusted)
    
    def pixel_to_world_coordinates(self, pixel_x: float, pixel_y: float) -> Tuple[float, float]:
        """Convert pixel coordinates to world coordinates using calibration"""
        if self.table_corners_pixels is None:
            # Fallback calculation for rectangular mousemat
            world_x = (pixel_x - 320) / 320 * (self.table_width / 2)   # ±11cm
            world_y = (pixel_y - 240) / 240 * (self.table_height / 2)  # ±9cm
            return world_x, world_y
        
        try:
            # Define world coordinates for rectangular mousemat (22cm x 18cm)
            table_corners_world = np.array([
                [-self.table_width/2, -self.table_height/2],   # Top-left: (-11cm, -9cm)
                [self.table_width/2, -self.table_height/2],    # Top-right: (+11cm, -9cm)
                [self.table_width/2, self.table_height/2],     # Bottom-right: (+11cm, +9cm)
                [-self.table_width/2, self.table_height/2]     # Bottom-left: (-11cm, +9cm)
            ], dtype=np.float32)
            
            # Create perspective transformation matrix
            pixel_point = np.array([[pixel_x, pixel_y]], dtype=np.float32)
            
            M = cv2.getPerspectiveTransform(self.table_corners_pixels, table_corners_world)
            world_point = cv2.perspectiveTransform(pixel_point.reshape(1, 1, 2), M)
            
            return float(world_point[0, 0, 0]), float(world_point[0, 0, 1])
            
        except Exception as e:
            print(f"⚠️ Coordinate transformation error: {e}")
            return 0.0, 0.0
    
    def run_ball_detection_test(self):
        """Run interactive ball detection test"""
        if self.table_corners_pixels is None:
            print("❌ No calibration data loaded")
            return
        
        if not self.pipeline:
            print("❌ Camera not initialized")
            return
        
        print("\n🏓 Ball Detection Test")
        print("=" * 30)
        print("📋 Instructions:")
        print("  1. Place a WHITE ping pong ball on your table/mousemat")
        print("  2. Move it around to test detection")
        print("  3. Watch the live coordinates")
        print("  4. Press 'q' to quit, 's' to save frame")
        print("  5. Press 'z' to toggle zoom/crop mode")
        print("  6. Press 'v' to toggle crop view window")
        print("  7. Press '+'/'-' to adjust crop margin")
        print("  8. Press 'm' to toggle HSV mask debug view")
        print(f"\n🔍 Crop mode: {'ENABLED' if self.use_crop else 'DISABLED'}")
        print(f"📏 Crop margin: {self.crop_margin}px")
        print(f"🎯 HSV range: {self.lower_white} to {self.upper_white}")
        print("📝 Note: Image and calibration are automatically flipped 180° for correct orientation")
        print("\nStarting in 3 seconds...")
        time.sleep(3)
        
        frame_count = 0
        detection_count = 0
        
        try:
            while True:  # Run indefinitely until 'q' is pressed
                # Get frame
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    continue
                
                color_image = np.asanyarray(color_frame.get_data())
                
                # Always flip image 180 degrees for correct camera orientation
                color_image = cv2.rotate(color_image, cv2.ROTATE_180)
                
                display_image = color_image.copy()
                
                # Crop image to focus on table area
                cropped_image, crop_offset = self.crop_image(color_image)
                
                # Draw table outline on full image
                if self.table_corners_pixels is not None:
                    corners_int = self.table_corners_pixels.astype(np.int32)
                    cv2.polylines(display_image, [corners_int], True, (0, 255, 0), 2)
                
                # Detect ball on cropped image
                ball_pos = self.detect_ball(cropped_image, crop_offset)
                
                if ball_pos:
                    pixel_x, pixel_y = ball_pos
                    world_x, world_y = self.pixel_to_world_coordinates(pixel_x, pixel_y)
                    
                    detection_count += 1
                    
                    # Draw ball position
                    cv2.circle(display_image, (int(pixel_x), int(pixel_y)), 15, (0, 0, 255), 3)
                    cv2.circle(display_image, (int(pixel_x), int(pixel_y)), 5, (255, 255, 255), -1)
                    
                    # Show coordinates
                    coord_text = f"Pixel: ({pixel_x:.0f}, {pixel_y:.0f})"
                    world_text = f"World: ({world_x:.3f}, {world_y:.3f})m"
                    
                    cv2.putText(display_image, coord_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(display_image, world_text, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Console output
                    if frame_count % 10 == 0:  # Every 10 frames
                        print(f"🎯 Ball at: ({world_x:.3f}, {world_y:.3f})m | Range: X±{self.table_width/2:.3f}m, Y±{self.table_height/2:.3f}m")
                
                else:
                    cv2.putText(display_image, "No ball detected", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Show crop info
                if self.use_crop:
                    crop_bounds = self.get_crop_bounds()
                    if crop_bounds:
                        crop_info = f"Crop: {crop_bounds[2]-crop_bounds[0]}x{crop_bounds[3]-crop_bounds[1]}px (margin: {self.crop_margin}px)"
                        cv2.putText(display_image, crop_info, (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Show detection stats
                detection_rate = (detection_count / max(frame_count, 1)) * 100
                stats_text = f"Detection rate: {detection_rate:.1f}% ({detection_count}/{frame_count})"
                cv2.putText(display_image, stats_text, (10, display_image.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show frame
                cv2.imshow('Ball Detection Test', display_image)
                
                # Show cropped view if enabled
                if self.show_crop_view and self.use_crop:
                    if cropped_image.size > 0:  # Make sure cropped image is valid
                        # Scale up cropped image for better visibility
                        scale_factor = min(400 / cropped_image.shape[1], 300 / cropped_image.shape[0])
                        new_width = int(cropped_image.shape[1] * scale_factor)
                        new_height = int(cropped_image.shape[0] * scale_factor)
                        
                        if new_width > 0 and new_height > 0:
                            scaled_crop = cv2.resize(cropped_image, (new_width, new_height))
                            
                            # Draw detection info on crop view
                            if ball_pos:
                                # Scale ball position to crop coordinates
                                crop_ball_x = (pixel_x - crop_offset[0]) * scale_factor
                                crop_ball_y = (pixel_y - crop_offset[1]) * scale_factor
                                cv2.circle(scaled_crop, (int(crop_ball_x), int(crop_ball_y)), 
                                         max(5, int(15 * scale_factor)), (0, 0, 255), 2)
                                cv2.circle(scaled_crop, (int(crop_ball_x), int(crop_ball_y)), 
                                         max(2, int(5 * scale_factor)), (255, 255, 255), -1)
                            
                            cv2.imshow('Cropped View - What Algorithm Sees', scaled_crop)
                
                # Show HSV mask debug view if enabled
                if self.show_debug_mask and hasattr(self, 'debug_mask'):
                    if self.use_crop and cropped_image.size > 0:
                        # Show mask of cropped area
                        crop_bounds = self.get_crop_bounds()
                        if crop_bounds:
                            min_x, min_y, max_x, max_y = crop_bounds
                            mask_crop = self.debug_mask[min_y:max_y, min_x:max_x]
                            if mask_crop.size > 0:
                                cv2.imshow('HSV Mask - White Detection', mask_crop)
                    else:
                        cv2.imshow('HSV Mask - White Detection', self.debug_mask)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and ball_pos:
                    timestamp = int(time.time() * 1000)
                    filename = f"ball_detection_{timestamp}.jpg"
                    cv2.imwrite(filename, display_image)
                    print(f"💾 Saved frame: {filename}")
                elif key == ord('z'):
                    self.use_crop = not self.use_crop
                    print(f"🔍 Crop mode: {'ENABLED' if self.use_crop else 'DISABLED'}")
                elif key == ord('v'):
                    self.show_crop_view = not self.show_crop_view
                    if not self.show_crop_view:
                        cv2.destroyWindow('Cropped View - What Algorithm Sees')
                    print(f"👁️ Crop view: {'ENABLED' if self.show_crop_view else 'DISABLED'}")
                elif key == ord('+') or key == ord('='):
                    self.crop_margin = min(100, self.crop_margin + 10)
                    print(f"📏 Crop margin increased to: {self.crop_margin}px")
                elif key == ord('-') or key == ord('_'):
                    self.crop_margin = max(10, self.crop_margin - 10)
                    print(f"📏 Crop margin decreased to: {self.crop_margin}px")
                elif key == ord('m'):
                    self.show_debug_mask = not self.show_debug_mask
                    if not self.show_debug_mask:
                        cv2.destroyWindow('HSV Mask - White Detection')
                    print(f"🎭 Debug mask: {'ENABLED' if self.show_debug_mask else 'DISABLED'}")
                    print(f"   HSV range: {self.lower_white} to {self.upper_white}")
                
                frame_count += 1
        
        except KeyboardInterrupt:
            print("\n🛑 Test interrupted")
        
        finally:
            cv2.destroyAllWindows()
            print(f"\n📊 Test Results:")
            print(f"   Frames processed: {frame_count}")
            print(f"   Ball detections: {detection_count}")
            print(f"   Detection rate: {(detection_count/max(frame_count,1))*100:.1f}%")
    
    def cleanup(self):
        """Clean up resources"""
        if self.pipeline:
            try:
                self.pipeline.stop()
            except:
                pass


def main():
    """Main test function"""
    print("🏓 Ball Detection Test")
    print("=====================")
    
    tester = BallDetectionTest()
    
    try:
        if tester.table_corners_pixels is None:
            print("❌ No calibration data found.")
            print("💡 Run 'python camera_calibration_test.py' first")
            return
        
        if tester.initialize_camera():
            print("🚀 Starting ball detection test...")
            tester.run_ball_detection_test()
        else:
            print("❌ Failed to initialize camera")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        tester.cleanup()


if __name__ == "__main__":
    main()

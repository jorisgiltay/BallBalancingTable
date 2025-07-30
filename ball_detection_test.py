"""
Ball Detection Test - Blue Marker Version

This script tests ball detection on your blue marker calibrated base plate.
Place a white ping pong ball on your table and see if it gets detected.

Setup Requirements:
- Run camera_calibration_color.py first to calibrate blue markers
- 35x35cm base plate with 4 blue markers at corners
- White ping pong ball on the tilting table surface

Usage: python ball_detection_test.py
"""

import cv2
import numpy as np
import time
import os
import json
from typing import Optional, Tuple
import pyrealsense2 as rs


class BlueMarkerBallDetectionTest:
    """Test ball detection with blue marker calibrated camera"""
    
    def __init__(self):
        self.pipeline = None
        self.config = None
        self.table_corners_pixels = None
        
        # Base plate and table dimensions  
        self.base_plate_size = 0.35  # 35cm x 35cm base plate
        self.table_size = 0.25       # 25cm x 25cm tilting table
        
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
        """Load the most recent blue marker calibration data"""
        calib_dir = "calibration_data"
        if not os.path.exists(calib_dir):
            print("❌ No calibration data found. Run camera_calibration_color.py first.")
            return False
        
        # Find latest color calibration file
        json_files = [f for f in os.listdir(calib_dir) if f.startswith("color_calibration_") and f.endswith(".json")]
        
        if not json_files:
            print("❌ No blue marker calibration files found.")
            return False
        
        # Sort by timestamp (filename contains timestamp)
        latest_file = sorted(json_files)[-1]
        json_path = os.path.join(calib_dir, latest_file)
        
        try:
            with open(json_path, 'r') as f:
                calib_data = json.load(f)
            
            self.table_corners_pixels = np.array(calib_data["corner_pixels"], dtype=np.float32)
            
            # Note: Blue marker calibration already accounts for 180° rotation during calibration
            # No coordinate flipping needed since calibration was done with rotated images
            
            print(f"✅ Loaded blue marker calibration from: {latest_file}")
            print(f"📐 Table corners loaded: {self.table_corners_pixels.shape}")
            print(f"🔵 Calibration type: {calib_data.get('calibration_type', 'unknown')}")
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
        
        # Adaptive thresholding based on image brightness
        gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        
        if avg_brightness > 150:  # Very bright/light background
            # Tighter thresholds for bright scenes - look for very white objects
            lower_white = np.array([0, 0, 200])     # Higher brightness threshold
            upper_white = np.array([180, 30, 255]) # Lower saturation tolerance
            print(f"🔆 Bright scene detected (avg: {avg_brightness:.0f}) - using tight thresholds")
        elif avg_brightness < 80:  # Dark scene
            # More lenient thresholds for dark scenes
            lower_white = np.array([0, 0, 150])     # Lower brightness threshold
            upper_white = np.array([180, 70, 255]) # Higher saturation tolerance
            print(f"🌙 Dark scene detected (avg: {avg_brightness:.0f}) - using lenient thresholds")
        else:  # Normal lighting
            # Default thresholds
            lower_white = self.lower_white
            upper_white = self.upper_white
        
        # Create mask for white ball
        mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # For bright backgrounds, also try edge-based detection
        if avg_brightness > 150:
            # Add edge detection to help with low contrast situations
            edges = cv2.Canny(gray, 50, 150)
            
            # Dilate edges to create thicker boundaries
            kernel_edge = np.ones((3, 3), np.uint8)
            edges_dilated = cv2.dilate(edges, kernel_edge, iterations=1)
            
            # Combine HSV mask with edge mask for better detection
            mask = cv2.bitwise_or(mask, edges_dilated)
        
        # Store original mask for debugging
        self.debug_mask = mask.copy()
        
        # Clean up mask - more aggressive cleaning for bright scenes
        if avg_brightness > 150:
            kernel = np.ones((7, 7), np.uint8)  # Larger kernel for bright scenes
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        else:
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Adaptive area thresholds based on brightness
        if avg_brightness > 150:
            # Stricter requirements for bright scenes
            min_area = 80   # Slightly larger minimum
            max_area = 6000 # Smaller maximum to avoid large bright areas
            circularity_thresh = 0.7  # Higher circularity requirement
        else:
            # Standard requirements
            min_area = self.min_contour_area
            max_area = self.max_contour_area
            circularity_thresh = self.circularity_threshold
        
        # Filter by area and circularity
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > circularity_thresh:
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
        """Convert pixel coordinates to world coordinates using blue marker calibration"""
        if self.table_corners_pixels is None:
            # Fallback calculation for square table
            world_x = (pixel_x - 320) / 320 * (self.table_size / 2)   # ±12.5cm
            world_y = (pixel_y - 240) / 240 * (self.table_size / 2)   # ±12.5cm
            return world_x, world_y

        try:
            # Load base plate size from calibration data
            calib_dir = "calibration_data"
            json_files = [f for f in os.listdir(calib_dir) if f.startswith("color_calibration_") and f.endswith(".json")]
            
            if json_files:
                latest_file = sorted(json_files)[-1]
                json_path = os.path.join(calib_dir, latest_file)
                
                with open(json_path, 'r') as f:
                    calib_data = json.load(f)
                
                # Use actual base plate size from calibration (35cm -> 0.35m)
                base_plate_size = calib_data.get("base_plate_size_cm", 35) / 100.0
            else:
                # Fallback to default
                base_plate_size = 0.35
            
            # Define world coordinates for actual base plate (35cm x 35cm)
            # Blue markers are 4x4cm squares placed at corners, so marker centers are 2cm inward from plate edges
            # Actual corners: ±17.5cm, marker centers: ±15.5cm from center
            marker_offset = 0.02  # 2cm offset from plate edge to marker center
            plate_edge = base_plate_size / 2  # 17.5cm from center to plate edge
            marker_center_distance = plate_edge - marker_offset  # 15.5cm from center to marker center
            
            table_corners_world = np.array([
                [-marker_center_distance, -marker_center_distance],   # Top-left marker center: (-15.5cm, -15.5cm)
                [marker_center_distance, -marker_center_distance],    # Top-right marker center: (+15.5cm, -15.5cm)
                [-marker_center_distance, marker_center_distance],    # Bottom-left marker center: (-15.5cm, +15.5cm)
                [marker_center_distance, marker_center_distance]      # Bottom-right marker center: (+15.5cm, +15.5cm)
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
        print("📝 Note: Image is automatically flipped 180° to match blue marker calibration")
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
                
                # Draw table outline on full image - show actual base plate edges, not marker centers
                if self.table_corners_pixels is not None:
                    try:
                        # Load calibration data to get accurate measurements
                        calib_dir = "calibration_data"
                        json_files = [f for f in os.listdir(calib_dir) if f.startswith("color_calibration_") and f.endswith(".json")]
                        
                        if json_files:
                            latest_file = sorted(json_files)[-1]
                            json_path = os.path.join(calib_dir, latest_file)
                            
                            with open(json_path, 'r') as f:
                                calib_data = json.load(f)
                            
                            # Simple approach: expand the marker center rectangle outward by 2cm equivalent pixels
                            # Calculate the center of the marker rectangle
                            center_x = np.mean(self.table_corners_pixels[:, 0])
                            center_y = np.mean(self.table_corners_pixels[:, 1])
                            
                            # Calculate expansion factor: need to go from 31cm (marker center to center) to 35cm (edge to edge)
                            # Current marker distance represents 31cm (35cm - 4cm), target is 35cm
                            expansion_factor = 35.0 / 31.0  # ≈1.129
                            
                            # Expand each corner outward from the center
                            plate_corners_pixel = []
                            for corner in self.table_corners_pixels:
                                # Vector from center to corner
                                dx = corner[0] - center_x
                                dy = corner[1] - center_y
                                
                                # Expand by factor
                                new_x = center_x + (dx * expansion_factor)
                                new_y = center_y + (dy * expansion_factor)
                                
                                plate_corners_pixel.append([new_x, new_y])
                            
                            plate_corners_pixel = np.array(plate_corners_pixel, dtype=np.float32)
                            
                            # Draw the expanded plate edge rectangle (green, thick)
                            corners_rect = np.array([
                                plate_corners_pixel[0],  # TL (Top-Left plate edge)
                                plate_corners_pixel[1],  # TR (Top-Right plate edge)  
                                plate_corners_pixel[3],  # BR (Bottom-Right plate edge)
                                plate_corners_pixel[2]   # BL (Bottom-Left plate edge)
                            ])
                            corners_int = corners_rect.astype(np.int32)
                            cv2.polylines(display_image, [corners_int], True, (0, 255, 0), 2)
                            
                            # Draw marker centers for reference (yellow, thin)
                            marker_corners_rect = np.array([
                                self.table_corners_pixels[0],  # TL marker center
                                self.table_corners_pixels[1],  # TR marker center
                                self.table_corners_pixels[3],  # BR marker center
                                self.table_corners_pixels[2]   # BL marker center
                            ])
                            marker_corners_int = marker_corners_rect.astype(np.int32)
                            cv2.polylines(display_image, [marker_corners_int], True, (0, 255, 255), 1)  # Yellow, thin line
                            
                        else:
                            # No calibration data - just draw marker centers
                            corners_rect = np.array([
                                self.table_corners_pixels[0],  # TL
                                self.table_corners_pixels[1],  # TR  
                                self.table_corners_pixels[3],  # BR
                                self.table_corners_pixels[2]   # BL
                            ])
                            corners_int = corners_rect.astype(np.int32)
                            cv2.polylines(display_image, [corners_int], True, (0, 255, 0), 2)
                            
                    except Exception as e:
                        print(f"⚠️ Error calculating plate edges: {e}")
                        # Fallback to marker centers
                        corners_rect = np.array([
                            self.table_corners_pixels[0],  # TL
                            self.table_corners_pixels[1],  # TR  
                            self.table_corners_pixels[3],  # BR
                            self.table_corners_pixels[2]   # BL
                        ])
                        corners_int = corners_rect.astype(np.int32)
                        cv2.polylines(display_image, [corners_int], True, (0, 255, 0), 2)
                
                # Detect ball on cropped image
                ball_pos = self.detect_ball(cropped_image, crop_offset)
                
                # Show adaptive lighting info
                if cropped_image.size > 0:
                    gray_crop = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                    crop_brightness = np.mean(gray_crop)
                    
                    if crop_brightness > 150:
                        lighting_status = f"🔆 Bright scene (avg: {crop_brightness:.0f}) - Enhanced detection"
                        status_color = (0, 255, 255)  # Yellow
                    elif crop_brightness < 80:
                        lighting_status = f"🌙 Dark scene (avg: {crop_brightness:.0f}) - Lenient detection"  
                        status_color = (255, 0, 255)  # Magenta
                    else:
                        lighting_status = f"💡 Normal lighting (avg: {crop_brightness:.0f})"
                        status_color = (255, 255, 255)  # White
                    
                    cv2.putText(display_image, lighting_status, (10, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
                
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
                        print(f"🎯 Ball at: ({world_x:.3f}, {world_y:.3f})m | Range: X±{self.table_size/2:.3f}m, Y±{self.table_size/2:.3f}m")
                
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
    print("🏓 Ball Detection Test - Blue Marker Version")
    print("============================================")
    
    tester = BlueMarkerBallDetectionTest()
    
    try:
        if tester.table_corners_pixels is None:
            print("❌ No blue marker calibration data found.")
            print("💡 Run 'python camera_calibration_color.py' first")
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

"""
PyRealSense Camera Interface for Ball Position Detection

This module provides real-time ball position detection using Intel RealSense cameras
for the ball balancing table project. It bridges the gap between simulation and
real-world hardware deployment.

Features:
- Real-time ball detection and tracking
- Coordinate transformation from camera to table space
- Color-based ball detection with HSV filtering
- Depth information for 3D position estimation
- Calibration utilities for camera-table alignment
- Compatible with existing simulation interface

Author: Ball Balancing Table Project
"""

import numpy as np
import cv2
import time
from typing import Tuple, Optional, Dict, Any
import threading
import queue
import os
import json

# Blue marker detection system - no ArUco needed


class BallDetector:
    """
    Ball detection using color filtering and contour detection
    """
    
    def __init__(self):
        # HSV color range for white ping pong ball detection
        # These values are tuned for better stationary ball detection
        self.lower_white = np.array([0, 0, 180])    # Lower brightness threshold
        self.upper_white = np.array([180, 50, 255]) # Higher saturation tolerance
        
        # Alternative: Orange ball detection (uncomment if using orange ball)
        # self.lower_orange = np.array([10, 100, 100])
        # self.upper_orange = np.array([25, 255, 255])
        
        # Contour filtering parameters - More flexible for better detection
        self.min_contour_area = 50       # Smaller minimum area
        self.max_contour_area = 8000     # Larger maximum area
        self.circularity_threshold = 0.6  # More lenient circularity
        
    def detect_ball(self, color_frame: np.ndarray, depth_frame: Optional[np.ndarray] = None, crop: Optional[Tuple[int, int, int, int]] = None) -> Optional[Tuple[float, float, float]]:
        """
        Detect ball position in the given frame, with optional cropping (ROI)
        
        Args:
            color_frame: RGB color image from camera
            depth_frame: Optional depth image for Z coordinate
            crop: Optional (x, y, w, h) tuple to crop the input frames before detection
        Returns:
            Tuple of (x, y, z) in pixels (relative to full image), or None if no ball detected
        """
        # Apply crop if specified
        x0, y0 = 0, 0
        if crop is not None:
            x0, y0, w, h = crop
            color_frame = color_frame[y0:y0+h, x0:x0+w]
            if depth_frame is not None:
                depth_frame = depth_frame[y0:y0+h, x0:x0+w]

        # Convert BGR to HSV for better color filtering
        hsv = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for white ball
        mask = cv2.inRange(hsv, self.lower_white, self.upper_white)
        
        # Morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Filter contours by area and circularity
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_contour_area < area < self.max_contour_area:
                # Check circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > self.circularity_threshold:
                        valid_contours.append((contour, area))
        
        if not valid_contours:
            return None
            
        # Use the largest valid contour
        best_contour = max(valid_contours, key=lambda x: x[1])[0]
        
        # Get centroid
        M = cv2.moments(best_contour)
        if M["m00"] == 0:
            return None
        cx = float(M["m10"] / M["m00"]) + x0
        cy = float(M["m01"] / M["m00"]) + y0
        
        # Get depth if available
        z = 0.0
        if depth_frame is not None and 0 <= cx-x0 < depth_frame.shape[1] and 0 <= cy-y0 < depth_frame.shape[0]:
            z = depth_frame[int(cy-y0), int(cx-x0)]
        return (cx, cy, z)


class RealSenseCameraInterface:
    """
    Intel RealSense camera interface for ball position detection
    """
    
    def __init__(self, table_size: float = 0.25, camera_height: float = 0.5):
        """
        Initialize RealSense camera interface
        
        Args:
            table_size: Size of the square table in meters (default 25cm)
            camera_height: Height of camera above table in meters
        """
        self.table_size = table_size
        self.camera_height = camera_height
        
        # Camera calibration parameters (to be determined during calibration)
        self.camera_matrix = None
        self.dist_coeffs = None
        self.table_corners_pixels = None  # Table corners in pixel coordinates
        self.table_corners_world = np.array([  # Table corners in world coordinates (meters)
            [-table_size/2, -table_size/2, 0],
            [table_size/2, -table_size/2, 0],
            [table_size/2, table_size/2, 0],
            [-table_size/2, table_size/2, 0]
        ], dtype=np.float32)
        
        # RealSense pipeline and configuration
        self.pipeline = None
        self.config = None
        self.ball_detector = BallDetector()
        
        # Threading for continuous capture
        self.capture_thread = None
        self.position_queue = queue.Queue(maxsize=10)
        self.running = False
        
        # Latest position data
        self.latest_position = (0.0, 0.0, 0.0)
        self.position_lock = threading.Lock()
        
        # Try to load existing calibration data
        self.load_existing_calibration()
        
    def initialize_camera(self, width: int = 640, height: int = 480, fps: int = 30) -> bool:
        """
        Initialize the RealSense camera
        
        Args:
            width: Frame width
            height: Frame height
            fps: Frames per second
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import pyrealsense2 as rs
            
            # Create pipeline and config
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # Configure streams
            self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            
            # Start streaming
            profile = self.pipeline.start(self.config)
            
            # Get camera intrinsics for calibration
            color_stream = profile.get_stream(rs.stream.color)
            intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            
            self.camera_matrix = np.array([
                [intrinsics.fx, 0, intrinsics.ppx],
                [0, intrinsics.fy, intrinsics.ppy],
                [0, 0, 1]
            ])
            
            self.dist_coeffs = np.array(intrinsics.coeffs)
            
            print("✅ RealSense camera initialized successfully")
            print(f"   Resolution: {width}x{height} @ {fps}fps")
            print(f"   Camera matrix shape: {self.camera_matrix.shape}")
            
            return True
            
        except ImportError:
            print("❌ pyrealsense2 not installed. Install with: pip install pyrealsense2")
            return False
        except Exception as e:
            print(f"❌ Failed to initialize RealSense camera: {e}")
            return False
    
    def load_existing_calibration(self) -> bool:
        """
        Load existing calibration data (prioritize blue markers, then ArUco, fallback to corner detection)
        
        Returns:
            True if calibration loaded successfully
        """
        import os
        import json
        
        calib_dir = "calibration_data"
        if not os.path.exists(calib_dir):
            return False
        
        # Find calibration files
        json_files = [f for f in os.listdir(calib_dir) if f.endswith(".json")]
        
        if not json_files:
            return False
        
        # Sort by timestamp and prefer blue marker calibrations
        blue_marker_files = [f for f in json_files if "color_calibration_" in f]
        
        # Try blue marker calibration first (most recent and accurate)
        if blue_marker_files:
            latest_file = sorted(blue_marker_files)[-1]
            json_path = os.path.join(calib_dir, latest_file)
            
            try:
                with open(json_path, 'r') as f:
                    calib_data = json.load(f)
                
                self.table_corners_pixels = np.array(calib_data["corner_pixels"], dtype=np.float32)
                
                # Update world coordinates based on actual base plate size
                base_plate_size = calib_data.get("base_plate_size_cm", 35) / 100.0  # Convert to meters
                marker_size = calib_data.get("marker_size_cm", 4) / 100.0  # Convert to meters
                
                # Blue markers are placed at corners, but their centers are offset inward by half marker size
                marker_offset = marker_size / 2  # 2cm offset from plate edge to marker center
                marker_center_distance = (base_plate_size / 2) - marker_offset  # Distance from center to marker center
                
                self.table_corners_world = np.array([
                    [-marker_center_distance, -marker_center_distance, 0],  # Top-left marker center
                    [marker_center_distance, -marker_center_distance, 0],   # Top-right marker center
                    [-marker_center_distance, marker_center_distance, 0],   # Bottom-left marker center
                    [marker_center_distance, marker_center_distance, 0]     # Bottom-right marker center
                ], dtype=np.float32)
                
                print(f"✅ Loaded blue marker calibration from: {latest_file}")
                print(f"📐 Base plate size: {base_plate_size*100:.1f}cm x {base_plate_size*100:.1f}cm")
                print(f"📐 Corner coordinates: ±{base_plate_size/2*100:.1f}cm from center")
                print(f"📐 Pixel corners: {self.table_corners_pixels}")
                return True
                
            except Exception as e:
                print(f"⚠️ Failed to load blue marker calibration: {e}")
        
        print("❌ No blue marker calibration found. Please run camera_calibration_color.py first")
        return False
    
    def pixel_to_world_coordinates(self, pixel_x: float, pixel_y: float) -> Tuple[float, float]:
        """
        Convert pixel coordinates to world coordinates (table coordinate system)
        
        Args:
            pixel_x: X coordinate in pixels
            pixel_y: Y coordinate in pixels
            
        Returns:
            Tuple of (x, y) in meters relative to table center
        """
        # Check for valid pixel coordinates
        if not (np.isfinite(pixel_x) and np.isfinite(pixel_y)):
            print(f"❌ Invalid pixel coordinates: ({pixel_x}, {pixel_y})")
            return 0.0, 0.0

        # Robust check for calibration data
        if (
            self.table_corners_pixels is None or
            not isinstance(self.table_corners_pixels, np.ndarray) or
            self.table_corners_pixels.shape != (4, 2) or
            self.table_corners_pixels.dtype != np.float32
        ):
            print(f"❌ Table not calibrated or invalid shape: {self.table_corners_pixels}")
            # Fallback: simple linear mapping (not accurate)
            world_x = (pixel_x - 320) / 320 * (self.table_size / 2)
            world_y = (pixel_y - 240) / 240 * (self.table_size / 2)
            return world_x, world_y

        # Use perspective transformation
        try:
            pixel_point = np.array([[pixel_x, pixel_y]], dtype=np.float32)
            # Force both arrays to be contiguous np.float32
            table_corners_pixels = np.ascontiguousarray(self.table_corners_pixels, dtype=np.float32)
            world_corners = np.ascontiguousarray(self.table_corners_world[:, :2], dtype=np.float32)
            if table_corners_pixels.shape != (4, 2):
                print(f"❌ Invalid table_corners_pixels shape: {table_corners_pixels.shape}")
                return 0.0, 0.0
            if world_corners.shape != (4, 2):
                print(f"❌ Invalid world corners shape: {world_corners.shape}")
                return 0.0, 0.0
            M = cv2.getPerspectiveTransform(table_corners_pixels, world_corners)
            world_point = cv2.perspectiveTransform(pixel_point.reshape(1, 1, 2), M)
            return float(world_point[0, 0, 0]), float(world_point[0, 0, 1])
        except Exception as e:
            print(f"❌ Coordinate transformation error: {e}")
            return 0.0, 0.0
    
    def start_continuous_capture(self):
        """Start continuous ball position capture in background thread"""
        if self.running:
            return
            
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        print("✅ Started continuous ball tracking")
    
    def stop_continuous_capture(self):
        """Stop continuous capture"""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join()
        print("⏹️ Stopped continuous ball tracking")
    
    def _capture_loop(self):
        """Main capture loop running in background thread"""
        if not self.pipeline:
            return
            
        try:
            import pyrealsense2 as rs
            
            while self.running:
                # Wait for frames
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                
                if not color_frame:
                    continue
                
                # Convert to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data()) if depth_frame else None
                
                # Always flip image 180 degrees for correct camera orientation
                color_image = cv2.rotate(color_image, cv2.ROTATE_180)
                if depth_image is not None:
                    depth_image = cv2.rotate(depth_image, cv2.ROTATE_180)
                
                # Detect ball
                ball_position = self.ball_detector.detect_ball(color_image, depth_image)
                
                if ball_position:
                    pixel_x, pixel_y, depth_z = ball_position
                    # Convert to world coordinates
                    world_x, world_y = self.pixel_to_world_coordinates(pixel_x, pixel_y)
                    #print(f"Ball position in world coordinates: x={world_x:.4f}, y={world_y:.4f}, z={depth_z:.4f}")
                    # Update latest position
                    with self.position_lock:
                        self.latest_position = (world_x, world_y, depth_z)
                    # Add to queue (non-blocking)
                    try:
                        self.position_queue.put_nowait((world_x, world_y, depth_z, time.time()))
                    except queue.Full:
                        # Remove oldest item and add new one
                        try:
                            self.position_queue.get_nowait()
                            self.position_queue.put_nowait((world_x, world_y, depth_z, time.time()))
                        except queue.Empty:
                            pass
                   

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
                        cv2.polylines(color_image, [corners_int], True, (0, 255, 0), 2)
                        
                        # Draw marker centers for reference (yellow, thin)
                        marker_corners_rect = np.array([
                            self.table_corners_pixels[0],  # TL marker center
                            self.table_corners_pixels[1],  # TR marker center
                            self.table_corners_pixels[3],  # BR marker center
                            self.table_corners_pixels[2]   # BL marker center
                        ])
                        marker_corners_int = marker_corners_rect.astype(np.int32)
                        cv2.polylines(color_image, [marker_corners_int], True, (0, 255, 255), 1)  # Yellow, thin line

                  
                    
                    # Draw the ball
                    if pixel_x is not None and pixel_y is not None:
                        world_x, world_y = self.pixel_to_world_coordinates(pixel_x, pixel_y)
                        
                        # Draw ball position
                        cv2.circle(color_image, (int(pixel_x), int(pixel_y)), 15, (0, 0, 255), 3)
                        cv2.circle(color_image, (int(pixel_x), int(pixel_y)), 5, (255, 255, 255), -1)

                        # Show coordinates
                        coord_text = f"Pixel: ({pixel_x:.0f}, {pixel_y:.0f})"
                        world_text = f"World: ({world_x:.3f}, {world_y:.3f})m"

                        cv2.putText(color_image, coord_text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(color_image, world_text, (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.imshow("Ball Tracking", color_image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break  # Optionally allow quitting the feed
                
                time.sleep(0.01)  # ~100 Hz capture rate
                
        except Exception as e:
            print(f"❌ Capture loop error: {e}")
    
    def get_ball_position(self) -> Tuple[float, float, float]:
        """
        Get the latest ball position
        
        Returns:
            Tuple of (x, y, z) in meters, where (0,0) is table center
        """
        with self.position_lock:
            return self.latest_position
    
    def get_ball_position_history(self, max_age: float = 1.0) -> list:
        """
        Get recent ball positions for velocity estimation
        
        Args:
            max_age: Maximum age of positions to return (seconds)
            
        Returns:
            List of (x, y, z, timestamp) tuples
        """
        current_time = time.time()
        history = []
        
        # Drain queue
        while not self.position_queue.empty():
            try:
                pos_data = self.position_queue.get_nowait()
                if current_time - pos_data[3] <= max_age:
                    history.append(pos_data)
            except queue.Empty:
                break
        
        return sorted(history, key=lambda x: x[3])  # Sort by timestamp
    
    def cleanup(self):
        """Clean up camera resources"""
        self.stop_continuous_capture()
        
        if self.pipeline:
            self.pipeline.stop()
            print("✅ Camera cleaned up")


# Simulation interface compatibility
class CameraSimulationInterface:
    """
    Interface that bridges camera input with existing simulation code
    This allows easy switching between simulation and real camera
    """
    
    def __init__(self, use_camera: bool = False, table_size: float = 0.25):
        """
        Initialize the interface
        
        Args:
            use_camera: If True, use real camera. If False, use simulation
            table_size: Table size in meters
        """
        self.use_camera = use_camera
        self.table_size = table_size
        
        if use_camera:
            self.camera = RealSenseCameraInterface(table_size)
            if not self.camera.initialize_camera():
                print("❌ Failed to initialize camera, falling back to simulation mode")
                self.use_camera = False
                self.camera = None
        else:
            self.camera = None
            
        # Velocity estimation
        self.prev_position = None
        self.prev_time = None
    
    def get_ball_state(self, simulation_ball_id=None, pybullet_module=None) -> Tuple[float, float, float, float]:
        """
        Get ball position and estimated velocity
        Compatible with existing simulation interface
        
        Args:
            simulation_ball_id: PyBullet ball ID (used only in simulation mode)
            pybullet_module: PyBullet module (used only in simulation mode)
            
        Returns:
            Tuple of (x, y, vx, vy) in meters and m/s
        """
        current_time = time.time()
        
        if self.use_camera and self.camera:
            # Get position from camera
            x, y, z = self.camera.get_ball_position()
        else:
            # Get position from simulation
            if simulation_ball_id and pybullet_module:
                ball_pos, _ = pybullet_module.getBasePositionAndOrientation(simulation_ball_id)
                x, y = ball_pos[0], ball_pos[1]
            else:
                x, y = 0.0, 0.0
        
        # Estimate velocity
        vx, vy = 0.0, 0.0
        if self.prev_position and self.prev_time:
            dt = current_time - self.prev_time
            if dt > 0:
                vx = (x - self.prev_position[0]) / dt
                vy = (y - self.prev_position[1]) / dt
        
        # Update previous values
        self.prev_position = (x, y)
        self.prev_time = current_time
        
        return x, y, vx, vy
    
    def start_tracking(self):
        """Start ball tracking (camera mode only)"""
        if self.use_camera and self.camera:
            self.camera.start_continuous_capture()
    
    def stop_tracking(self):
        """Stop ball tracking (camera mode only)"""
        if self.use_camera and self.camera:
            self.camera.stop_continuous_capture()
    
    def cleanup(self):
        """Clean up resources"""
        if self.camera:
            self.camera.cleanup()


def main():
    """
    Test script for camera interface
    """
    print("🎯 Ball Position Detection Test")
    print("==============================")
    
    # Test camera initialization
    camera = RealSenseCameraInterface()
    
    if camera.initialize_camera():
        print("\n📋 Available commands:")
        print("  's' - Start continuous tracking")
        print("  'p' - Get current ball position")
        print("  'q' - Quit")
        print("\n💡 Note: Run camera_calibration_color.py first to calibrate with blue markers")
        
        try:
            while True:
                cmd = input("\nEnter command: ").lower().strip()
                
                if cmd == 's':
                    camera.start_continuous_capture()
                    print("Tracking started. Press 'p' to get positions...")
                elif cmd == 'p':
                    pos = camera.get_ball_position()
                    print(f"Ball position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) meters")
                elif cmd == 'q':
                    break
                else:
                    print("Unknown command")
                    
        except KeyboardInterrupt:
            pass
        finally:
            camera.cleanup()
    
    else:
        print("❌ Camera test failed")


if __name__ == "__main__":
    main()

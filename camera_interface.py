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

# Import ArUco for marker detection
try:
    import cv2.aruco as aruco
    ARUCO_AVAILABLE = True
except ImportError:
    ARUCO_AVAILABLE = False
    print("⚠️ ArUco not available - install with: pip install opencv-contrib-python")


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
        
    def detect_ball(self, color_frame: np.ndarray, depth_frame: Optional[np.ndarray] = None) -> Optional[Tuple[float, float, float]]:
        """
        Detect ball position in the given frame
        
        Args:
            color_frame: RGB color image from camera
            depth_frame: Optional depth image for Z coordinate
            
        Returns:
            Tuple of (x, y, z) in pixels, or None if no ball detected
        """
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
            
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Get depth if available
        z = 0.0
        if depth_frame is not None and 0 <= cx < depth_frame.shape[1] and 0 <= cy < depth_frame.shape[0]:
            z = depth_frame[cy, cx]
            
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
        Load existing calibration data (prioritize ArUco, fallback to corner detection)
        
        Returns:
            True if calibration loaded successfully
        """
        import os
        import json
        
        calib_dir = "calibration_data"
        if not os.path.exists(calib_dir):
            return False
        
        # Find calibration files (prioritize ArUco)
        json_files = [f for f in os.listdir(calib_dir) if f.endswith(".json")]
        
        if not json_files:
            return False
        
        # Sort by timestamp and prefer ArUco calibrations
        aruco_files = [f for f in json_files if "aruco" in f]
        corner_files = [f for f in json_files if "calibration_" in f and "aruco" not in f]
        
        # Try ArUco calibration first
        if aruco_files:
            latest_file = sorted(aruco_files)[-1]
            json_path = os.path.join(calib_dir, latest_file)
            
            try:
                with open(json_path, 'r') as f:
                    calib_data = json.load(f)
                
                self.table_corners_pixels = np.array(calib_data["corner_pixels"], dtype=np.float32)
                
                print(f"✅ Loaded ArUco calibration from: {latest_file}")
                print(f"📐 Base plate corners loaded: {self.table_corners_pixels.shape}")
                return True
                
            except Exception as e:
                print(f"⚠️ Failed to load ArUco calibration: {e}")
        
        # Fallback to corner detection calibration
        if corner_files:
            latest_file = sorted(corner_files)[-1]
            json_path = os.path.join(calib_dir, latest_file)
            
            try:
                with open(json_path, 'r') as f:
                    calib_data = json.load(f)
                
                self.table_corners_pixels = np.array(calib_data["corner_pixels"], dtype=np.float32)
                
                # Flip calibration corners to match flipped image orientation
                # When image is rotated 180°, coordinates transform: (x,y) -> (640-x, 480-y)
                self.table_corners_pixels[:, 0] = 640 - self.table_corners_pixels[:, 0]  # Flip X
                self.table_corners_pixels[:, 1] = 480 - self.table_corners_pixels[:, 1]  # Flip Y
                
                print(f"✅ Loaded corner detection calibration from: {latest_file}")
                print(f"📐 Table corners loaded and flipped: {self.table_corners_pixels.shape}")
                return True
                
            except Exception as e:
                print(f"⚠️ Failed to load corner calibration: {e}")
        
        return False
    
    def calibrate_table_detection(self, num_samples: int = 10) -> bool:
        """
        Calibrate the camera by detecting ArUco markers on static base plate
        This should be run once with all 4 ArUco markers visible
        
        Args:
            num_samples: Number of samples to average for calibration
            
        Returns:
            True if calibration successful
        """
        if not self.pipeline:
            print("❌ Camera not initialized")
            return False
            
        print(f"🎯 Starting ArUco marker calibration with {num_samples} samples...")
        print("   Please ensure all 4 ArUco markers (IDs 0,1,2,3) are visible")
        print("   Make sure no ball is blocking the markers")
        print("   Recommended: 35x35cm base plate with 4x4cm markers")
        print("   Markers placed 2cm from edges for optimal detection")
        
        try:
            import pyrealsense2 as rs
            
            corner_samples = []
            successful_detections = 0
            
            for i in range(num_samples * 2):  # Allow more attempts
                # Wait for a coherent pair of frames
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    continue
                    
                # Convert to numpy array
                color_image = np.asanyarray(color_frame.get_data())
                
                # Always flip image 180 degrees for correct camera orientation
                color_image = cv2.rotate(color_image, cv2.ROTATE_180)
                
                # Detect ArUco markers
                corners = self._detect_table_corners(color_image)
                
                if corners is not None:
                    corner_samples.append(corners)
                    successful_detections += 1
                    print(f"   ✅ Sample {successful_detections}/{num_samples} captured")
                    
                    if successful_detections >= num_samples:
                        break
                else:
                    print(f"   ⚠️ Attempt {i+1} failed - ensure all 4 markers visible")
                    
                time.sleep(0.2)  # Slightly longer delay for marker detection
            
            if len(corner_samples) >= num_samples // 2:  # At least half successful
                # Average the corner positions for better accuracy
                self.table_corners_pixels = np.mean(corner_samples, axis=0)
                
                # Save calibration data
                self._save_aruco_calibration()
                
                print("✅ ArUco marker calibration successful!")
                print(f"   Base plate corners: {self.table_corners_pixels}")
                print(f"   Calibration saved for future use")
                return True
            else:
                print("❌ ArUco calibration failed - not enough valid detections")
                print("   Check that all 4 markers (IDs 0,1,2,3) are clearly visible")
                return False
                
        except Exception as e:
            print(f"❌ Calibration error: {e}")
            return False
    
    def _save_aruco_calibration(self):
        """Save ArUco calibration data to file"""
        import os
        import json
        from datetime import datetime
        
        # Create calibration directory
        calib_dir = "calibration_data"
        os.makedirs(calib_dir, exist_ok=True)
        
        # Prepare calibration data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        calib_data = {
            "timestamp": timestamp,
            "calibration_type": "aruco_markers",
            "marker_dictionary": "DICT_4X4_50",
            "marker_ids": [0, 1, 2, 3],
            "base_plate_size_cm": 35,  # 35x35cm recommended for 4x4cm markers
            "table_size_cm": 25,
            "marker_size_cm": 4,       # 4x4cm ArUco markers
            "corner_pixels": self.table_corners_pixels.tolist(),
            "image_orientation": "flipped_180_degrees"
        }
        
        # Save to JSON file
        json_path = os.path.join(calib_dir, f"aruco_calibration_{timestamp}.json")
        try:
            with open(json_path, 'w') as f:
                json.dump(calib_data, f, indent=2)
            print(f"   📁 Calibration saved: {json_path}")
        except Exception as e:
            print(f"   ⚠️ Failed to save calibration: {e}")
    
    def _detect_table_corners(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect table corners using ArUco markers on static base plate
        
        Args:
            image: Input color image
            
        Returns:
            Array of 4 corner points in pixel coordinates, or None if not found
        """
        if not ARUCO_AVAILABLE:
            print("❌ ArUco not available - falling back to corner detection")
            return self._detect_corners_fallback(image)
        
        # Convert to grayscale for ArUco detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Define ArUco dictionary (DICT_4X4_50 recommended for reliability)
        # Handle both old and new OpenCV ArUco API
        try:
            # New OpenCV 4.7+ API
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            detector = cv2.aruco.ArucoDetector(aruco_dict)
            corners, ids, _ = detector.detectMarkers(gray)
        except AttributeError:
            try:
                # Older OpenCV API
                aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
                aruco_params = aruco.DetectorParameters_create()
                corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
            except AttributeError:
                # Even older API fallback
                aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
                corners, ids, _ = aruco.detectMarkers(gray, aruco_dict)
        
        if ids is None or len(ids) < 4:
            print(f"⚠️ Only found {len(ids) if ids is not None else 0}/4 ArUco markers")
            return None
        
        # We expect markers with IDs 0, 1, 2, 3 for the four corners
        expected_ids = [0, 1, 2, 3]
        marker_centers = {}
        
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in expected_ids:
                # Get center of marker
                marker_corner = corners[i][0]  # corners[i] is shape (1, 4, 2)
                center_x = np.mean(marker_corner[:, 0])
                center_y = np.mean(marker_corner[:, 1])
                marker_centers[marker_id] = [center_x, center_y]
        
        # Check if we have all 4 expected markers
        if len(marker_centers) != 4:
            missing = set(expected_ids) - set(marker_centers.keys())
            print(f"⚠️ Missing ArUco markers: {missing}")
            return None
        
        # Arrange corners in consistent order: [0, 1, 2, 3] -> [TL, TR, BR, BL]
        # You can customize this mapping based on how you place your markers
        corner_points = np.array([
            marker_centers[0],  # Top-left
            marker_centers[1],  # Top-right  
            marker_centers[2],  # Bottom-right
            marker_centers[3]   # Bottom-left
        ], dtype=np.float32)
        
        print(f"✅ Detected all 4 ArUco markers: {list(marker_centers.keys())}")
        return corner_points
    
    def _detect_corners_fallback(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Fallback corner detection method when ArUco is not available
        """
        # Original corner detection code as fallback
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use adaptive threshold or edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for rectangular contours
        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's a quadrilateral with reasonable area
            if len(approx) == 4 and cv2.contourArea(contour) > 10000:
                # Sort corners in consistent order
                corners = approx.reshape(4, 2).astype(np.float32)
                return self._sort_corners(corners)
        
        return None
    
    def _sort_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        Sort corners in consistent order: top-left, top-right, bottom-right, bottom-left
        """
        # Calculate centroid
        center = np.mean(corners, axis=0)
        
        # Sort by angle from center
        def angle_from_center(point):
            return np.arctan2(point[1] - center[1], point[0] - center[0])
        
        sorted_corners = sorted(corners, key=angle_from_center)
        return np.array(sorted_corners, dtype=np.float32)
    
    def pixel_to_world_coordinates(self, pixel_x: float, pixel_y: float) -> Tuple[float, float]:
        """
        Convert pixel coordinates to world coordinates (table coordinate system)
        
        Args:
            pixel_x: X coordinate in pixels
            pixel_y: Y coordinate in pixels
            
        Returns:
            Tuple of (x, y) in meters relative to table center
        """
        if self.table_corners_pixels is None:
            print("❌ Table not calibrated - using approximate conversion")
            # Fallback: simple linear mapping (not accurate)
            # Assuming 640x480 resolution and table roughly in center
            world_x = (pixel_x - 320) / 320 * (self.table_size / 2)
            world_y = (pixel_y - 240) / 240 * (self.table_size / 2)
            return world_x, world_y
        
        # Use perspective transformation
        try:
            # Create perspective transformation matrix
            pixel_point = np.array([[pixel_x, pixel_y]], dtype=np.float32)
            
            # Get transformation matrix
            M = cv2.getPerspectiveTransform(self.table_corners_pixels, 
                                          self.table_corners_world[:, :2])
            
            # Transform point
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
        print("  'c' - Calibrate table detection")
        print("  's' - Start continuous tracking")
        print("  'p' - Get current ball position")
        print("  'q' - Quit")
        
        try:
            while True:
                cmd = input("\nEnter command: ").lower().strip()
                
                if cmd == 'c':
                    camera.calibrate_table_detection()
                elif cmd == 's':
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

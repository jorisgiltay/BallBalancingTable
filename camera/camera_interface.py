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
from typing import Tuple, Optional, Dict, Any, Callable
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


        
        # Contour filtering parameters - Adjusted for ball on platform (higher/closer to camera)
        self.min_contour_area = 100      # Slightly higher minimum area
        self.max_contour_area = 15000    # Much larger maximum area for closer ball
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
    
    def __init__(self, table_size: float = 0.25, camera_height: float = 0.5, disable_rendering: bool = False):
        """
        Initialize RealSense camera interface
        
        Args:
            table_size: Size of the square table in meters (default 25cm)
            camera_height: Height of camera above table in meters
            disable_rendering: If True, disable camera feed display for performance
        """
        self.table_size = table_size
        self.camera_height = camera_height
        self.disable_rendering = disable_rendering
        self.target_x = 0.0
        self.target_y = 0.0
        
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
        
        # Optional callback when user selects a new target via mouse
        self._on_target_update_callback: Optional[Callable[[float, float], None]] = None
        self._on_keypress_callback: Optional[Callable[[int], None]] = None
        self._mouse_is_down: bool = False
        # Right-button trajectory drawing/playback
        self._rmb_is_down: bool = False
        self._trajectory_world: list[tuple[float, float]] = []
        self._trajectory_pixel: list[tuple[int, int]] = []
        self._trajectory_lock = threading.Lock()
        self._trajectory_playback_active: bool = False
        self._trajectory_playback_thread: Optional[threading.Thread] = None
        self._trajectory_playback_hz: float = 30.0
        self._trajectory_loop_enabled: bool = True

        # Overlay info
        self._overlay_control_method: str = ""
        self._overlay_info_lock = threading.Lock()

        # UI / overlay layout (bottom strip with controls)
        self._frame_width: Optional[int] = None
        self._frame_height: Optional[int] = None
        self._ui_lock = threading.Lock()
        self._ui_rects: Dict[str, Tuple[int, int, int, int]] = {}
        
        # Try to load existing calibration data
        self.load_existing_calibration()
        
    def initialize_camera(self, width: int = 640, height: int = 480, fps: int = 60) -> bool:
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
                base_plate_size = calib_data.get("base_plate_size_cm", 25) / 100.0  # Convert to meters
                marker_size = calib_data.get("marker_size_cm", 4) / 100.0  # Convert to meters
                
                # Blue markers are placed at corners, but their centers are offset inward by half marker size
                marker_offset = marker_size / 1  # 2cm offset from plate edge to marker center
                marker_center_distance = (base_plate_size / 2) - marker_offset  # Distance from center to marker center
                
                # We want to map to the actual table coordinates (25cm), not the marker coordinates (31cm)
                table_half_size = self.table_size / 2  # 12.5cm for 25cm table
                
                self.table_corners_world = np.array([
                    [-table_half_size, -table_half_size, 0],  # Top-left table corner
                    [table_half_size, -table_half_size, 0],   # Top-right table corner
                    [-table_half_size, table_half_size, 0],   # Bottom-left table corner
                    [table_half_size, table_half_size, 0]     # Bottom-right table corner
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
            
            world_x = float(world_point[0, 0, 0])
            world_y = float(world_point[0, 0, 1])

            # Invert Y coordinate to match simulation coordinate system
            world_y = -world_y

            
            return world_x, world_y
        except Exception as e:
            print(f"❌ Coordinate transformation error: {e}")
            return 0.0, 0.0
        
    def world_to_pixel_coordinates(self, world_x: float, world_y: float) -> Tuple[float, float]:
        """
        Convert world coordinates (in meters relative to table center) to pixel coordinates.

        Args:
            world_x: X coordinate in world space (meters)
            world_y: Y coordinate in world space (meters)

        Returns:
            Tuple of (pixel_x, pixel_y)
        """
        # Check for valid world coordinates
        if not (np.isfinite(world_x) and np.isfinite(world_y)):
            print(f"❌ Invalid world coordinates: ({world_x}, {world_y})")
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
            pixel_x = world_x / (self.table_size / 2) * 320 + 320
            pixel_y = world_y / (self.table_size / 2) * 240 + 240
            return pixel_x, pixel_y

        try:
            #Y inversion
            world_y = -world_y

            # Create 1x1x2 array of world point
            world_point = np.array([[world_x, world_y]], dtype=np.float32).reshape(1, 1, 2)

            # Ensure arrays are contiguous np.float32
            table_corners_pixels = np.ascontiguousarray(self.table_corners_pixels, dtype=np.float32)
            world_corners = np.ascontiguousarray(self.table_corners_world[:, :2], dtype=np.float32)

            # Compute inverse transform matrix (world → pixel)
            M_inv = cv2.getPerspectiveTransform(world_corners, table_corners_pixels)

            # Apply the transformation
            pixel_point = cv2.perspectiveTransform(world_point, M_inv)

            pixel_x = float(pixel_point[0, 0, 0])
            pixel_y = float(pixel_point[0, 0, 1])

            return pixel_x, pixel_y
        except Exception as e:
            print(f"❌ Coordinate inverse transformation error: {e}")
            return 0.0, 0.0
    
    def start_continuous_capture(self):
        """Start continuous ball position capture in background thread"""
        if self.running:
            return
            
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        print("✅ Started continuous ball tracking")

    def set_on_target_update_callback(self, callback: Callable[[float, float], None]) -> None:
        """Register a callback invoked with (x_world, y_world) when user clicks/drags in the video window."""
        self._on_target_update_callback = callback

    def set_on_keypress_callback(self, callback: Callable[[int], None]) -> None:
        """Register a callback invoked with raw OpenCV key codes from the camera window."""
        self._on_keypress_callback = callback

    def set_target(self, x: float, y: float):
        self.target_x = x
        self.target_y = y
        #print(f"🎯 Target position set to: ({x:.3f}, {y:.3f})")
    
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
            
            # Prepare display window and mouse callback if rendering is enabled
            if not self.disable_rendering:
                cv2.namedWindow("Ball Tracking")
                cv2.setMouseCallback("Ball Tracking", self._handle_mouse_event)
            
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
                
                # Load calibration data to get accurate measurements
                calib_dir = "calibration_data"
                json_files = [f for f in os.listdir(calib_dir) if f.startswith("color_calibration_") and f.endswith(".json")]
                
                if json_files:
                    latest_file = sorted(json_files)[-1]
                    json_path = os.path.join(calib_dir, latest_file)
                    
                    with open(json_path, 'r') as f:
                        calib_data = json.load(f)
                    

                    corners_rect = np.array([
                        self.table_corners_pixels[0],  # TL (Top-Left plate edge)
                        self.table_corners_pixels[1],  # TR (Top-Right plate edge)  
                        self.table_corners_pixels[3],  # BR (Bottom-Right plate edge)
                        self.table_corners_pixels[2]   # BL (Bottom-Left plate edge)
                    ])
                    corners_int = corners_rect.astype(np.int32)
                    cv2.polylines(color_image, [corners_int], True, (0, 255, 0), 2)
                    
                    x_min = int(np.min(corners_rect[:, 0]))  - 5 # Add 5px padding
                    y_min = int(np.min(corners_rect[:, 1]))  - 5# Add 5px padding
                    x_max = int(np.max(corners_rect[:, 0]))  + 5# Add 5px padding
                    y_max = int(np.max(corners_rect[:, 1]))  +5 # Add 5px padding
                    crop_tuple = (x_min, y_min, x_max - x_min, y_max - y_min)

                    ball_position = self.ball_detector.detect_ball(color_image, depth_image, crop=crop_tuple)
                    
                # Draw the target position
                    if self.target_x is not None and self.target_y is not None:
                        target_pixel_x, target_pixel_y = self.world_to_pixel_coordinates(self.target_x, self.target_y)
                        if target_pixel_x is not None and target_pixel_y is not None:
                            cv2.circle(color_image, (int(target_pixel_x), int(target_pixel_y)), 15, (0, 255, 0), 2)  # Larger outer circle

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
                    
                    
                    
                    # Draw the ball
                    if pixel_x is not None and pixel_y is not None:
                        world_x, world_y = self.pixel_to_world_coordinates(pixel_x, pixel_y)
                        
                        # Draw ball position - adjusted for platform height
                        cv2.circle(color_image, (int(pixel_x), int(pixel_y)), 8, (0, 0, 255), 2)  # Larger outer circle
                        cv2.circle(color_image, (int(pixel_x), int(pixel_y)), 3, (255, 255, 255), -1)  # Larger center dot

                        # Show coordinates
                        coord_text = f"Pixel: ({pixel_x:.0f}, {pixel_y:.0f})"
                        world_text = f"World: ({world_x:.3f}, {world_y:.3f})m"

                        cv2.putText(color_image, coord_text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(color_image, world_text, (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                
                    
                # Draw any trajectory (in blue)
                try:
                    with self._trajectory_lock:
                        if len(self._trajectory_pixel) >= 2:
                            pts = np.array(self._trajectory_pixel, dtype=np.int32).reshape((-1, 1, 2))
                            # Lighter blue (BGR): more cyan-ish
                            cv2.polylines(color_image, [pts], False, (255, 200, 100), 2)
                except Exception:
                    pass

                # Update frame size and UI rects, then draw overlay panel as a bottom strip
                try:
                    h, w = color_image.shape[:2]
                    self._frame_width, self._frame_height = w, h
                    self._update_ui_layout(w, h)
                    self._draw_ui(color_image)
                except Exception:
                    pass
                
                # Only show video feed if rendering is not disabled
                if not self.disable_rendering:
                    cv2.imshow("Ball Tracking", color_image)
                    raw_key = cv2.waitKey(1)
                    key = raw_key & 0xFF if raw_key != -1 else -1
                    if raw_key != -1 and self._on_keypress_callback is not None:
                        try:
                            self._on_keypress_callback(key)
                        except Exception:
                            pass
                    elif key == ord('q'):
                        break  # Allow quitting if no external handler is set
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

    # -------------------------
    # Mouse interaction support
    # -------------------------
    def _handle_mouse_event(self, event, x, y, flags, param):
        """Convert mouse position in the displayed frame to world coords and update target.

        - Left click sets a new setpoint.
        - Dragging with left button held continuously updates the setpoint (trajectory drawing).
        """
        try:
            # First, check if click is inside UI controls (consume and return)
            if event == cv2.EVENT_LBUTTONDOWN:
                if self._handle_ui_click(x, y):
                    return

            if event == cv2.EVENT_LBUTTONDOWN:
                self._mouse_is_down = True
                wx, wy = self.pixel_to_world_coordinates(float(x), float(y))
                # Clamp to table bounds (±table_size/2 with small margin)
                half = self.table_size / 2.0 - 0.001
                wx = float(np.clip(wx, -half, half))
                wy = float(np.clip(wy, -half, half))
                self.set_target(wx, wy)
                if self._on_target_update_callback is not None:
                    self._on_target_update_callback(wx, wy)
            elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
                if self._mouse_is_down:
                    wx, wy = self.pixel_to_world_coordinates(float(x), float(y))
                    half = self.table_size / 2.0 - 0.001
                    wx = float(np.clip(wx, -half, half))
                    wy = float(np.clip(wy, -half, half))
                    self.set_target(wx, wy)
                    if self._on_target_update_callback is not None:
                        self._on_target_update_callback(wx, wy)
            elif event == cv2.EVENT_LBUTTONUP:
                self._mouse_is_down = False
            # Right button: draw trajectory in blue and play back after release
            elif event == cv2.EVENT_RBUTTONDOWN:
                # If pressing inside UI, ignore trajectory start
                if self._point_in_rect((x, y), self._ui_rects.get('panel')):
                    return
                self._rmb_is_down = True
                # Stop any previous playback and clear previous trajectory
                self._stop_trajectory_playback()
                with self._trajectory_lock:
                    self._trajectory_world = []
                    self._trajectory_pixel = []
                    # Add first point
                    wx, wy = self.pixel_to_world_coordinates(float(x), float(y))
                    half = self.table_size / 2.0 - 0.001
                    wx = float(np.clip(wx, -half, half))
                    wy = float(np.clip(wy, -half, half))
                    self._trajectory_world.append((wx, wy))
                    self._trajectory_pixel.append((int(x), int(y)))
            elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_RBUTTON):
                if self._rmb_is_down:
                    wx, wy = self.pixel_to_world_coordinates(float(x), float(y))
                    half = self.table_size / 2.0 - 0.001
                    wx = float(np.clip(wx, -half, half))
                    wy = float(np.clip(wy, -half, half))
                    with self._trajectory_lock:
                        # Avoid excessive duplicates: only add if moved a few pixels
                        if not self._trajectory_pixel or (abs(self._trajectory_pixel[-1][0] - x) + abs(self._trajectory_pixel[-1][1] - y)) >= 2:
                            self._trajectory_world.append((wx, wy))
                            self._trajectory_pixel.append((int(x), int(y)))
            elif event == cv2.EVENT_RBUTTONUP:
                self._rmb_is_down = False
                # Start playback of the drawn trajectory
                self._start_trajectory_playback()
        except Exception:
            # Ignore errors to keep capture loop robust
            pass

    # -------- UI helpers (bottom-right panel) --------
    def _update_ui_layout(self, frame_w: int, frame_h: int) -> None:
        # Bottom strip spanning width, minimal height to avoid covering table
        margin = 6
        panel_h = 36
        panel_w = max(120, frame_w - 2 * margin)
        x0 = margin
        y0 = int(frame_h - panel_h - margin)
        # Buttons aligned to the right within the strip
        gap = 8
        pad_x = 10
        center_y = y0 + panel_h // 2
        loop_w, loop_h = 90, 22
        hz_btn_w, hz_btn_h = 40, 22
        # Place from right to left: [Hz+][Hz-][Loop]
        hz_plus_x = x0 + panel_w - pad_x - hz_btn_w
        hz_plus_y = int(center_y - hz_btn_h / 2)
        hz_minus_x = hz_plus_x - gap - hz_btn_w
        hz_minus_y = hz_plus_y
        loop_x = hz_minus_x - gap - loop_w
        loop_y = hz_plus_y
        loop_rect = (loop_x, loop_y, loop_w, loop_h)
        hz_minus_rect = (hz_minus_x, hz_minus_y, hz_btn_w, hz_btn_h)
        hz_plus_rect = (hz_plus_x, hz_plus_y, hz_btn_w, hz_btn_h)
        # Text area occupies left side up to the loop button
        text_rect = (x0 + pad_x, y0, max(0, loop_x - gap - (x0 + pad_x)), panel_h)
        with self._ui_lock:
            self._ui_rects = {
                'panel': (x0, y0, panel_w, panel_h),
                'text': text_rect,
                'loop': loop_rect,
                'hz_minus': hz_minus_rect,
                'hz_plus': hz_plus_rect,
            }

    def _draw_ui(self, img: np.ndarray) -> None:
        with self._ui_lock:
            rects = dict(self._ui_rects)
        panel = rects.get('panel')
        if not panel:
            return
        x0, y0, w, h = panel
        # Panel background (solid dark gray)
        cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (40, 40, 40), -1)
        cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (80, 80, 80), 1)

        # One-line text on the left side
        with self._overlay_info_lock:
            method = self._overlay_control_method
        loop_text = "ON" if self._trajectory_loop_enabled else "OFF"
        status = f"Control: {method.upper() if method else ''} | Traj: {'PLAY' if self._trajectory_playback_active else 'IDLE'} | Loop: {loop_text} | {int(self._trajectory_playback_hz)} Hz"
        tx, ty, tw, th = rects.get('text', (x0 + 10, y0, max(0, w - 200), h))
        # Fit text horizontally by adjusting font scale if needed
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        color = (220, 220, 220)
        (text_w, text_h), _ = cv2.getTextSize(status, font, scale, thickness)
        while text_w > max(0, tw - 4) and scale > 0.35:
            scale -= 0.05
            (text_w, text_h), _ = cv2.getTextSize(status, font, scale, thickness)
        baseline_y = y0 + (h + text_h) // 2 - 2
        cv2.putText(img, status, (tx, baseline_y), font, scale, color, thickness, cv2.LINE_AA)

        # Loop toggle button
        lx, ly, lw, lh = rects['loop']
        loop_color = (90, 180, 90) if self._trajectory_loop_enabled else (90, 90, 90)
        cv2.rectangle(img, (lx, ly), (lx + lw, ly + lh), loop_color, -1)
        cv2.rectangle(img, (lx, ly), (lx + lw, ly + lh), (20, 20, 20), 1)
        cv2.putText(img, f"Loop: {'ON' if self._trajectory_loop_enabled else 'OFF'}", (lx + 8, ly + lh - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
        # Hz buttons and text
        mx, my, mw, mh = rects['hz_minus']
        px, py, pw, ph = rects['hz_plus']
        cv2.rectangle(img, (mx, my), (mx + mw, my + mh), (70, 70, 160), -1)
        cv2.rectangle(img, (mx, my), (mx + mw, my + mh), (20, 20, 20), 1)
        cv2.putText(img, "Hz -", (mx + 6, my + mh - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
        cv2.rectangle(img, (px, py), (px + pw, py + ph), (70, 160, 70), -1)
        cv2.rectangle(img, (px, py), (px + pw, py + ph), (20, 20, 20), 1)
        cv2.putText(img, "Hz +", (px + 6, py + ph - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)

    def _handle_ui_click(self, x: int, y: int) -> bool:
        """Return True if the click was handled by UI (and should not affect setpoints)."""
        with self._ui_lock:
            rects = dict(self._ui_rects)
        if not rects:
            return False
        # Check loop toggle
        if self._point_in_rect((x, y), rects.get('loop')):
            self._trajectory_loop_enabled = not self._trajectory_loop_enabled
            return True
        # Hz minus
        if self._point_in_rect((x, y), rects.get('hz_minus')):
            new_hz = max(5.0, self._trajectory_playback_hz - 5.0)
            self.set_trajectory_playback_rate_hz(new_hz)
            return True
        # Hz plus
        if self._point_in_rect((x, y), rects.get('hz_plus')):
            new_hz = min(60.0, self._trajectory_playback_hz + 5.0)
            self.set_trajectory_playback_rate_hz(new_hz)
            return True
        # Panel area (but not a control) -> consume nothing
        return False

    @staticmethod
    def _point_in_rect(pt: Tuple[int, int], rect: Optional[Tuple[int, int, int, int]]) -> bool:
        if rect is None:
            return False
        x, y = pt
        rx, ry, rw, rh = rect
        return (rx <= x <= rx + rw) and (ry <= y <= ry + rh)

    def _start_trajectory_playback(self):
        # If no points, do nothing
        with self._trajectory_lock:
            if len(self._trajectory_world) < 2:
                return
        # Stop any existing playback
        self._stop_trajectory_playback()
        self._trajectory_playback_active = True
        self._trajectory_playback_thread = threading.Thread(target=self._trajectory_playback_worker, daemon=True)
        self._trajectory_playback_thread.start()

    def _stop_trajectory_playback(self):
        if self._trajectory_playback_active:
            self._trajectory_playback_active = False
            if self._trajectory_playback_thread is not None:
                try:
                    self._trajectory_playback_thread.join(timeout=0.1)
                except Exception:
                    pass
            self._trajectory_playback_thread = None

    def _trajectory_playback_worker(self):
        try:
            dt = 1.0 / max(1.0, float(self._trajectory_playback_hz))
        except Exception:
            dt = 0.02
        idx = 0
        while self._trajectory_playback_active:
            with self._trajectory_lock:
                total = len(self._trajectory_world)
                if total == 0:
                    break
                if idx >= total:
                    if self._trajectory_loop_enabled:
                        idx = 0
                    else:
                        break
                wx, wy = self._trajectory_world[idx]
            # Update target and notify
            self.set_target(wx, wy)
            if self._on_target_update_callback is not None:
                try:
                    self._on_target_update_callback(wx, wy)
                except Exception:
                    pass
            idx += 1
            time.sleep(dt)
        self._trajectory_playback_active = False

    # Public configuration helpers
    def set_overlay_control_method(self, method: str) -> None:
        with self._overlay_info_lock:
            self._overlay_control_method = str(method)

    def set_trajectory_loop(self, enabled: bool) -> None:
        self._trajectory_loop_enabled = bool(enabled)

    def set_trajectory_playback_rate_hz(self, hz: float) -> None:
        try:
            hz = float(hz)
            if hz <= 0:
                return
            self._trajectory_playback_hz = hz
        except Exception:
            pass


# Simulation interface compatibility
class CameraSimulationInterface:
    """
    Interface that bridges camera input with existing simulation code
    This allows easy switching between simulation and real camera
    """
    
    def __init__(self, use_camera: bool = False, table_size: float = 0.25, disable_rendering: bool = False):
        """
        Initialize the interface
        
        Args:
            use_camera: If True, use real camera. If False, use simulation
            table_size: Table size in meters
            disable_rendering: If True, disable camera feed display for performance
        """
        self.use_camera = use_camera
        self.table_size = table_size
        self.disable_rendering = disable_rendering
        
        if use_camera:
            self.camera = RealSenseCameraInterface(table_size, disable_rendering=disable_rendering)
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
    
    def set_target_position(self, x: float, y: float):  
        """        Set target position for the ball (camera mode only)
        Args:
            x: Target X position in meters
            y: Target Y position in meters
        """
        if self.use_camera and self.camera:
            self.camera.set_target(x, y)
    
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

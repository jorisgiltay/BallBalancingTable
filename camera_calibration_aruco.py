"""
Camera Calibration Test - ArUco Marker Version

This script tests the ArUco marker calibration process.
It will detect your 4 ArUco markers on the static base plate and save calibration data.

Setup Requirements:
- 35x35cm wooden base plate
- 4x4cm ArUco markers (IDs 0,1,2,3) placed at corners
- Print markers using: python generate_aruco_markers.py

Usage: python camera_calibration_test.py
"""

import cv2
import numpy as np
import time
import os
import json
from typing import Optional, Tuple
import pyrealsense2 as rs

# Import ArUco for marker detection
try:
    import cv2.aruco as aruco
    ARUCO_AVAILABLE = True
except ImportError:
    ARUCO_AVAILABLE = False
    print("⚠️ ArUco not available - install with: pip install opencv-contrib-python")


class ArUcoCalibrationTest:
    """
    ArUco marker calibration test for static base plate
    """
    
    def __init__(self):
        self.pipeline = None
        self.config = None
        self.table_corners_pixels = None
        
        # Base plate and table dimensions
        self.base_plate_size = 0.35  # 35cm x 35cm base plate
        self.table_size = 0.25       # 25cm x 25cm tilting table
        self.marker_size = 0.04      # 4cm x 4cm ArUco markers
        
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
    
    def detect_aruco_markers(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect ArUco markers on static base plate
        
        Args:
            image: Input color image
            
        Returns:
            Array of 4 corner points in pixel coordinates, or None if not found
        """
        if not ARUCO_AVAILABLE:
            print("❌ ArUco not available - install opencv-contrib-python")
            return self.detect_table_corners_fallback(image)
        
        # Convert to grayscale for ArUco detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Enhanced preprocessing for better detection on colored backgrounds
        # Apply adaptive histogram equalization for better contrast
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Additional contrast enhancement for challenging backgrounds
        # Apply gamma correction to brighten the image
        gamma = 1.2
        gamma_corrected = np.power(gray / 255.0, gamma) * 255.0
        gray = gamma_corrected.astype(np.uint8)
        
        # Apply unsharp masking to enhance edges
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        unsharp = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
        gray = unsharp
        
        # Ensure we're in valid range
        gray = np.clip(gray, 0, 255).astype(np.uint8)
        
        # Optional: Apply slight Gaussian blur to reduce noise (after enhancement)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Define ArUco dictionary (DICT_4X4_50 recommended for reliability)
        # Handle both old and new OpenCV ArUco API
        try:
            # New OpenCV 4.7+ API
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            
            # Enhanced detection parameters for better reliability on challenging backgrounds
            detector_params = cv2.aruco.DetectorParameters()
            detector_params.adaptiveThreshWinSizeMin = 3
            detector_params.adaptiveThreshWinSizeMax = 31  # Increased for larger markers
            detector_params.adaptiveThreshWinSizeStep = 4   # Smaller steps for finer tuning
            detector_params.adaptiveThreshConstant = 5      # Reduced for better contrast
            detector_params.minMarkerPerimeterRate = 0.01   # Lower to detect smaller markers
            detector_params.maxMarkerPerimeterRate = 6.0    # Higher range
            detector_params.polygonalApproxAccuracyRate = 0.05  # More lenient
            detector_params.minCornerDistanceRate = 0.03    # Closer corners allowed
            detector_params.minDistanceToBorder = 1         # Allow closer to border
            detector_params.minMarkerDistanceRate = 0.03    # Closer markers allowed
            detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            detector_params.cornerRefinementWinSize = 7     # Larger window
            detector_params.cornerRefinementMaxIterations = 50  # More iterations
            detector_params.cornerRefinementMinAccuracy = 0.05  # More lenient
            
            detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
            corners, ids, _ = detector.detectMarkers(gray)
            
        except AttributeError:
            try:
                # Older OpenCV API with enhanced parameters
                aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
                aruco_params = aruco.DetectorParameters_create()
                
                # Enhanced parameters for older API - matching new API settings
                aruco_params.adaptiveThreshWinSizeMin = 3
                aruco_params.adaptiveThreshWinSizeMax = 31
                aruco_params.adaptiveThreshWinSizeStep = 4
                aruco_params.adaptiveThreshConstant = 5
                aruco_params.minMarkerPerimeterRate = 0.01
                aruco_params.maxMarkerPerimeterRate = 6.0
                aruco_params.polygonalApproxAccuracyRate = 0.05
                aruco_params.minCornerDistanceRate = 0.03
                aruco_params.minDistanceToBorder = 1
                aruco_params.minMarkerDistanceRate = 0.03
                aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
                aruco_params.cornerRefinementWinSize = 7
                aruco_params.cornerRefinementMaxIterations = 50
                aruco_params.cornerRefinementMinAccuracy = 0.05
                
                corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
                
            except AttributeError:
                # Even older API fallback - basic detection
                aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
                corners, ids, _ = aruco.detectMarkers(gray, aruco_dict)
        
        # Debug: Save the processed image for troubleshooting
        debug_dir = "debug_frames"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        cv2.imwrite(f"{debug_dir}/processed_gray.jpg", gray)
        
        # If no markers found, try with different thresholding methods
        if ids is None or len(ids) == 0:
            print("🔍 No markers with standard processing, trying enhanced detection...")
            
            # Try multiple threshold methods
            threshold_methods = [
                ("OTSU", lambda g: cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
                ("Adaptive Mean", lambda g: cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)),
                ("Adaptive Gaussian", lambda g: cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
                ("Fixed High", lambda g: cv2.threshold(g, 150, 255, cv2.THRESH_BINARY)[1]),
                ("Fixed Low", lambda g: cv2.threshold(g, 100, 255, cv2.THRESH_BINARY)[1])
            ]
            
            for method_name, threshold_func in threshold_methods:
                try:
                    thresh = threshold_func(gray)
                    cv2.imwrite(f"{debug_dir}/threshold_{method_name.lower().replace(' ', '_')}.jpg", thresh)
                    
                    if hasattr(cv2.aruco, 'ArucoDetector'):
                        detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
                        corners_try, ids_try, _ = detector.detectMarkers(thresh)
                    else:
                        corners_try, ids_try, _ = aruco.detectMarkers(thresh, aruco_dict, parameters=aruco_params)
                    
                    if ids_try is not None and len(ids_try) >= len(ids if ids is not None else []):
                        print(f"   🎯 Better result with {method_name} thresholding: {len(ids_try)} markers")
                        corners, ids = corners_try, ids_try
                        
                        if len(ids) >= 4:
                            break
                            
                except Exception as e:
                    print(f"   ⚠️ {method_name} thresholding failed: {e}")
                    continue
        
        if ids is None or len(ids) < 4:
            print(f"⚠️ Only found {len(ids) if ids is not None else 0}/4 ArUco markers")
            if ids is not None:
                print(f"   Detected marker IDs: {ids.flatten().tolist()}")
            print(f"💡 Debug images saved to '{debug_dir}/' - check lighting and marker contrast")
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
        
        # Arrange corners in consistent order: [0, 1, 3, 2] -> [TL, TR, BL, BR]
        corner_points = np.array([
            marker_centers[0],  # Top-left
            marker_centers[1],  # Top-right  
            marker_centers[3],  # Bottom-left
            marker_centers[2]   # Bottom-right
        ], dtype=np.float32)
        
        print(f"✅ Detected all 4 ArUco markers: {list(marker_centers.keys())}")
        return corner_points
    
    def detect_table_corners_fallback(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect table corners (fallback method when ArUco not available)"""
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
    
    def preview_detection(self):
        """Live preview to help position camera and markers"""
        if not self.pipeline:
            print("❌ Camera not initialized")
            return
        
        print("\n👁️ Live ArUco Detection Preview")
        print("=" * 30)
        print("📋 Position your camera to see all 4 markers")
        print("🎯 Markers should appear as large as possible")
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
                corners = self.detect_aruco_markers(color_image)
                
                if corners is not None:
                    # Draw detected corners
                    corners_int = corners.astype(np.int32)
                    cv2.polylines(preview_image, [corners_int], True, (0, 255, 0), 3)
                    
                    # Number the corners
                    for i, corner in enumerate(corners):
                        x, y = int(corner[0]), int(corner[1])
                        cv2.circle(preview_image, (x, y), 8, (0, 255, 255), -1)
                        cv2.putText(preview_image, f"{i}", (x-10, y-15), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # Status text
                    cv2.putText(preview_image, "✓ All 4 markers detected", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    # Status text
                    cv2.putText(preview_image, "✗ Markers not detected", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(preview_image, "Move closer or improve lighting", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Show the frame
                cv2.imshow('ArUco Detection Preview', preview_image)
                
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
        
        print(f"\n🎯 Starting ArUco Marker Calibration")
        print("=" * 40)
        print(f"📊 Will take {num_samples} samples")
        print("📋 Make sure all 4 ArUco markers (IDs 0,1,2,3) are clearly visible")
        print("🏗️ Ensure 35x35cm base plate with 4x4cm markers is in view")
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
                corners = self.detect_aruco_markers(color_image)
                
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
        """Save ArUco calibration results to file"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        calibration_data = {
            "timestamp": timestamp,
            "calibration_type": "aruco_markers",
            "marker_dictionary": "DICT_4X4_50",
            "marker_ids": [0, 1, 2, 3],
            "base_plate_size_cm": 35,      # 35x35cm base plate
            "table_size_cm": 25,           # 25x25cm tilting table
            "marker_size_cm": 4,           # 4x4cm ArUco markers
            "valid_samples": valid_samples,
            "corner_pixels": self.table_corners_pixels.tolist(),
            "images_saved": images_saved,
            "camera_resolution": [640, 480],
            "image_orientation": "flipped_180_degrees"
        }
        
        # Save as JSON
        json_filename = f"{self.output_dir}/aruco_calibration_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        print(f"💾 ArUco calibration data saved: {json_filename}")
        
        # Save as numpy array for easy loading
        npy_filename = f"{self.output_dir}/aruco_corners_{timestamp}.npy"
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
    print("This will calibrate your camera to detect ArUco markers.")
    print("Make sure your base plate with 4 ArUco markers is visible.\n")
    
    calibrator = ArUcoCalibrationTest()
    
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

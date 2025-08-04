"""
Camera Preview and Table Detection Test

This script helps you:
1. See what your RealSense camera captures
2. Test table edge detection in real-time
3. Check if your table is properly visible
4. Adjust camera position and lighting

Usage:
- Press 's' to save current frame
- Press 'e' to toggle edge detection overlay
- Press 'd' to toggle debug visualization
- Press 'c' to test corner detection
- Press 'q' to quit

Author: Ball Balancing Table Project
"""

import cv2
import numpy as np
import time
import os
from typing import Optional, Tuple
import pyrealsense2 as rs


class CameraPreview:
    """
    Real-time camera preview with table detection visualization
    """
    
    def __init__(self):
        self.pipeline = None
        self.config = None
        self.show_edges = False
        self.show_debug = False
        self.frame_count = 0
        self.save_count = 0
        
        # Create output directory for saved frames
        self.output_dir = "camera_frames"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"📁 Created output directory: {self.output_dir}")
    
    def initialize_camera(self, width: int = 640, height: int = 480, fps: int = 30) -> bool:
        """Initialize RealSense camera"""
        try:
            # Create pipeline and config
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # Configure streams
            self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            
            # Start streaming
            profile = self.pipeline.start(self.config)
            
            print("✅ RealSense camera initialized successfully")
            print(f"   Resolution: {width}x{height} @ {fps}fps")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize RealSense camera: {e}")
            return False
    
    def detect_table_edges(self, image: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Detect table edges and corners
        
        Returns:
            Tuple of (edge_image, corners) where corners is None if not found
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection with multiple thresholds for testing
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for rectangular contours
        best_corners = None
        best_area = 0
        
        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's a quadrilateral with reasonable area
            area = cv2.contourArea(contour)
            if len(approx) == 4 and area > 10000:
                if area > best_area:
                    best_area = area
                    corners = approx.reshape(4, 2).astype(np.float32)
                    best_corners = self._sort_corners(corners)
        
        return edges, best_corners
    
    def _sort_corners(self, corners: np.ndarray) -> np.ndarray:
        """Sort corners in consistent order"""
        # Calculate centroid
        center = np.mean(corners, axis=0)
        
        # Sort by angle from center
        def angle_from_center(point):
            return np.arctan2(point[1] - center[1], point[0] - center[0])
        
        sorted_corners = sorted(corners, key=angle_from_center)
        return np.array(sorted_corners, dtype=np.float32)
    
    def draw_debug_info(self, image: np.ndarray, edges: np.ndarray, corners: Optional[np.ndarray]) -> np.ndarray:
        """Draw debug information on the image"""
        debug_image = image.copy()
        h, w = image.shape[:2]
        
        # Draw crosshairs at center
        cv2.line(debug_image, (w//2 - 20, h//2), (w//2 + 20, h//2), (0, 255, 0), 2)
        cv2.line(debug_image, (w//2, h//2 - 20), (w//2, h//2 + 20), (0, 255, 0), 2)
        
        # Draw edges overlay if enabled
        if self.show_edges:
            # Convert edges to 3-channel for overlay
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            edges_colored[:, :, 1] = 0  # Remove green channel
            edges_colored[:, :, 0] = 0  # Remove blue channel
            # Overlay edges in red
            debug_image = cv2.addWeighted(debug_image, 0.7, edges_colored, 0.3, 0)
        
        # Draw detected corners
        if corners is not None:
            # Draw corner points
            for i, corner in enumerate(corners):
                x, y = int(corner[0]), int(corner[1])
                cv2.circle(debug_image, (x, y), 8, (0, 255, 255), -1)  # Yellow circles
                cv2.putText(debug_image, f"{i+1}", (x-10, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Draw table outline
            corners_int = corners.astype(np.int32)
            cv2.polylines(debug_image, [corners_int], True, (0, 255, 0), 3)
            
            # Calculate and display table info
            # Estimate table size based on corners
            width_px = np.linalg.norm(corners[1] - corners[0])
            height_px = np.linalg.norm(corners[3] - corners[0])
            area_px = cv2.contourArea(corners_int)
            
            cv2.putText(debug_image, f"Table detected!", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(debug_image, f"Width: {width_px:.0f}px", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(debug_image, f"Height: {height_px:.0f}px", (10, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(debug_image, f"Area: {area_px:.0f}px²", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(debug_image, "No table detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Show mode info
        mode_text = []
        if self.show_edges:
            mode_text.append("EDGES")
        if self.show_debug:
            mode_text.append("DEBUG")
        
        if mode_text:
            cv2.putText(debug_image, " | ".join(mode_text), (w - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return debug_image
    
    def save_frame(self, color_image: np.ndarray, depth_image: Optional[np.ndarray] = None):
        """Save current frame(s) to disk"""
        timestamp = int(time.time() * 1000)
        
        # Save color frame
        color_filename = f"{self.output_dir}/frame_{self.save_count:03d}_{timestamp}_color.jpg"
        cv2.imwrite(color_filename, color_image)
        print(f"💾 Saved color frame: {color_filename}")
        
        # Save depth frame if available
        if depth_image is not None:
            depth_filename = f"{self.output_dir}/frame_{self.save_count:03d}_{timestamp}_depth.png"
            # Normalize depth for visualization
            depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            cv2.imwrite(depth_filename, depth_normalized)
            print(f"💾 Saved depth frame: {depth_filename}")
        
        self.save_count += 1
    
    def run_preview(self):
        """Run the main preview loop with safety checks"""
        if not self.pipeline:
            print("❌ Camera not initialized")
            return
        
        print("\n🎥 Camera Preview Started")
        print("==========================================")
        print("Controls:")
        print("  's' - Save current frame")
        print("  'e' - Toggle edge detection overlay") 
        print("  'd' - Toggle debug visualization")
        print("  'c' - Test corner detection")
        print("  'q' - Quit")
        print("==========================================")
        print("⚠️  Safety: Auto-exit after 1000 frames to prevent hanging")
        
        max_frames = 1000  # Safety limit
        timeout_count = 0
        max_timeout = 10  # Max consecutive timeouts before exit
        
        try:
            while self.frame_count < max_frames:
                try:
                    # Wait for frames with timeout
                    frames = self.pipeline.wait_for_frames(timeout_ms=5000)  # 5 second timeout
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()
                    
                    if not color_frame:
                        timeout_count += 1
                        if timeout_count >= max_timeout:
                            print("❌ Too many frame timeouts, stopping preview")
                            break
                        continue
                    
                    # Reset timeout counter on successful frame
                    timeout_count = 0
                    
                    # Convert to numpy arrays
                    color_image = np.asanyarray(color_frame.get_data())
                    depth_image = np.asanyarray(depth_frame.get_data()) if depth_frame else None
                    
                    # Detect table edges (with error handling)
                    try:
                        edges, corners = self.detect_table_edges(color_image)
                    except Exception as e:
                        print(f"⚠️ Edge detection error: {e}")
                        edges, corners = None, None
                    
                    # Create display image
                    try:
                        if self.show_debug and edges is not None:
                            display_image = self.draw_debug_info(color_image, edges, corners)
                        else:
                            display_image = color_image.copy()
                            if corners is not None:
                                # Just draw the table outline without debug info
                                corners_int = corners.astype(np.int32)
                                cv2.polylines(display_image, [corners_int], True, (0, 255, 0), 2)
                    except Exception as e:
                        print(f"⚠️ Display error: {e}")
                        display_image = color_image.copy()
                    
                    # Show frame with error handling
                    try:
                        cv2.imshow('Camera Preview - Table Detection', display_image)
                    except Exception as e:
                        print(f"⚠️ Display window error: {e}")
                        break
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('q'):
                        print("🛑 User requested quit")
                        break
                    elif key == ord('s'):
                        try:
                            self.save_frame(color_image, depth_image)
                        except Exception as e:
                            print(f"⚠️ Save error: {e}")
                    elif key == ord('e'):
                        self.show_edges = not self.show_edges
                        print(f"🔄 Edge detection overlay: {'ON' if self.show_edges else 'OFF'}")
                    elif key == ord('d'):
                        self.show_debug = not self.show_debug
                        print(f"🔄 Debug visualization: {'ON' if self.show_debug else 'OFF'}")
                    elif key == ord('c'):
                        if corners is not None:
                            print("🎯 Corner detection test:")
                            for i, corner in enumerate(corners):
                                print(f"   Corner {i+1}: ({corner[0]:.1f}, {corner[1]:.1f})")
                        else:
                            print("❌ No corners detected in current frame")
                    
                    self.frame_count += 1
                    
                    # Progress indicator every 100 frames
                    if self.frame_count % 100 == 0:
                        print(f"📊 Processed {self.frame_count} frames...")
                    
                except Exception as frame_error:
                    print(f"⚠️ Frame processing error: {frame_error}")
                    timeout_count += 1
                    if timeout_count >= max_timeout:
                        print("❌ Too many errors, stopping preview")
                        break
                    time.sleep(0.1)  # Small delay before retry
                
        except Exception as main_error:
            print(f"❌ Main loop error: {main_error}")
        except KeyboardInterrupt:
            print("🛑 Interrupted by user")
        
        finally:
            print("🧹 Cleaning up...")
            try:
                cv2.destroyAllWindows()
            except:
                pass
            try:
                if self.pipeline:
                    self.pipeline.stop()
            except:
                pass
            print(f"✅ Preview stopped. Processed {self.frame_count} frames, saved {self.save_count} frames")
            
            if self.frame_count >= max_frames:
                print(f"⚠️ Reached safety limit of {max_frames} frames")


def main():
    """Main function"""
    print("🎥 RealSense Camera Preview and Table Detection Test")
    print("====================================================")
    
    preview = CameraPreview()
    
    if preview.initialize_camera():
        preview.run_preview()
    else:
        print("❌ Failed to start camera preview")
        print("\nTroubleshooting:")
        print("1. Check that your RealSense camera is connected")
        print("2. Install pyrealsense2: pip install pyrealsense2")
        print("3. Make sure no other application is using the camera")


if __name__ == "__main__":
    main()

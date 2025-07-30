#!/usr/bin/env python3
"""
HSV Color Debugging Tool
Shows HSV values at mouse position and helps adjust color ranges
"""

import cv2
import numpy as np
import pyrealsense2 as rs


class HSVDebugger:
    def __init__(self):
        self.pipeline = None
        self.config = None
        self.current_hsv = None
        
    def initialize_camera(self):
        """Initialize RealSense camera"""
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # Configure stream
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
            # Start streaming
            self.pipeline.start(self.config)
            print("✅ Camera initialized")
            return True
            
        except Exception as e:
            print(f"❌ Camera initialization failed: {e}")
            return False
    
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback to display HSV values"""
        if event == cv2.EVENT_MOUSEMOVE and self.current_hsv is not None:
            hsv_value = self.current_hsv[y, x]
            print(f"Position ({x}, {y}) - HSV: {hsv_value}")
    
    def run_debug(self):
        """Run HSV debugging"""
        if not self.initialize_camera():
            return
            
        cv2.namedWindow("HSV Debug")
        cv2.setMouseCallback("HSV Debug", self.mouse_callback)
        
        print("🎨 HSV Color Debugger")
        print("Move mouse over colors to see HSV values")
        print("Press 'q' or ESC to quit")
        
        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                
                # Convert to numpy array
                color_image = np.asanyarray(color_frame.get_data())
                
                # Convert to HSV
                hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
                self.current_hsv = hsv
                
                # Show original image
                cv2.imshow("HSV Debug", color_image)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # ESC
                    break
                    
        except KeyboardInterrupt:
            pass
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    debugger = HSVDebugger()
    debugger.run_debug()

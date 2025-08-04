"""
Simple RealSense Camera Test - Safe Version

This is a minimal script to test if your RealSense camera works
without hanging your system. It takes just a few frames and exits.

Usage: python camera_test_simple.py
"""

import cv2
import numpy as np
import time
import os


def test_camera_basic():
    """Basic camera test - just capture a few frames"""
    print("🔍 Testing RealSense Camera - Basic Mode")
    print("=======================================")
    
    try:
        import pyrealsense2 as rs
        print("✅ pyrealsense2 imported successfully")
    except ImportError:
        print("❌ pyrealsense2 not installed")
        print("   Install with: pip install pyrealsense2")
        return False
    
    pipeline = None
    
    try:
        # Initialize camera
        print("🔌 Initializing camera...")
        pipeline = rs.pipeline()
        config = rs.config()
        
        # Configure streams (lower resolution for safety)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)  # Lower FPS
        
        # Start pipeline
        print("▶️ Starting camera stream...")
        profile = pipeline.start(config)
        
        print("✅ Camera started successfully!")
        print("📸 Capturing 5 test frames...")
        
        # Create output directory
        output_dir = "camera_test_frames"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Capture a few frames
        for i in range(5):
            print(f"   Frame {i+1}/5...")
            
            # Wait for frame with timeout
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                print(f"   ⚠️ Frame {i+1} failed - no color data")
                continue
            
            # Convert to numpy array
            color_image = np.asanyarray(color_frame.get_data())
            
            # Save frame
            filename = f"{output_dir}/test_frame_{i+1}.jpg"
            cv2.imwrite(filename, color_image)
            print(f"   💾 Saved: {filename}")
            
            # Small delay
            time.sleep(0.5)
        
        print("✅ Basic camera test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False
        
    finally:
        if pipeline:
            try:
                pipeline.stop()
                print("🛑 Camera stopped")
            except:
                pass


def test_camera_info():
    """Get camera information without starting streams"""
    print("\n🔍 Camera Information Test")
    print("=========================")
    
    try:
        import pyrealsense2 as rs
        
        # Create context to enumerate devices
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) == 0:
            print("❌ No RealSense devices found")
            return False
        
        print(f"✅ Found {len(devices)} RealSense device(s):")
        
        for i, device in enumerate(devices):
            print(f"\n📷 Device {i+1}:")
            print(f"   Name: {device.get_info(rs.camera_info.name)}")
            print(f"   Serial: {device.get_info(rs.camera_info.serial_number)}")
            print(f"   Firmware: {device.get_info(rs.camera_info.firmware_version)}")
            
            # Get available streams
            sensors = device.query_sensors()
            for j, sensor in enumerate(sensors):
                print(f"   Sensor {j+1}: {sensor.get_info(rs.camera_info.name)}")
                
                profiles = sensor.get_stream_profiles()
                for profile in profiles:
                    if profile.stream_type() == rs.stream.color:
                        vp = profile.as_video_stream_profile()
                        print(f"     Color: {vp.width()}x{vp.height()} @ {vp.fps()}fps")
                    elif profile.stream_type() == rs.stream.depth:
                        vp = profile.as_video_stream_profile()
                        print(f"     Depth: {vp.width()}x{vp.height()} @ {vp.fps()}fps")
        
        return True
        
    except Exception as e:
        print(f"❌ Device enumeration failed: {e}")
        return False


def main():
    """Main test function"""
    print("🎯 RealSense Camera Safety Test")
    print("===============================")
    print("This script will:")
    print("1. Check camera information")
    print("2. Capture 5 test frames safely")
    print("3. Exit automatically")
    print("\nPress Ctrl+C anytime to stop safely.\n")
    
    try:
        # Test 1: Get camera info (safe)
        info_success = test_camera_info()
        
        if not info_success:
            print("\n❌ Camera not detected or accessible")
            return
        
        # Ask user before proceeding to frame capture
        print("\n" + "="*50)
        response = input("Continue with frame capture test? (y/n): ").lower().strip()
        
        if response != 'y':
            print("🛑 Test cancelled by user")
            return
        
        # Test 2: Basic frame capture
        capture_success = test_camera_basic()
        
        if capture_success:
            print("\n🎉 All tests passed!")
            print("Your RealSense camera is working correctly.")
            print("You can now try the full camera_preview.py script.")
        else:
            print("\n❌ Frame capture failed")
            print("Check camera connection and drivers.")
    
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    
    print("\n✅ Test completed safely")


if __name__ == "__main__":
    main()

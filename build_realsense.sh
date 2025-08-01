#!/bin/bash
# build_realsense.sh - Automated RealSense build script for AGX Orin

set -e  # Exit on any error

echo "=== RealSense Build Script for AGX Orin ==="
echo "Detecting system capabilities..."

# Get system info
CORES=$(nproc)
RAM_GB=$(free -g | awk '/^Mem:/{print $2}')

echo "CPU cores: $CORES"
echo "RAM: ${RAM_GB}GB"

if [[ $RAM_GB -ge 32 ]]; then
    echo "🚀 High-performance system detected! Build should take 5-10 minutes"
    MAKE_JOBS=$((CORES * 2))  # Use more parallel jobs with lots of RAM
elif [[ $RAM_GB -ge 16 ]]; then
    echo "💪 Good system specs. Build should take 10-15 minutes"
    MAKE_JOBS=$CORES
else
    echo "⏱️  Standard build time: 15-30 minutes"
    MAKE_JOBS=$((CORES / 2))  # Be conservative with limited RAM
fi

echo "Will use $MAKE_JOBS parallel build jobs"
echo

# Check if virtual environment is active
if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "✓ Virtual environment detected: $VIRTUAL_ENV"
    echo "pyrealsense2 will be installed into this venv"
else
    echo "⚠️  No virtual environment detected"
    echo "pyrealsense2 will be installed system-wide"
    echo "To install into venv, activate it first: source venv/bin/activate"
fi
echo

# Check if we're on ARM64
if [[ $(uname -m) != "aarch64" ]]; then
    echo "Warning: This script is designed for ARM64/aarch64 systems"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Remove existing pyrealsense2
echo "Step 1: Removing existing pyrealsense2..."
pip uninstall pyrealsense2 -y || true

# Install dependencies
echo "Step 2: Installing build dependencies..."
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    git \
    libssl-dev \
    libusb-1.0-0-dev \
    pkg-config \
    libgtk-3-dev \
    libglfw3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    python3-dev

# Clone librealsense
echo "Step 3: Cloning librealsense repository..."
cd ~
if [ -d "librealsense" ]; then
    echo "Removing existing librealsense directory..."
    rm -rf librealsense
fi

git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
git checkout v2.55.1

# Configure build
echo "Step 4: Configuring build..."
mkdir -p build
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PYTHON_BINDINGS=bool:true \
    -DPYTHON_EXECUTABLE=$(which python3) \
    -DBUILD_EXAMPLES=bool:false \
    -DBUILD_GRAPHICAL_EXAMPLES=bool:false

# Build
echo "Step 5: Building librealsense..."
echo "Using $MAKE_JOBS parallel jobs for faster compilation"
echo "You can monitor progress with: watch -n 2 'ps aux | grep make'"
make -j$MAKE_JOBS

# Install
echo "Step 6: Installing librealsense..."
sudo make install

# Update library cache so Python can find the libraries
echo "Step 6.5: Updating library cache..."
sudo ldconfig

# Install Python bindings
echo "Step 7: Installing Python bindings..."
cd ../wrappers/python

# Build Python bindings with proper library paths
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
python3 setup.py build_ext --inplace
python3 setup.py build
pip install .

# Test installation
echo "Step 8: Testing installation..."
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
if python3 -c "import pyrealsense2 as rs; print('✓ RealSense import successful!')"; then
    echo
    echo "=== BUILD SUCCESSFUL! ==="
    echo "RealSense is now installed and working."
    echo
    echo "Test your camera with:"
    echo "LD_LIBRARY_PATH=/usr/local/lib:\$LD_LIBRARY_PATH python3 -c \"import pyrealsense2 as rs; ctx = rs.context(); print(f'Found {len(ctx.query_devices())} device(s)')\""
    echo
    echo "💡 Note: You may need to add this to your ~/.bashrc:"
    echo "export LD_LIBRARY_PATH=/usr/local/lib:\$LD_LIBRARY_PATH"
    echo
else
    echo "❌ Installation test failed"
    echo "Try running: export LD_LIBRARY_PATH=/usr/local/lib:\$LD_LIBRARY_PATH"
    exit 1
fi

# Cleanup option
echo
read -p "Remove build directory to save space? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd ~
    rm -rf librealsense
    echo "✓ Build directory cleaned up"
fi

echo "Done! You can now use your ball balancing system with RealSense camera."

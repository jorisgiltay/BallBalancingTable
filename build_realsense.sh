#!/bin/bash
# build_realsense.sh - Automated RealSense build script for AGX Orin

set -e  # Exit on any error

echo "=== RealSense Build Script for AGX Orin ==="
echo "This will take 15-30 minutes depending on your system"
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
echo "Step 5: Building librealsense (this takes 15-30 minutes)..."
echo "You can monitor progress with: watch -n 5 'ps aux | grep make'"
make -j$(nproc)

# Install
echo "Step 6: Installing librealsense..."
sudo make install

# Install Python bindings
echo "Step 7: Installing Python bindings..."
cd ../wrappers/python
python3 setup.py build
pip install .

# Test installation
echo "Step 8: Testing installation..."
if python3 -c "import pyrealsense2 as rs; print('✓ RealSense import successful!')"; then
    echo
    echo "=== BUILD SUCCESSFUL! ==="
    echo "RealSense is now installed and working."
    echo
    echo "Test your camera with:"
    echo "python3 -c \"import pyrealsense2 as rs; ctx = rs.context(); print(f'Found {len(ctx.query_devices())} device(s)')\""
    echo
else
    echo "❌ Installation test failed"
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

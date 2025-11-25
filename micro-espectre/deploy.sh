#!/bin/bash
# Micro-ESPectre Deployment Script
# 
# Usage: ./deploy.sh [port] [--run]
# Example: ./deploy.sh /dev/cu.usbmodem14201
#          ./deploy.sh /dev/cu.usbmodem14201 --run
#
# Author: Francesco Pace <francesco.pace@gmail.com>
# License: GPLv3

set -e

# Parse arguments
PORT="/dev/cu.usbmodem*"
RUN_AFTER_DEPLOY=false

for arg in "$@"; do
    case $arg in
        --run)
            RUN_AFTER_DEPLOY=true
            ;;
        *)
            PORT="$arg"
            ;;
    esac
done

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                 Micro-ESPectre Deployment                 ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Check if mpremote is installed
if ! command -v mpremote &> /dev/null; then
    echo "❌ mpremote not found. Installing..."
    pip install mpremote
fi

# Check if config_local.py exists
if [ ! -f "config_local.py" ]; then
    echo "⚠️  config_local.py not found!"
    echo "   Please create it from config_local.py.example"
    echo ""
    echo "   cp config_local.py.example config_local.py"
    echo "   # Then edit config_local.py with your credentials"
    echo ""
    exit 1
fi

echo "📡 Connecting to ESP32 on $PORT..."
echo ""

# Upload files
echo "📤 Uploading files..."

# Upload src package
mpremote connect "$PORT" mkdir :src || true
mpremote connect "$PORT" mkdir :src/mqtt || true

mpremote connect "$PORT" cp src/__init__.py :src/ || { echo "❌ Failed to upload src/__init__.py"; exit 1; }
echo "   ✅ src/__init__.py"

mpremote connect "$PORT" cp src/config.py :src/ || { echo "❌ Failed to upload src/config.py"; exit 1; }
echo "   ✅ src/config.py"

mpremote connect "$PORT" cp src/segmentation.py :src/ || { echo "❌ Failed to upload src/segmentation.py"; exit 1; }
echo "   ✅ src/segmentation.py"

mpremote connect "$PORT" cp src/traffic_generator.py :src/ || { echo "❌ Failed to upload src/traffic_generator.py"; exit 1; }
echo "   ✅ src/traffic_generator.py"

mpremote connect "$PORT" cp src/nvs_storage.py :src/ || { echo "❌ Failed to upload src/nvs_storage.py"; exit 1; }
echo "   ✅ src/nvs_storage.py"

mpremote connect "$PORT" cp src/main.py :src/ || { echo "❌ Failed to upload src/main.py"; exit 1; }
echo "   ✅ src/main.py"

mpremote connect "$PORT" cp src/mqtt/__init__.py :src/mqtt/ || { echo "❌ Failed to upload src/mqtt/__init__.py"; exit 1; }
echo "   ✅ src/mqtt/__init__.py"

mpremote connect "$PORT" cp src/mqtt/handler.py :src/mqtt/ || { echo "❌ Failed to upload src/mqtt/handler.py"; exit 1; }
echo "   ✅ src/mqtt/handler.py"

mpremote connect "$PORT" cp src/mqtt/commands.py :src/mqtt/ || { echo "❌ Failed to upload src/mqtt/commands.py"; exit 1; }
echo "   ✅ src/mqtt/commands.py"

# Upload config_local.py to root
mpremote connect "$PORT" cp config_local.py : || { echo "❌ Failed to upload config_local.py"; exit 1; }
echo "   ✅ config_local.py"

echo ""
echo "✅ Deployment complete!"
echo ""

# Run application based on flags
if [ "$RUN_AFTER_DEPLOY" = true ]; then
    echo "🚀 Starting application..."
    echo ""
    mpremote connect "$PORT" run src/main.py
else
    echo "To run the application:"
    echo "  mpremote connect $PORT run src/main.py"
    echo ""
    echo "Or auto-run on next deployment:"
    echo "  ./deploy.sh $PORT --run        # Run src/main.py"
    echo ""
    echo "Or connect to REPL:"
    echo "  mpremote connect $PORT"
    echo "  >>> from src import main"
    echo "  >>> main.main()"
    echo ""
fi

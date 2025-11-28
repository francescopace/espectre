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
COLLECT_BASELINE=false
COLLECT_MOVEMENT=false

for arg in "$@"; do
    case $arg in
        --run)
            RUN_AFTER_DEPLOY=true
            ;;
        --collect-baseline)
            COLLECT_BASELINE=true
            ;;
        --collect-movement)
            COLLECT_MOVEMENT=true
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
mpremote connect "$PORT" cp src/config.py :src/ || { echo "❌ Failed to upload src/config.py"; exit 1; }
mpremote connect "$PORT" cp src/segmentation.py :src/ || { echo "❌ Failed to upload src/segmentation.py"; exit 1; }
mpremote connect "$PORT" cp src/traffic_generator.py :src/ || { echo "❌ Failed to upload src/traffic_generator.py"; exit 1; }
mpremote connect "$PORT" cp src/nvs_storage.py :src/ || { echo "❌ Failed to upload src/nvs_storage.py"; exit 1; }
mpremote connect "$PORT" cp src/filters.py :src/ || { echo "❌ Failed to upload src/filters.py"; exit 1; }
mpremote connect "$PORT" cp src/nbvi_calibrator.py :src/ || { echo "❌ Failed to upload src/nbvi_calibrator.py"; exit 1; }
mpremote connect "$PORT" cp src/main.py :src/ || { echo "❌ Failed to upload src/main.py"; exit 1; }
mpremote connect "$PORT" cp src/mqtt/__init__.py :src/mqtt/ || { echo "❌ Failed to upload src/mqtt/__init__.py"; exit 1; }
mpremote connect "$PORT" cp src/mqtt/handler.py :src/mqtt/ || { echo "❌ Failed to upload src/mqtt/handler.py"; exit 1; }
mpremote connect "$PORT" cp src/mqtt/commands.py :src/mqtt/ || { echo "❌ Failed to upload src/mqtt/commands.py"; exit 1; }
mpremote connect "$PORT" cp src/data_collector.py :src/ || { echo "❌ Failed to upload src/data_collector.py"; exit 1; }
mpremote connect "$PORT" cp config_local.py : || { echo "❌ Failed to upload config_local.py"; exit 1; }
echo ""
echo "✅ Deployment complete!"
echo ""

# Run application based on flags
if [ "$COLLECT_BASELINE" = true ]; then
    mpremote connect "$PORT" exec "from src import data_collector as dc; dc.collect_baseline()"
    echo ""
    echo "📥 Downloading baseline_data.bin..."
    mpremote connect "$PORT" cp :baseline_data.bin tools/baseline_data.bin
    echo "✅ Downloaded: baseline_data.bin"
    echo ""
elif [ "$COLLECT_MOVEMENT" = true ]; then
    mpremote connect "$PORT" exec "from src import data_collector as dc; dc.collect_movement()"
    echo ""
    echo "📥 Downloading movement_data.bin..."
    mpremote connect "$PORT" cp :movement_data.bin tools/movement_data.bin
    echo "✅ Downloaded: movement_data.bin"
    echo ""
elif [ "$RUN_AFTER_DEPLOY" = true ]; then
    echo "🚀 Starting application..."
    echo ""
    mpremote connect "$PORT" run src/main.py
else
    echo "Usage:"
    echo "  ./deploy.sh $PORT --run                # Run main application"
    echo "  ./deploy.sh $PORT --collect-baseline   # Collect baseline data"
    echo "  ./deploy.sh $PORT --collect-movement   # Collect movement data"
    echo ""
fi

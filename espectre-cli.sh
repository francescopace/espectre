#!/bin/bash
#
# ESPectre CLI - Interactive MQTT Control Interface
#

BROKER="${MQTT_BROKER:-homeassistant.local}"
PORT="${MQTT_PORT:-1883}"
TOPIC_CMD="${MQTT_TOPIC:-home/espectre/node1}/cmd"
TOPIC_RESPONSE="${MQTT_TOPIC:-home/espectre/node1}/response"
USERNAME="${MQTT_USERNAME:-mqtt}"
PASSWORD="${MQTT_PASSWORD:-mqtt}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# PID file for background listener
LISTENER_PID_FILE="/tmp/espectre-cli-listener.pid"

print_error() {
    echo -e "${RED}ERROR: $1${NC}" >&2
}

print_success() {
    echo -e "${GREEN}$1${NC}"
}

print_info() {
    echo -e "${BLUE}$1${NC}"
}

print_warning() {
    echo -e "${YELLOW}$1${NC}"
}

print_prompt() {
    echo -ne "${CYAN}espectre>${NC} "
}

check_dependencies() {
    if ! command -v mosquitto_pub &> /dev/null; then
        print_error "mosquitto_pub not found. Please install mosquitto-clients:"
        echo "  macOS:  brew install mosquitto"
        echo "  Ubuntu: sudo apt-get install mosquitto-clients"
        exit 1
    fi
}

# Build mosquitto command with auth
build_mqtt_cmd() {
    local base_cmd="$1"
    local cmd="$base_cmd -h $BROKER -p $PORT"
    [ -n "$USERNAME" ] && cmd="$cmd -u $USERNAME"
    [ -n "$PASSWORD" ] && cmd="$cmd -P $PASSWORD"
    echo "$cmd"
}

# Start background MQTT listener
start_listener() {
    local sub_cmd=$(build_mqtt_cmd "mosquitto_sub")
    
    # Start listener in background and save PID
    $sub_cmd -t "$TOPIC_RESPONSE" -v 2>/dev/null | while IFS= read -r line; do
        topic=$(echo "$line" | cut -d' ' -f1)
        message=$(echo "$line" | cut -d' ' -f2-)
        
        # Check message type
        local msg_type=$(echo "$message" | jq -r '.type' 2>/dev/null)
        
        # Emit beep based on message type (using macOS system sound)
        if echo "$message" | grep -q '"phase":"BASELINE"'; then
            # 1 beep for BASELINE phase start
            afplay /System/Library/Sounds/Ping.aiff 2>/dev/null &
        elif echo "$message" | grep -q '"phase":"MOVEMENT"'; then
            # 2 beeps for MOVEMENT phase start
            afplay /System/Library/Sounds/Ping.aiff 2>/dev/null &
            sleep 0.3
            afplay /System/Library/Sounds/Ping.aiff 2>/dev/null &
        elif echo "$message" | grep -q '"type":"calibration_complete"'; then
            # 3 beeps for calibration complete
            afplay /System/Library/Sounds/Ping.aiff 2>/dev/null &
            sleep 0.3
            afplay /System/Library/Sounds/Ping.aiff 2>/dev/null &
            sleep 0.3
            afplay /System/Library/Sounds/Ping.aiff 2>/dev/null &
            
            # Create completion marker for calibrate_with_save function
            touch "/tmp/espectre-calibration-complete" 2>/dev/null
        fi
        
        # Clear current line and print response
        echo -ne "\r\033[K"
        echo -n "$(date '+%H:%M:%S') "
        echo "$message" | jq '.' 2>/dev/null || echo "$message"
        print_prompt
    done &
    
    echo $! > "$LISTENER_PID_FILE"
}

# Stop background listener
stop_listener() {
    if [ -f "$LISTENER_PID_FILE" ]; then
        local pid=$(cat "$LISTENER_PID_FILE")
        # Kill the listener and its child processes
        pkill -P $pid 2>/dev/null
        kill $pid 2>/dev/null
        rm -f "$LISTENER_PID_FILE"
    fi
}

# Cleanup on exit
cleanup() {
    # Prevent multiple executions
    if [ -n "$CLEANUP_DONE" ]; then
        return
    fi
    CLEANUP_DONE=1
    
    echo ""
    print_info "Shutting down..."
    stop_listener
    exit 0
}

send_command() {
    local cmd_json="$1"
    
    local pub_cmd=$(build_mqtt_cmd "mosquitto_pub")
    $pub_cmd -t "$TOPIC_CMD" -m "$cmd_json" 2>/dev/null
    
    if [ $? -ne 0 ]; then
        print_error "Failed to send command to broker"
        return 1
    fi
}

# Helper for toggle commands (on/off/true/false)
send_toggle_command() {
    local cmd_name="$1"
    local enabled="$2"
    
    if [ "$enabled" = "on" ] || [ "$enabled" = "true" ] || [ "$enabled" = "1" ]; then
        send_command "{\"cmd\":\"$cmd_name\",\"enabled\":true}"
    elif [ "$enabled" = "off" ] || [ "$enabled" = "false" ] || [ "$enabled" = "0" ]; then
        send_command "{\"cmd\":\"$cmd_name\",\"enabled\":false}"
    else
        print_error "Usage: $cmd_name <on|off>"
        return 1
    fi
}

cmd_segmentation_threshold() {
    local value="$1"
    
    if [ -z "$value" ]; then
        print_error "Usage: segmentation_threshold <value>"
        echo "  Range: 0.5-10.0 (MVS adaptive threshold)"
        echo "  Example: segmentation_threshold 2.5"
        return 1
    fi
    
    send_command "{\"cmd\":\"segmentation_threshold\",\"value\":$value}"
}

cmd_features_enable() {
    local enabled="$1"
    send_toggle_command "features_enable" "$enabled"
}

cmd_info() {
    send_command "{\"cmd\":\"info\"}"
}

cmd_stats() {
    send_command "{\"cmd\":\"stats\"}"
}

cmd_hampel_filter() {
    local enabled="$1"
    send_toggle_command "hampel_filter" "$enabled"
}

cmd_hampel_threshold() {
    local value="$1"
    
    if [ -z "$value" ]; then
        print_error "Usage: hampel_threshold <value>"
        echo "  Range: 1.0-10.0 (MAD multiplier)"
        echo "  Example: hampel_threshold 3.0"
        return 1
    fi
    
    send_command "{\"cmd\":\"hampel_threshold\",\"value\":$value}"
}

cmd_savgol_filter() {
    local enabled="$1"
    send_toggle_command "savgol_filter" "$enabled"
}

cmd_butterworth_filter() {
    local enabled="$1"
    send_toggle_command "butterworth_filter" "$enabled"
}

cmd_wavelet_filter() {
    local enabled="$1"
    send_toggle_command "wavelet_filter" "$enabled"
}

cmd_wavelet_level() {
    local value="$1"
    
    if [ -z "$value" ]; then
        print_error "Usage: wavelet_level <level>"
        echo "  Range: 1-3 (decomposition level)"
        echo "  1 = minimal denoising, fastest"
        echo "  2 = moderate denoising"
        echo "  3 = maximum denoising (recommended)"
        echo "  Example: wavelet_level 3"
        return 1
    fi
    
    send_command "{\"cmd\":\"wavelet_level\",\"value\":$value}"
}

cmd_wavelet_threshold() {
    local value="$1"
    
    if [ -z "$value" ]; then
        print_error "Usage: wavelet_threshold <value>"
        echo "  Range: 0.5-2.0 (noise threshold)"
        echo "  0.5 = minimal noise removal"
        echo "  1.0 = balanced (recommended)"
        echo "  2.0 = aggressive noise removal"
        echo "  Example: wavelet_threshold 1.0"
        return 1
    fi
    
    send_command "{\"cmd\":\"wavelet_threshold\",\"value\":$value}"
}

cmd_smart_publishing() {
    local enabled="$1"
    send_toggle_command "smart_publishing" "$enabled"
}


cmd_traffic_generator_rate() {
    local value="$1"
    
    if [ -z "$value" ]; then
        print_error "Usage: traffic_generator_rate <packets_per_sec>"
        echo "  Range: 0-50 (0=disabled, recommended: 15)"
        echo "  Generates WiFi traffic for continuous CSI packets"
        echo "  Example: traffic_generator_rate 15"
        return 1
    fi
    
    send_command "{\"cmd\":\"traffic_generator_rate\",\"value\":$value}"
}

cmd_factory_reset() {
    print_warning "⚠️  WARNING: This will reset ALL settings to factory defaults!"
    print_warning "This includes:"
    print_warning "  - Detection parameters (threshold, debounce, etc.)"
    print_warning "  - Filter settings"
    print_warning "  - Calibration data"
    print_warning "  - All saved configurations"
    echo ""
    read -p "Are you sure you want to continue? (yes/no): " confirm
    
    if [ "$confirm" = "yes" ]; then
        print_info "Performing factory reset..."
        send_command "{\"cmd\":\"factory_reset\"}"
    else
        print_info "Factory reset cancelled"
    fi
}

show_help() {
    echo ""
    echo -e "${CYAN}ESPectre CLI - Interactive Commands${NC}"
    echo ""
    echo -e "${YELLOW}Segmentation Commands:${NC}"
    echo "  segmentation_threshold <val> Set segmentation threshold (0.5-10.0)"
    echo "  features_enable <on|off>  Enable/disable feature extraction during MOTION"
    echo ""
    echo -e "${YELLOW}Filter Commands:${NC}"
    echo "  butterworth_filter <on|off> Enable/disable Butterworth filter (high freq)"
    echo "  wavelet_filter <on|off>   Enable/disable Wavelet filter (low freq)"
    echo "  wavelet_level <1-3>       Set wavelet decomposition level (rec: 3)"
    echo "  wavelet_threshold <val>   Set wavelet threshold (0.5-2.0, rec: 1.0)"
    echo "  hampel_filter <on|off>    Enable/disable Hampel outlier filter"
    echo "  hampel_threshold <val>    Set Hampel threshold (1.0-10.0)"
    echo "  savgol_filter <on|off>    Enable/disable Savitzky-Golay filter"
    echo ""
    echo -e "${YELLOW}State Commands:${NC}"
    echo "  smart_publishing <on|off> Enable/disable smart publishing"
    echo ""
    echo -e "${YELLOW}Information Commands:${NC}"
    echo "  info                      Show current configuration (static)"
    echo "  stats                     Show runtime statistics (dynamic)"
    echo ""
    echo -e "${YELLOW}Traffic Generator:${NC}"
    echo "  traffic_generator_rate <pps> Set traffic rate (0=off, 5-100, rec: 20)"
    echo ""
    echo -e "${YELLOW}Utility Commands:${NC}"
    echo "  factory_reset             Reset all settings to factory defaults"
    echo "  clear                     Clear screen"
    echo "  help                      Show this help message"
    echo "  exit, quit                Exit interactive mode"
    echo ""
    echo -e "${YELLOW}Shortcuts:${NC}"
    echo "  st, fe, i, s, hampel, sg, bw, wv, wvl, wvt, sp, tgr, fr"
    echo ""
    echo -e "${YELLOW}Environment Variables:${NC}"
    echo "  MQTT_BROKER               MQTT broker hostname (default: homeassistant.local)"
    echo "  MQTT_PORT                 MQTT broker port (default: 1883)"
    echo "  MQTT_TOPIC                Base MQTT topic (default: home/espectre/node1)"
    echo "  MQTT_USERNAME             MQTT username (default: mqtt)"
    echo "  MQTT_PASSWORD             MQTT password (default: mqtt)"
    echo ""
}

# Process command in interactive mode
process_command() {
    local input="$1"
    
    # Trim whitespace
    input=$(echo "$input" | xargs)
    
    # Skip empty input
    [ -z "$input" ] && return 0
    
    # Parse command and arguments
    local cmd=$(echo "$input" | awk '{print $1}')
    local args=$(echo "$input" | cut -d' ' -f2- -s)
    
    case "$cmd" in
        segmentation_threshold|st)
            cmd_segmentation_threshold $args
            ;;
        features_enable|fe)
            cmd_features_enable $args
            ;;
        info|i)
            cmd_info
            ;;
        stats|s)
            cmd_stats
            ;;
        hampel_filter|hampel)
            cmd_hampel_filter $args
            ;;
        hampel_threshold|ht)
            cmd_hampel_threshold $args
            ;;
        savgol_filter|savgol|sg)
            cmd_savgol_filter $args
            ;;
        butterworth_filter|butterworth|bw)
            cmd_butterworth_filter $args
            ;;
        wavelet_filter|wavelet|wv)
            cmd_wavelet_filter $args
            ;;
        wavelet_level|wvl)
            cmd_wavelet_level $args
            ;;
        wavelet_threshold|wvt)
            cmd_wavelet_threshold $args
            ;;
        smart_publishing|smart|sp)
            cmd_smart_publishing $args
            ;;
        traffic_generator_rate|tgr|traffic)
            cmd_traffic_generator_rate $args
            ;;
        factory_reset|reset|fr)
            cmd_factory_reset
            ;;
        clear|cls)
            clear
            ;;
        help|h)
            show_help
            ;;
        exit|quit|q)
            cleanup
            ;;
        "")
            # Empty command, do nothing
            ;;
        *)
            print_error "Unknown command: $cmd"
            echo "Type 'help' for available commands"
            ;;
    esac
}

# Main interactive mode
main() {
    check_dependencies
    
    # Setup cleanup trap
    trap cleanup SIGINT SIGTERM EXIT
    
    # Print banner
    clear
    echo -e "${MAGENTA}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║                                                           ║"
    echo "║                   🛜  E S P e c t r e 👻                   ║"
    echo "║                                                           ║"
    echo "║                Wi-Fi motion detection system              ║"
    echo "║          based on Channel State Information (CSI)         ║"
    echo "║                                                           ║"
    echo "╠═══════════════════════════════════════════════════════════╣"
    echo "║                                                           ║"
    echo "║                   Interactive CLI Mode                    ║"
    echo "║                                                           ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    print_info "Connected to: $BROKER:$PORT"
    print_info "Command topic: $TOPIC_CMD"
    print_info "Listening on: $TOPIC_RESPONSE"
    echo ""
    print_warning "Type 'help' for commands, 'exit' to quit"
    echo ""
    
    # Start background listener
    start_listener
    
    # Give listener time to start
    sleep 0.5
    
    # Main interactive loop
    while true; do
        print_prompt
        read -r input
        
        # Process command
        process_command "$input"
    done
}

main "$@"

#!/bin/bash
#
# ESPectre CLI - MQTT Command Line Interface
# Control your ESPectre device via MQTT
#

# Configuration
BROKER="${MQTT_BROKER:-homeassistant.local}"
PORT="${MQTT_PORT:-1883}"
TOPIC_CMD="${MQTT_TOPIC:-home/pisense/node1}/cmd"
TOPIC_RESPONSE="${MQTT_TOPIC:-home/pisense/node1}/response"
USERNAME="${MQTT_USERNAME:-mqtt}"
PASSWORD="${MQTT_PASSWORD:-mqtt}"
TIMEOUT=5

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Check if mosquitto clients are installed
check_dependencies() {
    if ! command -v mosquitto_pub &> /dev/null; then
        print_error "mosquitto_pub not found. Please install mosquitto-clients:"
        echo "  macOS:  brew install mosquitto"
        echo "  Ubuntu: sudo apt-get install mosquitto-clients"
        exit 1
    fi
}

# Send command and wait for response
send_command() {
    local cmd_json="$1"
    local wait_response="${2:-true}"
    
    # Build mosquitto_pub command with optional credentials
    local pub_cmd="mosquitto_pub -h $BROKER -p $PORT"
    if [ -n "$USERNAME" ]; then
        pub_cmd="$pub_cmd -u $USERNAME"
    fi
    if [ -n "$PASSWORD" ]; then
        pub_cmd="$pub_cmd -P $PASSWORD"
    fi
    
    # Send command
    $pub_cmd -t "$TOPIC_CMD" -m "$cmd_json" 2>/dev/null
    
    if [ $? -ne 0 ]; then
        print_error "Failed to send command to broker"
        return 1
    fi
    
    if [ "$wait_response" = "true" ]; then
        print_info "Waiting for response..."
        
        # Build mosquitto_sub command with optional credentials
        local sub_cmd="mosquitto_sub -h $BROKER -p $PORT"
        if [ -n "$USERNAME" ]; then
            sub_cmd="$sub_cmd -u $USERNAME"
        fi
        if [ -n "$PASSWORD" ]; then
            sub_cmd="$sub_cmd -P $PASSWORD"
        fi
        
        # Listen for response with timeout
        timeout $TIMEOUT $sub_cmd -t "$TOPIC_RESPONSE" -C 1 2>/dev/null | \
        while IFS= read -r response; do
            echo "$response" | jq -r 'if .response then .response else . end' 2>/dev/null || echo "$response"
        done
        
        if [ ${PIPESTATUS[0]} -eq 124 ]; then
            print_warning "Timeout waiting for response"
            return 1
        fi
    fi
}

# Command: Set threshold
cmd_threshold() {
    if [ -z "$1" ]; then
        print_error "Usage: $0 threshold <value>"
        echo "  Example: $0 threshold 0.05"
        return 1
    fi
    
    local value="$1"
    send_command "{\"cmd\":\"threshold\",\"value\":$value}"
}

# Command: Get statistics
cmd_stats() {
    send_command "{\"cmd\":\"stats\"}"
}

# Command: Get info
cmd_info() {
    send_command "{\"cmd\":\"info\"}"
}

# Command: Toggle logs
cmd_logs() {
    local enabled="${1:-toggle}"
    
    if [ "$enabled" = "on" ] || [ "$enabled" = "true" ] || [ "$enabled" = "1" ]; then
        send_command "{\"cmd\":\"logs\",\"enabled\":true}"
    elif [ "$enabled" = "off" ] || [ "$enabled" = "false" ] || [ "$enabled" = "0" ]; then
        send_command "{\"cmd\":\"logs\",\"enabled\":false}"
    else
        print_error "Usage: $0 logs <on|off>"
        return 1
    fi
}

# Command: Analyze
cmd_analyze() {
    send_command "{\"cmd\":\"analyze\"}"
}

# Command: Set persistence timeout
cmd_persistence() {
    if [ -z "$1" ]; then
        print_error "Usage: $0 persistence <seconds>"
        echo "  Example: $0 persistence 2"
        return 1
    fi
    
    local value="$1"
    send_command "{\"cmd\":\"persistence\",\"value\":$value}"
}

# Command: Set debounce count
cmd_debounce() {
    if [ -z "$1" ]; then
        print_error "Usage: $0 debounce <count>"
        echo "  Example: $0 debounce 3"
        return 1
    fi
    
    local value="$1"
    send_command "{\"cmd\":\"debounce\",\"value\":$value}"
}

# Command: Set hysteresis ratio
cmd_hysteresis() {
    if [ -z "$1" ]; then
        print_error "Usage: $0 hysteresis <ratio>"
        echo "  Example: $0 hysteresis 0.7"
        return 1
    fi
    
    local value="$1"
    send_command "{\"cmd\":\"hysteresis\",\"value\":$value}"
}

# Command: Set variance scale
cmd_variance_scale() {
    if [ -z "$1" ]; then
        print_error "Usage: $0 variance_scale <value>"
        echo "  Example: $0 variance_scale 400"
        echo "  Lower values = higher sensitivity"
        return 1
    fi
    
    local value="$1"
    send_command "{\"cmd\":\"variance_scale\",\"value\":$value}"
}

# Command: Show advanced features
cmd_features() {
    send_command "{\"cmd\":\"features\"}"
}

# Command: Show feature weights
cmd_weights() {
    send_command "{\"cmd\":\"weights\"}"
}

# Command: Enable/disable granular states
cmd_granular_states() {
    local enabled="${1:-toggle}"
    
    if [ "$enabled" = "on" ] || [ "$enabled" = "true" ] || [ "$enabled" = "1" ]; then
        send_command "{\"cmd\":\"granular_states\",\"enabled\":true}"
    elif [ "$enabled" = "off" ] || [ "$enabled" = "false" ] || [ "$enabled" = "0" ]; then
        send_command "{\"cmd\":\"granular_states\",\"enabled\":false}"
    else
        print_error "Usage: $0 granular_states <on|off>"
        return 1
    fi
}

# Command: Enable/disable Hampel filter
cmd_hampel_filter() {
    local enabled="${1:-toggle}"
    
    if [ "$enabled" = "on" ] || [ "$enabled" = "true" ] || [ "$enabled" = "1" ]; then
        send_command "{\"cmd\":\"hampel_filter\",\"enabled\":true}"
    elif [ "$enabled" = "off" ] || [ "$enabled" = "false" ] || [ "$enabled" = "0" ]; then
        send_command "{\"cmd\":\"hampel_filter\",\"enabled\":false}"
    else
        print_error "Usage: $0 hampel_filter <on|off>"
        return 1
    fi
}

# Command: Set Hampel threshold
cmd_hampel_threshold() {
    if [ -z "$1" ]; then
        print_error "Usage: $0 hampel_threshold <value>"
        echo "  Range: 1.0-10.0 (MAD multiplier)"
        echo "  Example: $0 hampel_threshold 3.0"
        return 1
    fi
    
    local value="$1"
    send_command "{\"cmd\":\"hampel_threshold\",\"value\":$value}"
}

# Command: Enable/disable Savitzky-Golay filter
cmd_savgol_filter() {
    local enabled="${1:-toggle}"
    
    if [ "$enabled" = "on" ] || [ "$enabled" = "true" ] || [ "$enabled" = "1" ]; then
        send_command "{\"cmd\":\"savgol_filter\",\"enabled\":true}"
    elif [ "$enabled" = "off" ] || [ "$enabled" = "false" ] || [ "$enabled" = "0" ]; then
        send_command "{\"cmd\":\"savgol_filter\",\"enabled\":false}"
    else
        print_error "Usage: $0 savgol_filter <on|off>"
        return 1
    fi
}

# Command: Show filter status
cmd_filters() {
    send_command "{\"cmd\":\"filters\"}"
}

# Command: Set weight_variance
cmd_weight_variance() {
    if [ -z "$1" ]; then
        print_error "Usage: $0 weight_variance <value>"
        echo "  Range: 0.0-1.0"
        echo "  Example: $0 weight_variance 0.35"
        return 1
    fi
    
    local value="$1"
    send_command "{\"cmd\":\"weight_variance\",\"value\":$value}"
}

# Command: Set weight_spatial_gradient
cmd_weight_spatial_gradient() {
    if [ -z "$1" ]; then
        print_error "Usage: $0 weight_spatial_gradient <value>"
        echo "  Range: 0.0-1.0"
        echo "  Example: $0 weight_spatial_gradient 0.30"
        return 1
    fi
    
    local value="$1"
    send_command "{\"cmd\":\"weight_spatial_gradient\",\"value\":$value}"
}

# Command: Set weight_variance_short
cmd_weight_variance_short() {
    if [ -z "$1" ]; then
        print_error "Usage: $0 weight_variance_short <value>"
        echo "  Range: 0.0-1.0"
        echo "  Example: $0 weight_variance_short 0.25"
        return 1
    fi
    
    local value="$1"
    send_command "{\"cmd\":\"weight_variance_short\",\"value\":$value}"
}

# Command: Set weight_iqr
cmd_weight_iqr() {
    if [ -z "$1" ]; then
        print_error "Usage: $0 weight_iqr <value>"
        echo "  Range: 0.0-1.0"
        echo "  Example: $0 weight_iqr 0.10"
        return 1
    fi
    
    local value="$1"
    send_command "{\"cmd\":\"weight_iqr\",\"value\":$value}"
}

# Command: Listen to responses
cmd_listen() {
    print_info "Listening to responses on $TOPIC_RESPONSE (Ctrl+C to stop)..."
    
    # Build mosquitto_sub command with optional credentials
    local sub_cmd="mosquitto_sub -h $BROKER -p $PORT"
    if [ -n "$USERNAME" ]; then
        sub_cmd="$sub_cmd -u $USERNAME"
    fi
    if [ -n "$PASSWORD" ]; then
        sub_cmd="$sub_cmd -P $PASSWORD"
    fi
    
    $sub_cmd -t "$TOPIC_RESPONSE" -v | \
    while IFS= read -r line; do
        # Extract topic and message
        topic=$(echo "$line" | cut -d' ' -f1)
        message=$(echo "$line" | cut -d' ' -f2-)
        
        # Pretty print JSON if possible
        echo -n "$(date '+%H:%M:%S') "
        echo "$message" | jq '.' 2>/dev/null || echo "$message"
    done
}

# Command: Monitor data stream
cmd_monitor() {
    local data_topic="${MQTT_TOPIC:-home/espectre/node1}"
    print_info "Monitoring data stream on $data_topic (Ctrl+C to stop)..."
    print_info "Note: Smart publishing may filter some messages. Use 'logs on' to see all CSI values."
    
    # Build mosquitto_sub command with optional credentials
    local sub_cmd="mosquitto_sub -h $BROKER -p $PORT"
    if [ -n "$USERNAME" ]; then
        sub_cmd="$sub_cmd -u $USERNAME"
    fi
    if [ -n "$PASSWORD" ]; then
        sub_cmd="$sub_cmd -P $PASSWORD"
    fi
    
    # Subscribe to both data and response topics to see everything
    $sub_cmd -t "$data_topic" -t "${data_topic}/response" -v | \
    while IFS= read -r line; do
        # Extract topic and message
        topic=$(echo "$line" | cut -d' ' -f1)
        message=$(echo "$line" | cut -d' ' -f2-)
        
        # Pretty print JSON if possible
        echo -n "$(date '+%H:%M:%S') [$topic] "
        if echo "$message" | jq -e . >/dev/null 2>&1; then
            # It's valid JSON
            if echo "$message" | jq -e '.state' >/dev/null 2>&1; then
                # It's a data message
                echo "$message" | jq -c '{state, movement, confidence, threshold, baseline}'
            else
                # It's a response message
                echo "$message" | jq -c '.'
            fi
        else
            echo "$message"
        fi
    done
}

# Show help
show_help() {
    cat << EOF
ESPectre CLI - MQTT Command Line Interface

Usage: $0 <command> [arguments]

Commands:
  threshold <value>    Set detection threshold (0.0-1.0)
                       Example: $0 threshold 0.05
  
  persistence <sec>    Set persistence timeout (1-30 seconds)
                       Time to wait before returning to IDLE
                       Example: $0 persistence 2
  
  debounce <count>     Set debounce count (1-10)
                       Consecutive detections needed
                       Example: $0 debounce 3
  
  hysteresis <ratio>   Set hysteresis ratio (0.1-1.0)
                       Lower threshold = high * ratio
                       Example: $0 hysteresis 0.7
  
  variance_scale <val> Set variance scale (100-2000)
                       Lower = higher sensitivity
                       Example: $0 variance_scale 400
  
  features             Show all extracted CSI features (17 features)
                       Always available
                       Example: $0 features
  
  weights              Show current feature weights
                       Example: $0 weights
  
  granular_states <on|off> Enable/disable 5-state detection
                       OFF: IDLE, DETECTED (default)
                       ON: IDLE, MICRO, DETECTED, INTENSE
                       Example: $0 granular_states on
  
  hampel_filter <on|off> Enable/disable Hampel outlier filter
                       Removes outliers using MAD method
                       Example: $0 hampel_filter on
  
  hampel_threshold <val> Set Hampel filter threshold (1.0-10.0)
                       MAD multiplier for outlier detection
                       Example: $0 hampel_threshold 3.0
  
  savgol_filter <on|off> Enable/disable Savitzky-Golay smoothing
                       Polynomial smoothing filter
                       Example: $0 savgol_filter on
  
  filters              Show current filter status and statistics
                       Example: $0 filters
  
  weight_variance <val>    Set variance feature weight (0.0-1.0)
                       Example: $0 weight_variance 0.35
  
  weight_spatial_gradient <val> Set spatial gradient weight (0.0-1.0)
                       Example: $0 weight_spatial_gradient 0.30
  
  weight_variance_short <val> Set short variance weight (0.0-1.0)
                       Example: $0 weight_variance_short 0.25
  
  weight_iqr <val>     Set IQR feature weight (0.0-1.0)
                       Example: $0 weight_iqr 0.10
  
  stats                Show CSI statistics
  
  info                 Show current configuration
  
  logs <on|off>        Enable/disable CSI logs
  
  analyze              Analyze data and suggest threshold
  
  listen               Listen to command responses
  
  monitor              Monitor real-time data stream
  
  help                 Show this help message

Environment Variables:
  MQTT_BROKER          MQTT broker hostname (default: homeassistant.local)
  MQTT_PORT            MQTT broker port (default: 1883)
  MQTT_TOPIC           Base MQTT topic (default: home/espectre/node1)
  MQTT_USERNAME        MQTT username (optional)
  MQTT_PASSWORD        MQTT password (optional)

Examples:
  $0 threshold 0.05    # Set threshold to 0.05
  $0 stats             # Get statistics
  $0 logs on           # Enable CSI logs
  $0 monitor           # Monitor data stream

EOF
}

# Main
main() {
    check_dependencies
    
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi
    
    local command="$1"
    shift
    
    case "$command" in
        threshold|t)
            cmd_threshold "$@"
            ;;
        stats|s)
            cmd_stats
            ;;
        info|i)
            cmd_info
            ;;
        logs|l)
            cmd_logs "$@"
            ;;
        analyze|a)
            cmd_analyze
            ;;
        persistence|p)
            cmd_persistence "$@"
            ;;
        debounce|d)
            cmd_debounce "$@"
            ;;
        hysteresis|hyst)
            cmd_hysteresis "$@"
            ;;
        variance_scale|var|sensitivity)
            cmd_variance_scale "$@"
            ;;
        features|f)
            cmd_features
            ;;
        weights|w)
            cmd_weights
            ;;
        granular_states|gs)
            cmd_granular_states "$@"
            ;;
        hampel_filter|hampel)
            cmd_hampel_filter "$@"
            ;;
        hampel_threshold|ht)
            cmd_hampel_threshold "$@"
            ;;
        savgol_filter|savgol|sg)
            cmd_savgol_filter "$@"
            ;;
        filters)
            cmd_filters
            ;;
        weight_variance|wv)
            cmd_weight_variance "$@"
            ;;
        weight_spatial_gradient|wsg)
            cmd_weight_spatial_gradient "$@"
            ;;
        weight_variance_short|wvs)
            cmd_weight_variance_short "$@"
            ;;
        weight_iqr|wiqr)
            cmd_weight_iqr "$@"
            ;;
        listen)
            cmd_listen
            ;;
        monitor|m)
            cmd_monitor
            ;;
        help|h|-h|--help)
            show_help
            ;;
        *)
            print_error "Unknown command: $command"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

main "$@"

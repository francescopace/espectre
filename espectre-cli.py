#!/usr/bin/env python3
import os
import sys
import json
import time
import argparse
import shlex
import webbrowser
from datetime import datetime
from typing import Dict, Any

try:
    import yaml
    import paho.mqtt.client as mqtt
    from colorama import init, Fore, Style
    from dotenv import load_dotenv
    from prompt_toolkit import PromptSession, print_formatted_text
    from prompt_toolkit.completion import NestedCompleter
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.styles import Style as PromptStyle
    from prompt_toolkit.formatted_text import HTML, FormattedText
    from prompt_toolkit.lexers import PygmentsLexer
    from pygments.lexers.data import YamlLexer
except ImportError as e:
    print(f"Error: Missing dependency {e.name}. Please install requirements.txt")
    print("pip install -r requirements.txt")
    sys.exit(1)

# Initialize colorama
init()

# Load environment variables
load_dotenv()

# Custom Dumper to force inline lists
class CompactDumper(yaml.SafeDumper):
    pass

def represent_list(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

CompactDumper.add_representer(list, represent_list)

class EspectreCLI:
    def __init__(self):
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="ESPectre CLI - Interactive MQTT Control Interface")
        parser.add_argument("--broker", default=os.getenv("MQTT_BROKER", "homeassistant.local"), help="MQTT broker hostname")
        parser.add_argument("--port", type=int, default=int(os.getenv("MQTT_PORT", 1883)), help="MQTT broker port")
        parser.add_argument("--topic", default=os.getenv("MQTT_TOPIC", "home/espectre/node1"), help="Base MQTT topic")
        parser.add_argument("--username", default=os.getenv("MQTT_USERNAME", "mqtt"), help="MQTT username")
        parser.add_argument("--password", default=os.getenv("MQTT_PASSWORD", "mqtt"), help="MQTT password")
        args = parser.parse_args()

        self.broker = args.broker
        self.port = args.port
        self.base_topic = args.topic
        self.username = args.username
        self.password = args.password
        
        self.topic_cmd = f"{self.base_topic}/cmd"
        self.topic_response = f"{self.base_topic}/response"
        
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        if self.username and self.password:
            self.client.username_pw_set(self.username, self.password)
            
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        
        self.running = True

        # Setup prompt_toolkit session with history and completion
        hist_file = os.path.join(os.path.expanduser("~"), ".espectre_cli_history")
        
        # Create nested completer for commands and their arguments (only full names, no shortcuts)
        completer_dict = {
            'segmentation_threshold': None,
            'segmentation_window_size': None,
            'subcarrier_selection': None,
            'features_enable': {'on': None, 'off': None},
            'info': None,
            'stats': None,
            'hampel_filter': {'on': None, 'off': None},
            'hampel_threshold': None,
            'savgol_filter': {'on': None, 'off': None},
            'butterworth_filter': {'on': None, 'off': None},
            'wavelet_filter': {'on': None, 'off': None},
            'wavelet_level': {'1': None, '2': None, '3': None},
            'wavelet_threshold': None,
            'smart_publishing': {'on': None, 'off': None},
            'traffic_generator_rate': None,
            'factory_reset': None,
            'webui': None,
            'clear': None,
            'help': None,
            'about': None,
            'exit': None,
        }
        
        # Custom style for the prompt
        prompt_style = PromptStyle.from_dict({
            'prompt': '#00aa00 bold',
        })
        
        self.session = PromptSession(
            history=FileHistory(hist_file),
            completer=NestedCompleter.from_nested_dict(completer_dict),
            style=prompt_style,
            complete_while_typing=True,
            enable_history_search=True,
        )


    def on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            print(f"{Fore.BLUE}Connected to: {self.broker}:{self.port}{Style.RESET_ALL}")
            print(f"{Fore.BLUE}Command topic: {self.topic_cmd}{Style.RESET_ALL}")
            print(f"{Fore.BLUE}Listening on: {self.topic_response}{Style.RESET_ALL}")
            client.subscribe(self.topic_response)
        else:
            print(f"{Fore.RED}Failed to connect, return code {rc}{Style.RESET_ALL}")

    def on_message(self, client, userdata, msg):
        try:
            payload = msg.payload.decode()
            data = json.loads(payload)
            
            # Print message
            timestamp = datetime.now().strftime("%H:%M:%S")
            print() # Add spacing between command and response
            
            # Format as YAML with inline lists
            formatted_yaml = yaml.dump(data, Dumper=CompactDumper, sort_keys=False, default_flow_style=False, width=1000)
            
            # Use Pygments for syntax highlighting (built into prompt_toolkit)
            print(f"{Fore.GREEN}[{timestamp}]{Style.RESET_ALL} Received:")
            print_formatted_text(
                FormattedText([("class:pygments", formatted_yaml)]),
                style=PromptStyle.from_dict({'pygments': '#ansiwhite'})
            )
            print() # Add spacing after response
            
        except Exception as e:
            print(f"\nError parsing message: {e}")





    def send_command(self, cmd_data: Dict[str, Any]):
        try:
            payload = json.dumps(cmd_data)
            self.client.publish(self.topic_cmd, payload)
        except Exception as e:
            print(f"{Fore.RED}Error sending command: {e}{Style.RESET_ALL}")

    def start(self):
        print(f"{Fore.MAGENTA}")
        print("╔═══════════════════════════════════════════════════════════╗")
        print("║                                                           ║")
        print("║                   🛜  E S P e c t r e 👻                   ║")
        print("║                                                           ║")
        print("║                Wi-Fi motion detection system              ║")
        print("║          based on Channel State Information (CSI)         ║")
        print("║                                                           ║")
        print("╠═══════════════════════════════════════════════════════════╣")
        print("║                                                           ║")
        print("║                   Interactive CLI Mode                    ║")
        print("║                                                           ║")
        print("╚═══════════════════════════════════════════════════════════╝")
        print(f"{Style.RESET_ALL}")

        try:
            self.client.connect(self.broker, self.port, 60)
            self.client.loop_start()
            
            # Wait a bit for connection
            time.sleep(0.5)
            
            print(f"\n{Fore.YELLOW}Type 'help' for commands, 'exit' to quit{Style.RESET_ALL}\n")
            print(f"{Fore.YELLOW}Tip: Use TAB for autocompletion, Ctrl+R to search history{Style.RESET_ALL}\n")
            
            while self.running:
                try:
                    user_input = self.session.prompt(HTML('<prompt>espectre></prompt> '))
                    self.process_input(user_input)
                except KeyboardInterrupt:
                    continue
                except EOFError:
                    break
                    
        except Exception as e:
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        finally:
            self.client.loop_stop()
            self.client.disconnect()
            print("\nExiting...")

    def process_input(self, user_input):
        if not user_input.strip():
            return

        parts = shlex.split(user_input)
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in ["exit", "quit", "q"]:
            self.running = False
            return
        elif cmd in ["help", "h"]:
            self.show_help()
            return
        elif cmd == "about":
            self.show_about()
            return
        elif cmd in ["clear", "cls"]:
            os.system('cls' if os.name == 'nt' else 'clear')
            return
        elif cmd in ["webui", "web", "ui"]:
            self.cmd_webui()
            return
        
        # Command mappings
        try:
            if cmd in ["segmentation_threshold", "st"]:
                self.cmd_segmentation_threshold(args)
            elif cmd in ["segmentation_window_size", "sws"]:
                self.cmd_segmentation_window_size(args)
            elif cmd in ["subcarrier_selection", "scs"]:
                self.cmd_subcarrier_selection(args)
            elif cmd in ["features_enable", "fe"]:
                self.cmd_features_enable(args)
            elif cmd in ["info", "i"]:
                self.send_command({"cmd": "info"})
            elif cmd in ["stats", "s"]:
                self.send_command({"cmd": "stats"})
            elif cmd in ["hampel_filter", "hampel"]:
                self.cmd_toggle("hampel_filter", args)
            elif cmd in ["hampel_threshold", "ht"]:
                self.cmd_value("hampel_threshold", args, float)
            elif cmd in ["savgol_filter", "savgol", "sg"]:
                self.cmd_toggle("savgol_filter", args)
            elif cmd in ["butterworth_filter", "butterworth", "bw"]:
                self.cmd_toggle("butterworth_filter", args)
            elif cmd in ["wavelet_filter", "wavelet", "wv"]:
                self.cmd_toggle("wavelet_filter", args)
            elif cmd in ["wavelet_level", "wvl"]:
                self.cmd_value("wavelet_level", args, int)
            elif cmd in ["wavelet_threshold", "wvt"]:
                self.cmd_value("wavelet_threshold", args, float)
            elif cmd in ["smart_publishing", "smart", "sp"]:
                self.cmd_toggle("smart_publishing", args)
            elif cmd in ["traffic_generator_rate", "tgr", "traffic"]:
                self.cmd_value("traffic_generator_rate", args, int)
            elif cmd in ["factory_reset", "reset", "fr"]:
                self.cmd_factory_reset()
            else:
                print(f"{Fore.RED}Unknown command: {cmd}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error executing command: {e}{Style.RESET_ALL}")

    def cmd_webui(self):
        # Find espectre-monitor.html in the current directory or script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        html_file = os.path.join(script_dir, "espectre-monitor.html")
        
        if not os.path.exists(html_file):
            # Try current directory
            html_file = os.path.join(os.getcwd(), "espectre-monitor.html")
        
        if not os.path.exists(html_file):
            print(f"{Fore.RED}Error: espectre-monitor.html not found{Style.RESET_ALL}")
            return
        
        # Convert to file:// URL (cross-platform using pathlib)
        from pathlib import Path
        file_url = Path(html_file).absolute().as_uri()
        print(f"{Fore.BLUE}Opening web UI: {os.path.basename(html_file)}...{Style.RESET_ALL}")
        try:
            webbrowser.open(file_url)
        except Exception as e:
            print(f"{Fore.RED}Error opening browser: {e}{Style.RESET_ALL}")

    def cmd_segmentation_threshold(self, args):
        if not args:
            print(f"{Fore.RED}Usage: segmentation_threshold <value>{Style.RESET_ALL}")
            return
        self.send_command({"cmd": "segmentation_threshold", "value": float(args[0])})

    def cmd_segmentation_window_size(self, args):
        if not args:
            print(f"{Fore.RED}Usage: segmentation_window_size <packets>{Style.RESET_ALL}")
            return
        self.send_command({"cmd": "segmentation_window_size", "value": int(args[0])})

    def cmd_subcarrier_selection(self, args):
        if not args:
            print(f"{Fore.RED}Usage: subcarrier_selection <comma-separated indices>{Style.RESET_ALL}")
            return
        try:
            indices = [int(x.strip()) for x in args[0].split(',')]
            self.send_command({"cmd": "subcarrier_selection", "indices": indices})
        except ValueError:
            print(f"{Fore.RED}Invalid format. Use comma separated numbers.{Style.RESET_ALL}")

    def cmd_features_enable(self, args):
        self.cmd_toggle("features_enable", args)

    def cmd_toggle(self, cmd_name, args):
        if not args:
            print(f"{Fore.RED}Usage: {cmd_name} <on|off>{Style.RESET_ALL}")
            return
        val = args[0].lower()
        if val in ["on", "true", "1"]:
            self.send_command({"cmd": cmd_name, "enabled": True})
        elif val in ["off", "false", "0"]:
            self.send_command({"cmd": cmd_name, "enabled": False})
        else:
            print(f"{Fore.RED}Invalid value. Use on/off.{Style.RESET_ALL}")

    def cmd_value(self, cmd_name, args, type_func):
        if not args:
            print(f"{Fore.RED}Usage: {cmd_name} <value>{Style.RESET_ALL}")
            return
        try:
            self.send_command({"cmd": cmd_name, "value": type_func(args[0])})
        except ValueError:
            print(f"{Fore.RED}Invalid value type.{Style.RESET_ALL}")

    def cmd_factory_reset(self):
        print(f"{Fore.YELLOW}⚠️  WARNING: This will reset ALL settings to factory defaults!{Style.RESET_ALL}")
        print("Are you sure you want to continue? (yes/no): ", end="")
        sys.stdout.flush()
        confirm = input().lower()
        if confirm == "yes":
            print(f"{Fore.BLUE}Performing factory reset...{Style.RESET_ALL}")
            self.send_command({"cmd": "factory_reset"})
        else:
            print(f"{Fore.BLUE}Factory reset cancelled{Style.RESET_ALL}")

    def show_help(self):
        help_text = HTML("""
<ansibrightcyan><b>ESPectre CLI - Interactive Commands</b></ansibrightcyan>

<ansiyellow><b>Segmentation Commands:</b></ansiyellow>
  <ansigreen>segmentation_threshold|st</ansigreen> &lt;val&gt;            Set segmentation threshold (0.5-10.0)
  <ansigreen>segmentation_window_size|sws</ansigreen> &lt;n&gt;           Set moving variance window (10-200 packets)
  <ansigreen>subcarrier_selection|scs</ansigreen> &lt;indices&gt;         Set subcarrier selection (0-63, comma-separated)

<ansiyellow><b>Features Commands:</b></ansiyellow>
  <ansigreen>features_enable|fe</ansigreen> &lt;on|off&gt;                Enable/disable feature extraction during MOTION
  <ansigreen>butterworth_filter|butterworth|bw</ansigreen> &lt;on|off&gt; Enable/disable Butterworth filter (high freq)
  <ansigreen>wavelet_filter|wavelet|wv</ansigreen> &lt;on|off&gt;         Enable/disable Wavelet filter (low freq)
  <ansigreen>wavelet_level|wvl</ansigreen> &lt;1-3&gt;                    Set wavelet decomposition level (rec: 3)
  <ansigreen>wavelet_threshold|wvt</ansigreen> &lt;val&gt;                Set wavelet threshold (0.5-2.0, rec: 1.0)
  <ansigreen>hampel_filter|hampel</ansigreen> &lt;on|off&gt;              Enable/disable Hampel outlier filter
  <ansigreen>hampel_threshold|ht</ansigreen> &lt;val&gt;                  Set Hampel threshold (1.0-10.0)
  <ansigreen>savgol_filter|savgol|sg</ansigreen> &lt;on|off&gt;           Enable/disable Savitzky-Golay filter

<ansiyellow><b>System Commands:</b></ansiyellow>
  <ansigreen>info|i</ansigreen>                                     Show current configuration (static)
  <ansigreen>stats|s</ansigreen>                                    Show runtime statistics (dynamic)
  <ansigreen>smart_publishing|smart|sp</ansigreen> &lt;on|off&gt;         Enable/disable smart publishing
  <ansigreen>traffic_generator_rate|tgr|traffic</ansigreen> &lt;pps&gt;   Set traffic rate (0=off, 5-100, rec: 20)
  <ansigreen>factory_reset|reset|fr</ansigreen>                     Reset all settings to factory defaults

<ansiyellow><b>Utility Commands:</b></ansiyellow>
  <ansigreen>webui|web|ui</ansigreen>                               Open web interface in browser
  <ansigreen>clear|cls</ansigreen>                                  Clear screen
  <ansigreen>help|h</ansigreen>                                     Show this help message
  <ansigreen>about</ansigreen>                                      Show about information
  <ansigreen>exit|quit|q</ansigreen>                                Exit interactive mode
""")
        print()
        print_formatted_text(help_text)
        print()

    def show_about(self):
        about_text = HTML(r"""
<ansibrightmagenta>
   _____ ____  ____            _            
  | ____/ ___||  _ \ ___  ___| |_ _ __ ___ 
  |  _| \___ | |_) / _  \/ __| __| '__/ _ \
  | |___ ___) |  __/  __/ (__| |_| | |  __/
  |_____|____/|_|   \___|\___|\___|_|  \___|
</ansibrightmagenta>
                                            
  <ansibrightcyan><b>Wi-Fi Motion Detection System</b></ansibrightcyan>
  <ansicyan>Based on Channel State Information (CSI)</ansicyan>
  
  <ansibrightgreen>Created with ❤️  by <b>Francesco Pace</b></ansibrightgreen>
  
  <ansiblue>🔗 GitHub:</ansiblue>   <u>github.com/francescopace</u>
  <ansiblue>💼 LinkedIn:</ansiblue> <u>linkedin.com/in/francescopace</u>
  <ansiblue>📧 Email:</ansiblue>    <u>francesco.pace@gmail.com</u>
    
  <ansiwhite>This project explores the fascinating world of Wi-Fi sensing,
  using Channel State Information to detect motion and presence
  without cameras or additional sensors.</ansiwhite>
""")
        print()
        print_formatted_text(about_text)
        print()

if __name__ == "__main__":
    cli = EspectreCLI()
    cli.start()

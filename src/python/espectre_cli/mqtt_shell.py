"""Interactive MQTT shell for ESPectre."""

from __future__ import annotations

import os
import shlex
import time
from datetime import datetime
from typing import Any, Dict

from .common import (
    CallbackAPIVersion,
    CompactDumper,
    Fore,
    FormattedText,
    HTML,
    FileHistory,
    NestedCompleter,
    PAHO_V2,
    PromptSession,
    PromptStyle,
    Style,
    mqtt,
    print_formatted_text,
    yaml,
)
from .host import open_web_ui

ASCII_BANNER = r"""
  __  __ _                    _____ ____  ____            _
 |  \/  (_) ___ _ __ ___     | ____/ ___||  _ \ ___  ___| |_ _ __ ___
 | |\/| | |/ __| '__/ _ \ __ |  _| \___ \| |_) / _ \/ __| __| '__/ _ \
 | |  | | | (__| | | (_) |__|| |___ ___) |  __/  __/ (__| |_| | |  __/
 |_|  |_|_|\___|_|  \___/    |_____|____/|_|   \___|\___|\__|_|  \___|
"""


class EspectreMQTTShell:
    """Interactive MQTT CLI for runtime commands."""

    def __init__(self, args):
        self.broker = args.broker
        self.port = args.port
        self.device_id = args.device_id
        self.base_topic = f"{args.topic_prefix.rstrip('/')}/{self.device_id}"
        self.username = args.username
        self.password = args.password

        self.topic_cmd = f"{self.base_topic}/commands/request"
        self.topic_responses = f"{self.base_topic}/commands/+"
        if PAHO_V2:
            self.client = mqtt.Client(callback_api_version=CallbackAPIVersion.VERSION1)
        else:
            self.client = mqtt.Client()
        if self.username and self.password:
            self.client.username_pw_set(self.username, self.password)

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.running = True

        hist_file = os.path.join(os.path.expanduser("~"), ".espectre_cli_history")
        completer_dict = {
            "set_threshold": None,
            "info": None,
            "stats": None,
            "webui": None,
            "clear": None,
            "help": None,
            "about": None,
            "exit": None,
        }
        prompt_style = PromptStyle.from_dict({"prompt": "#00aa00 bold"})
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
            print(f"{Fore.BLUE}Listening on: {self.topic_responses}{Style.RESET_ALL}")
            client.subscribe(self.topic_responses)
        else:
            print(f"{Fore.RED}Failed to connect, return code {rc}{Style.RESET_ALL}")

    def on_message(self, client, userdata, msg):
        try:
            payload = msg.payload.decode()
            data = __import__("json").loads(payload)
            timestamp = datetime.now().strftime("%H:%M:%S")
            print()
            formatted_yaml = yaml.dump(data, Dumper=CompactDumper, sort_keys=False, default_flow_style=False, width=1000)
            print(f"{Fore.GREEN}[{timestamp}]{Style.RESET_ALL} Received:")
            print_formatted_text(
                FormattedText([("class:pygments", formatted_yaml)]),
                style=PromptStyle.from_dict({"pygments": "#ansiwhite"}),
            )
            print()
        except Exception as e:
            print(f"\nError parsing message: {e}")

    def send_command(self, cmd_data: Dict[str, Any]):
        try:
            payload = __import__("json").dumps(cmd_data)
            self.client.publish(self.topic_cmd, payload)
        except Exception as e:
            print(f"{Fore.RED}Error sending command: {e}{Style.RESET_ALL}")

    def start(self):
        print(f"{Fore.MAGENTA}{ASCII_BANNER}")
        print("Motion detection system based on Wi-Fi spectrum analysis - Interactive CLI")
        print(f"{Style.RESET_ALL}")

        try:
            self.client.connect(self.broker, self.port, 60)
            self.client.loop_start()
            time.sleep(0.5)
            print(f"\n{Fore.YELLOW}Type 'help' for commands, 'exit' to quit{Style.RESET_ALL}\n")
            print(f"{Fore.YELLOW}Tip: Use TAB for autocompletion, Ctrl+R to search history{Style.RESET_ALL}\n")
            while self.running:
                try:
                    user_input = self.session.prompt(HTML("<prompt>espectre></prompt> "))
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
        if cmd in ["help", "h"]:
            self.show_help()
            return
        if cmd == "about":
            self.show_about()
            return
        if cmd in ["clear", "cls"]:
            os.system("cls" if os.name == "nt" else "clear")
            return
        if cmd in ["webui", "web", "ui"]:
            open_web_ui()
            return

        try:
            if cmd in ["set_threshold", "st"]:
                self.cmd_set_threshold(args)
            elif cmd in ["info", "i"]:
                self.send_command({"command": "info"})
            elif cmd in ["stats", "s"]:
                self.send_command({"command": "stats"})
            else:
                print(f"{Fore.RED}Unknown command: {cmd}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error executing command: {e}{Style.RESET_ALL}")

    def cmd_set_threshold(self, args):
        if not args:
            print(f"{Fore.RED}Usage: set_threshold <threshold>{Style.RESET_ALL}")
            return
        self.send_command({"command": "set_threshold", "threshold": float(args[0])})

    def show_help(self):
        help_text = HTML(
            """
<ansibrightcyan><b>Micro-ESPectre CLI - Interactive Commands</b></ansibrightcyan>

<ansiyellow><b>Configuration Commands:</b></ansiyellow>
  <ansigreen>set_threshold|st</ansigreen> &lt;val&gt;               Set segmentation threshold (0.5-10.0)

<ansiyellow><b>System Commands:</b></ansiyellow>
  <ansigreen>info|i</ansigreen>                              Show current configuration
  <ansigreen>stats|s</ansigreen>                             Show runtime statistics (memory, loop time)

<ansiyellow><b>Utility Commands:</b></ansiyellow>
  <ansigreen>webui|web|ui</ansigreen>                        Open web interface in browser
  <ansigreen>clear|cls</ansigreen>                           Clear screen
  <ansigreen>help|h</ansigreen>                              Show this help message
  <ansigreen>about</ansigreen>                               Show about information
  <ansigreen>exit|quit|q</ansigreen>                         Exit interactive mode
"""
        )
        print()
        print_formatted_text(help_text)
        print()

    def show_about(self):
        print(f"\n{Fore.MAGENTA}{ASCII_BANNER}{Style.RESET_ALL}")
        about_text = HTML(
            """
  <ansibrightcyan><b>Wi-Fi Motion Detection System</b></ansibrightcyan>
  <ansicyan>Based on Channel State Information (CSI)</ansicyan>

  <ansibrightgreen>Created by <b>Francesco Pace</b></ansibrightgreen>

  <ansiblue>GitHub:</ansiblue>   <u>github.com/francescopace</u>
  <ansiblue>LinkedIn:</ansiblue> <u>linkedin.com/in/francescopace</u>
  <ansiblue>Email:</ansiblue>    <u>francesco.pace@espectre.dev</u>

  <ansiwhite>This project explores the fascinating world of Wi-Fi sensing,
  using Channel State Information to detect motion and presence.</ansiwhite>
"""
        )
        print_formatted_text(about_text)
        print()

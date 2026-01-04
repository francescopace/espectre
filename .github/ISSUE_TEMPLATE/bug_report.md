---
name: Bug report
about: Standard bug report template
title: "[BUG] "
labels: bug
assignees: ''
---

## Description

A clear and concise description of the bug.

## Steps to Reproduce

1. Go to '...'
2. Configure '...'
3. Run '...'
4. See error

## Expected Behavior

What you expected to happen.

## Actual Behavior

What actually happened.

## Environment

- **ESP32 Model**: [e.g., ESP32-S3, ESP32-C6]
- **ESPectre Version/Commit**: [e.g., v1.0.0, main branch, or commit hash like `a1b2c3d`]
- **Platform**: [ESPHome / Micro-ESPectre]
- **Home Assistant Version** (if applicable): [e.g., 2024.1.0]
- **ESPHome Version** (if applicable): [e.g., 2024.1.0]

## Configuration

Relevant YAML or Python configuration (remove sensitive data):

```yaml
# Your configuration here
```

## Logs

Relevant log output:

```
# Paste logs here
```

## Crash Debug Files (if applicable)

If your bug involves a crash (kernel panic, guru meditation, stack overflow, segmentation fault, etc.), please attach the following files to help with debugging:

- **`.elf` file**: The compiled firmware binary with debug symbols
- **`.map` file**: The memory map file showing symbol addresses

These files are located in your build directory:
- ESPHome: `.esphome/build/<device-name>/.pioenvs/<device-name>/firmware.elf` and `firmware.map`

## Additional Context

Add any other context about the problem here (screenshots, related issues, etc.).


#!/usr/bin/env python3
"""
Generate ESP32 test header file from binary CSI data for Unity tests.

This script reads baseline_data.bin and movement_data.bin and generates
a C header file with all CSI packets for testing purposes.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import sys
from pathlib import Path
from mvs_utils import load_binary_data, BASELINE_FILE, MOVEMENT_FILE

def format_csi_array(csi_data, indent=0):
    """
    Format CSI data as C array initializer
    
    Args:
        csi_data: numpy array of int8 values (128 elements)
        indent: indentation level
    
    Returns:
        str: Formatted C array initializer
    """
    indent_str = ' ' * indent
    values = [str(int(x)) for x in csi_data]
    
    # Format as single line with proper wrapping
    result = '{'
    result += ', '.join(values)
    result += '}'
    
    return result

def generate_header_file(baseline_packets, movement_packets, output_path):
    """
    Generate C header file with CSI data arrays
    
    Args:
        baseline_packets: List of baseline packets
        movement_packets: List of movement packets
        output_path: Path to output header file
    """
    lines = []
    
    # Header comment
    lines.append("/*")
    lines.append(" * Real CSI Data Captured from ESP32 Device")
    lines.append(" * Used for Testing and Performance Evaluation")
    lines.append(" *")
    lines.append(f" * Generated from {len(baseline_packets)} baseline and {len(movement_packets)} movement packets")
    lines.append(" */")
    lines.append("")
    lines.append("#ifndef REAL_CSI_DATA_ESP32_H")
    lines.append("#define REAL_CSI_DATA_ESP32_H")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("")
    
    # Baseline packets
    lines.append("// Baseline CSI packets (static environment)")
    for i, pkt in enumerate(baseline_packets):
        array_def = f"static const int8_t real_baseline_{i}[128] = "
        array_def += format_csi_array(pkt['csi_data'])
        array_def += ";"
        lines.append(array_def)
    
    lines.append("")
    
    # Array of pointers for easy iteration
    lines.append("// Array of pointers to baseline packets")
    lines.append("static const int8_t* real_baseline_packets[] = {")
    for i in range(len(baseline_packets)):
        lines.append(f"    real_baseline_{i},")
    lines.append("};")
    lines.append("")
    
    lines.append("// Array of pointers to movement packets")
    lines.append("static const int8_t* real_movement_packets[] = {")
    for i in range(len(movement_packets)):
        lines.append(f"    real_movement_{i},")
    lines.append("};")
    lines.append("")
    
    lines.append("#endif // REAL_CSI_DATA_ESP32_H")
    lines.append("")
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Generated {output_path}")
    print(f"  - {len(baseline_packets)} baseline packets")
    print(f"  - {len(movement_packets)} movement packets")

def main():
    """Main function"""
    # Check if binary files exist
    if not BASELINE_FILE.exists():
        print(f"Error: {BASELINE_FILE} not found", file=sys.stderr)
        print("Run: ./deploy.sh --collect-baseline", file=sys.stderr)
        sys.exit(1)
    
    if not MOVEMENT_FILE.exists():
        print(f"Error: {MOVEMENT_FILE} not found", file=sys.stderr)
        print("Run: ./deploy.sh --collect-movement", file=sys.stderr)
        sys.exit(1)
    
    # Load data
    print(f"Loading {BASELINE_FILE}...")
    baseline_packets = load_binary_data(BASELINE_FILE)
    
    print(f"Loading {MOVEMENT_FILE}...")
    movement_packets = load_binary_data(MOVEMENT_FILE)
    
    # Generate header file
    output_path = Path('../../test_app/main/real_csi_data_esp32.h')
    generate_header_file(baseline_packets, movement_packets, output_path)
    
    print("\nDone!")

if __name__ == '__main__':
    main()

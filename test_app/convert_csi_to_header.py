#!/usr/bin/env python3
"""
Script to convert CSI data from CSV format to C header file
"""

import sys
import os

def parse_csv_line(line):
    """Parse a CSV line and extract type and data"""
    parts = line.strip().split('|')
    if len(parts) != 3 or parts[0] != 'CSI':
        return None, None
    
    data_type = parts[1]  # BASELINE or MOVEMENT
    values = [int(x) for x in parts[2].split(',')]
    
    return data_type, values

def generate_header_file(csv_file, output_file):
    """Generate C header file from CSV data"""
    
    baseline_arrays = []
    movement_arrays = []
    
    # Read and parse CSV file
    with open(csv_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            data_type, values = parse_csv_line(line)
            if data_type is None:
                continue
            
            if data_type == 'BASELINE':
                baseline_arrays.append(values)
            elif data_type == 'MOVEMENT':
                movement_arrays.append(values)
    
    # Generate header file content
    with open(output_file, 'w') as f:
        # Write header
        f.write("/*\n")
        f.write(" * Real CSI Data Captured from Calibration\n")
        f.write(" */\n\n")
        f.write("#ifndef REAL_CSI_DATA_H\n")
        f.write("#define REAL_CSI_DATA_H\n\n")
        f.write("#include <stdint.h>\n\n")
        
        # Write baseline arrays
        if baseline_arrays:
            f.write("// Baseline CSI packets (static environment)\n")
            for i, values in enumerate(baseline_arrays):
                f.write(f"static const int8_t real_baseline_{i}[128] = {{")
                f.write(", ".join(str(v) for v in values))
                f.write("};\n")
            f.write("\n")
        
        # Write movement arrays
        if movement_arrays:
            f.write("// Movement CSI packets (with human movement)\n")
            for i, values in enumerate(movement_arrays):
                f.write(f"static const int8_t real_movement_{i}[128] = {{")
                f.write(", ".join(str(v) for v in values))
                f.write("};\n")
            f.write("\n")
        
        # Write footer
        f.write("#endif // REAL_CSI_DATA_H\n")
    
    print(f"✓ Converted {len(baseline_arrays)} baseline and {len(movement_arrays)} movement packets")
    print(f"✓ Output written to: {output_file}")

def main():
    # Default paths
    csv_file = "test_app/main/csi_data.csv"
    output_file = "test_app/main/real_csi_data.h"
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    # Check if input file exists
    if not os.path.exists(csv_file):
        print(f"Error: Input file '{csv_file}' not found")
        sys.exit(1)
    
    # Generate header file
    generate_header_file(csv_file, output_file)

if __name__ == "__main__":
    main()

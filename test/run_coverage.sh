#!/bin/bash
#
# ESPectre - Code Coverage Script
#
# Runs all tests with coverage instrumentation and prints report.
# Supports both Clang (macOS) and GCC (Linux) coverage.
#
# Usage:
#   ./run_coverage.sh           # Local run (prints report, cleans up)
#   ./run_coverage.sh --ci      # CI run (generates lcov for Codecov)
#
# Author: Francesco Pace <francesco.pace@gmail.com>
# License: GPLv3

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CI_MODE=false
if [ "$1" = "--ci" ]; then
    CI_MODE=true
fi

# Colors for output (disabled in CI)
if [ "$CI_MODE" = true ]; then
    RED=''
    GREEN=''
    YELLOW=''
    NC=''
else
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    NC='\033[0m'
fi

echo -e "${GREEN}🧪 ESPectre Code Coverage${NC}"
echo "================================"

# Detect compiler type
detect_compiler() {
    if command -v clang++ &> /dev/null; then
        # Check if clang is the default compiler
        if [[ "$OSTYPE" == "darwin"* ]]; then
            echo "clang"
            return
        fi
    fi
    echo "gcc"
}

COMPILER=$(detect_compiler)
echo -e "${YELLOW}Detected compiler: $COMPILER${NC}"

# Clean previous coverage data
echo -e "\n${YELLOW}Cleaning previous coverage data...${NC}"
rm -f *.profraw *.profdata coverage.lcov 2>/dev/null || true
rm -rf .pio/build/native_coverage 2>/dev/null || true
find . -name "*.gcda" -delete 2>/dev/null || true
find . -name "*.gcno" -delete 2>/dev/null || true

# Run tests with coverage
echo -e "\n${YELLOW}Running tests with coverage instrumentation...${NC}"
if [ "$COMPILER" = "clang" ]; then
    LLVM_PROFILE_FILE="coverage_%p.profraw" pio test -e native_coverage || true
else
    pio test -e native_coverage || true
fi

# Find the test executable
PROGRAM=".pio/build/native_coverage/program"
if [ ! -f "$PROGRAM" ]; then
    echo -e "${RED}Error: Test executable not found${NC}"
    exit 1
fi

echo -e "\n${GREEN}📊 Coverage Report${NC}"
echo "================================"

if [ "$COMPILER" = "clang" ]; then
    # ===== CLANG/LLVM COVERAGE =====
    
    # Check if any profraw files were generated
    PROFRAW_COUNT=$(ls *.profraw 2>/dev/null | wc -l | tr -d ' ')
    if [ "$PROFRAW_COUNT" -eq 0 ]; then
        echo -e "${RED}Error: No coverage data generated${NC}"
        exit 1
    fi
    
    # Detect llvm tools
    if [[ "$OSTYPE" == "darwin"* ]]; then
        LLVM_PROFDATA="xcrun llvm-profdata"
        LLVM_COV="xcrun llvm-cov"
    else
        LLVM_PROFDATA="llvm-profdata"
        LLVM_COV="llvm-cov"
    fi
    
    # Merge all profraw files
    $LLVM_PROFDATA merge -sparse *.profraw -o coverage.profdata 2>/dev/null
    
    # Print report
    printf "%-30s %10s %12s %10s\n" "File" "Lines" "Functions" "Branches"
    echo "---------------------------------------------------------------------"
    $LLVM_COV report "$PROGRAM" \
        -instr-profile=coverage.profdata \
        ../components/espectre/ 2>/dev/null | \
        grep -E '\.(cpp|h)' | grep -v "^-" | \
        while read -r line; do
            file=$(echo "$line" | awk '{print $1}')
            func_pct=$(echo "$line" | awk '{print $7}')
            lines_pct=$(echo "$line" | awk '{print $10}')
            branch_pct=$(echo "$line" | awk '{print $13}')
            if [ -n "$file" ] && [ -n "$lines_pct" ]; then
                printf "%-30s %10s %12s %10s\n" "$file" "$lines_pct" "$func_pct" "$branch_pct"
            fi
        done
    
    echo "---------------------------------------------------------------------"
    $LLVM_COV report "$PROGRAM" \
        -instr-profile=coverage.profdata \
        ../components/espectre/ 2>/dev/null | \
        grep "^TOTAL" | \
        while read -r line; do
            func_pct=$(echo "$line" | awk '{print $7}')
            lines_pct=$(echo "$line" | awk '{print $10}')
            branch_pct=$(echo "$line" | awk '{print $13}')
            printf "%-30s %10s %12s %10s\n" "TOTAL" "$lines_pct" "$func_pct" "$branch_pct"
        done
    
    # Generate lcov format for CI/Codecov
    if [ "$CI_MODE" = true ]; then
        echo -e "\n${YELLOW}Generating lcov report for Codecov...${NC}"
        
        # Get workspace root for path conversion
        WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
        
        $LLVM_COV export "$PROGRAM" \
            -instr-profile=coverage.profdata \
            -format=lcov \
            ../components/espectre/ > coverage.lcov.tmp 2>/dev/null
        
        # Convert absolute paths to relative paths for Codecov
        sed "s|SF:$WORKSPACE_ROOT/|SF:|g" coverage.lcov.tmp > coverage.lcov
        rm -f coverage.lcov.tmp
        
        echo -e "${GREEN}Generated: coverage.lcov${NC}"
        echo -e "${YELLOW}Coverage file preview:${NC}"
        head -20 coverage.lcov
    else
        rm -f *.profraw *.profdata
    fi
    
else
    # ===== GCC/GCOV COVERAGE =====
    
    # Check if gcov data was generated
    GCDA_COUNT=$(find .pio -name "*.gcda" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$GCDA_COUNT" -eq 0 ]; then
        echo -e "${RED}Error: No coverage data generated (no .gcda files)${NC}"
        exit 1
    fi
    
    # Use gcovr if available, otherwise lcov
    if command -v gcovr &> /dev/null; then
        echo "Using gcovr for coverage report..."
        
        # Get absolute paths
        WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
        ESPECTRE_DIR="$WORKSPACE_ROOT/components/espectre"
        
        # Print text report
        gcovr --root "$ESPECTRE_DIR" \
              --filter "$ESPECTRE_DIR/.*" \
              --exclude '.*test.*' \
              --print-summary \
              .pio/build/native_coverage/
        
        # Generate coverage reports for CI
        if [ "$CI_MODE" = true ]; then
            echo -e "\n${YELLOW}Generating coverage reports for Codecov...${NC}"
            
            # Generate Cobertura XML format
            gcovr --root "$WORKSPACE_ROOT" \
                  --filter "$ESPECTRE_DIR/.*" \
                  --exclude '.*test.*' \
                  --xml coverage.xml \
                  .pio/build/native_coverage/
            
            # Generate LCOV format
            gcovr --root "$WORKSPACE_ROOT" \
                  --filter "$ESPECTRE_DIR/.*" \
                  --exclude '.*test.*' \
                  --lcov coverage.lcov \
                  .pio/build/native_coverage/
            
            echo -e "${GREEN}Generated: coverage.xml and coverage.lcov${NC}"
        fi
    elif command -v lcov &> /dev/null; then
        echo "Using lcov for coverage report..."
        
        # Capture coverage data
        lcov --capture \
             --directory .pio/build/native_coverage/ \
             --output-file coverage.lcov \
             --include '*/components/espectre/*' \
             --quiet
        
        # Print summary
        lcov --summary coverage.lcov
        
        if [ "$CI_MODE" != true ]; then
            rm -f coverage.lcov
        fi
    else
        echo -e "${RED}Error: Neither gcovr nor lcov found. Install one of them.${NC}"
        echo "  Ubuntu/Debian: sudo apt-get install lcov"
        echo "  Or: pip install gcovr"
        exit 1
    fi
    
    # Cleanup gcov files in local mode
    if [ "$CI_MODE" != true ]; then
        find . -name "*.gcda" -delete 2>/dev/null || true
        find . -name "*.gcno" -delete 2>/dev/null || true
    fi
fi

echo -e "\n${GREEN}Done!${NC}"

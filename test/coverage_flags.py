"""
PlatformIO extra script for code coverage support on macOS/Linux.

Detects compiler (clang vs gcc) and adds appropriate coverage flags.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import subprocess
import sys

Import("env")

def get_compiler_type():
    """Detect if we're using clang or gcc."""
    try:
        # Get the C++ compiler from environment
        cxx = env.get("CXX", "g++")
        result = subprocess.run([cxx, "--version"], capture_output=True, text=True)
        output = result.stdout.lower()
        if "clang" in output:
            return "clang"
        return "gcc"
    except Exception:
        return "gcc"

compiler = get_compiler_type()
print(f"Coverage: Detected compiler: {compiler}")

if compiler == "clang":
    # Clang/LLVM: source-based coverage (more accurate)
    env.Append(
        CCFLAGS=["-fprofile-instr-generate", "-fcoverage-mapping"],
        LINKFLAGS=["-fprofile-instr-generate", "-fcoverage-mapping"]
    )
    print("Coverage: Using Clang source-based coverage (llvm-cov)")
else:
    # GCC: gcov-based coverage
    env.Append(
        CCFLAGS=["--coverage"],
        LINKFLAGS=["--coverage"]
    )
    print("Coverage: Using GCC gcov-based coverage")

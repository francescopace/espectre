#!/bin/bash
#
# ESPectre - Website test runner
#
# Runs Node's built-in test runner on every *.mjs file in this directory.
#
# Usage:
#   ./run_tests.sh
#   ./run_tests.sh test_site_structure.mjs

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ $# -gt 0 ]]; then
    exec node --test "$@"
fi

exec node --test *.mjs

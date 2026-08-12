# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - SDK Surface Invariants

Guards the invariants that keep the published C++ SDK surface coherent, which
a compiler cannot see because they hold across separate artifacts: the
`espectre_sdk.h` facade, the Doxygen input list, and the embedding guide.

The failure these protect against is a real one: a public type added to the
runtime layer, used in a facade-visible signature, but never made reachable
through the facade, leaving integrators with an incomplete type.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CPP_ROOT = REPO_ROOT / "src" / "cpp"
FACADE = CPP_ROOT / "espectre_sdk.h"
DOXYFILE = REPO_ROOT / "docs" / "Doxyfile"
EMBEDDING_GUIDE = REPO_ROOT / "docs" / "EMBEDDING.md"

FACADE_INCLUDE_PATTERN = re.compile(r'^\s*#include\s+"([^"]+)"', re.MULTILINE)
FORWARD_DECLARATION_PATTERN = re.compile(r"^\s*(?:struct|class)\s+([A-Za-z_][A-Za-z0-9_]*)\s*;", re.MULTILINE)
DEFINITION_PATTERN = re.compile(
    r"^\s*(?:struct|class)\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:final\s*)?(?::[^;{]*)?\{",
    re.MULTILINE,
)


def facade_includes() -> list[str]:
    """Header paths the facade pulls in, relative to the SDK root."""
    return FACADE_INCLUDE_PATTERN.findall(FACADE.read_text(encoding="utf-8"))


def facade_reachable_headers() -> list[Path]:
    """The facade itself plus every header it includes, as resolved paths."""
    headers = [FACADE]
    for include in facade_includes():
        resolved = CPP_ROOT / include
        assert resolved.is_file(), f"{FACADE.name} includes a header that does not exist: {include}"
        headers.append(resolved)
    return headers


def doxygen_input_headers() -> set[str]:
    """Header paths listed in the Doxyfile INPUT block, relative to the SDK root."""
    source = DOXYFILE.read_text(encoding="utf-8")
    input_block = re.search(r"^INPUT\s*=(.*?)(?=^\w+\s*=)", source, re.MULTILINE | re.DOTALL)
    assert input_block is not None, "Doxyfile has no INPUT block"
    return {
        match.replace("src/cpp/", "", 1)
        for match in re.findall(r"src/cpp/[\w/]+\.h", input_block.group(1))
    }


def test_every_facade_include_resolves_to_a_real_header() -> None:
    headers = facade_reachable_headers()
    assert len(headers) > 1, "the facade should include the supported surface"


@pytest.mark.parametrize("header", facade_reachable_headers(), ids=lambda path: path.name)
def test_forward_declarations_are_defined_somewhere_reachable(header: Path) -> None:
    """A type named in the supported surface must be constructible from the facade.

    Forward declarations are a legitimate way to decouple headers, but only when
    the definition still arrives through `espectre_sdk.h`. Otherwise an
    integrator sees the type in a signature and cannot instantiate it.
    """
    defined: set[str] = set()
    for reachable in facade_reachable_headers():
        defined.update(DEFINITION_PATTERN.findall(reachable.read_text(encoding="utf-8")))

    declared = set(FORWARD_DECLARATION_PATTERN.findall(header.read_text(encoding="utf-8")))
    unresolved = sorted(declared - defined)
    assert not unresolved, (
        f"{header.name} forward-declares {unresolved} but no header reachable from "
        f"{FACADE.name} defines them, so the type is incomplete for SDK consumers"
    )


def test_facade_headers_are_all_in_the_generated_reference() -> None:
    """Everything the facade exposes has to appear in the API reference.

    The relation is a subset, not equality: the Doxyfile deliberately documents
    a few headers the facade does not include, such as `ble_protocol.h`, whose
    types are referenced from the surface.
    """
    missing = sorted(set(facade_includes()) - doxygen_input_headers())
    assert not missing, (
        f"headers reachable from {FACADE.name} are absent from the Doxyfile INPUT list: {missing}"
    )


def test_supported_headers_appear_in_the_embedding_header_map() -> None:
    """The embedding guide's header map is the human index of the same surface."""
    guide = EMBEDDING_GUIDE.read_text(encoding="utf-8")
    # The map groups related headers on one row, so match anywhere in the guide
    # rather than trying to parse the table structure.
    documented = set(re.findall(r"`([\w/]+\.h)`", guide))
    missing = sorted(set(facade_includes()) - documented)
    assert not missing, (
        f"headers reachable from {FACADE.name} are missing from the {EMBEDDING_GUIDE.name} "
        f"header map: {missing}"
    )

# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - SDK Surface Invariants

Guards the invariants that keep the published C++ SDK surface coherent, which
a compiler cannot see because they hold across separate artifacts: the
`espectre_sdk.h` facade, the Doxygen input list, and the SDK guide.

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
CORE_FACADE = CPP_ROOT / "espectre_core_sdk.h"
DOXYFILE = CPP_ROOT / "Doxyfile"
SDK_GUIDE = REPO_ROOT / "docs" / "SDK.md"
RUNTIME_INTERNAL_HEADERS = {
    "core/base_detector.h",
    "core/csi_features.h",
    "core/csi_format.h",
    "core/filtered_turbulence_ring.h",
    "core/filters.h",
    "core/high_accuracy_detector.h",
    "core/l1_delta_tracker.h",
    "core/lightweight_detector.h",
    "core/ml_feature_trackers.h",
    "core/ml_weights.h",
    "core/threshold.h",
    "core/utils.h",
}
CORE_PUBLIC_HEADERS = {
    "espectre_core_sdk.h",
    "runtime/espectre_sdk_version.h",
    "core/base_detector.h",
    "core/csi_format.h",
    "core/csi_types.h",
    "core/detector_limits.h",
    "core/detector_types.h",
    "core/filter_config.h",
    "core/high_accuracy_detector.h",
    "core/lightweight_detector.h",
    "core/temporal_csi_sampler.h",
}
CORE_IMPLEMENTATION_HEADERS = {
    "core/csi_features.h",
    "core/filtered_turbulence_ring.h",
    "core/filters.h",
    "core/l1_delta_tracker.h",
    "core/ml_feature_trackers.h",
    "core/threshold.h",
    "core/utils.h",
}

FACADE_INCLUDE_PATTERN = re.compile(r'^\s*#include\s+"([^"]+)"', re.MULTILINE)
FORWARD_DECLARATION_PATTERN = re.compile(r"^\s*(?:struct|class)\s+([A-Za-z_][A-Za-z0-9_]*)\s*;", re.MULTILINE)
DEFINITION_PATTERN = re.compile(
    r"^\s*(?:struct|class)\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:final\s*)?(?::[^;{]*)?\{",
    re.MULTILINE,
)


def facade_includes() -> list[str]:
    """Header paths the facade pulls in, relative to the SDK root."""
    return FACADE_INCLUDE_PATTERN.findall(FACADE.read_text(encoding="utf-8"))


def include_closure(root_header: Path) -> list[Path]:
    """The complete local include closure rooted at one published facade."""
    available_by_name: dict[str, list[Path]] = {}
    maintained_headers = list(CPP_ROOT.glob("*.h"))
    for root in (CPP_ROOT / "core", CPP_ROOT / "runtime"):
        maintained_headers.extend(root.rglob("*.h"))
    for header in maintained_headers:
        available_by_name.setdefault(header.name, []).append(header)

    headers: list[Path] = []
    pending = [root_header]
    visited: set[Path] = set()
    while pending:
        header = pending.pop()
        if header in visited:
            continue
        visited.add(header)
        headers.append(header)

        for include in FACADE_INCLUDE_PATTERN.findall(header.read_text(encoding="utf-8")):
            candidates = [header.parent / include, CPP_ROOT / include]
            if "/" not in include:
                candidates.extend(available_by_name.get(include, []))
            resolved = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
            if resolved is not None and resolved.is_relative_to(CPP_ROOT):
                pending.append(resolved)

    return sorted(headers)


def facade_reachable_headers() -> list[Path]:
    """The complete local include closure rooted at the runtime facade."""
    return include_closure(FACADE)


def facade_reachable_header_names() -> set[str]:
    """Paths in the public include closure, relative to the SDK root."""
    return {header.relative_to(CPP_ROOT).as_posix() for header in facade_reachable_headers()}


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
    for include in facade_includes():
        assert (CPP_ROOT / include).is_file(), (
            f"{FACADE.name} includes a header that does not exist: {include}"
        )
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

    The relation is a subset, not equality: the Doxyfile may deliberately document
    headers that the facade does not include directly.
    """
    missing = sorted(facade_reachable_header_names() - doxygen_input_headers())
    assert not missing, (
        f"headers reachable from {FACADE.name} are absent from the Doxyfile INPUT list: {missing}"
    )


def test_runtime_facade_does_not_reach_detector_implementation_headers() -> None:
    """The recommended include must stay free of the advanced detector implementation."""
    leaked = sorted(facade_reachable_header_names() & RUNTIME_INTERNAL_HEADERS)
    assert not leaked, (
        f"{FACADE.name} leaks core-only implementation headers: {leaked}; "
        "keep them behind espectre_core_sdk.h"
    )


def test_core_facade_is_complete_documented_and_mapped() -> None:
    """The opt-in detector facade gets the same completeness checks as the runtime facade."""
    reachable = include_closure(CORE_FACADE)
    assert len(reachable) > 1, "the core facade should include the detector surface"

    defined: set[str] = set()
    declared: set[str] = set()
    for header in reachable:
        source = header.read_text(encoding="utf-8")
        defined.update(DEFINITION_PATTERN.findall(source))
        declared.update(FORWARD_DECLARATION_PATTERN.findall(source))
    assert not (declared - defined), (
        f"{CORE_FACADE.name} leaves public types incomplete: {sorted(declared - defined)}"
    )

    names = {header.relative_to(CPP_ROOT).as_posix() for header in reachable}
    assert CORE_PUBLIC_HEADERS <= names, (
        f"the declared core SDK header set is not reachable from {CORE_FACADE.name}: "
        f"{sorted(CORE_PUBLIC_HEADERS - names)}"
    )
    assert not (CORE_PUBLIC_HEADERS - doxygen_input_headers()), (
        f"supported core SDK headers are absent from the Doxyfile INPUT list: "
        f"{sorted(CORE_PUBLIC_HEADERS - doxygen_input_headers())}"
    )
    assert not (CORE_IMPLEMENTATION_HEADERS & doxygen_input_headers()), (
        "core implementation dependencies leaked into the generated API reference: "
        f"{sorted(CORE_IMPLEMENTATION_HEADERS & doxygen_input_headers())}"
    )

    guide = SDK_GUIDE.read_text(encoding="utf-8")
    documented = set(re.findall(r"`([\w/]+\.h)`", guide))
    assert not (CORE_PUBLIC_HEADERS - documented), (
        f"supported core SDK headers are missing from the {SDK_GUIDE.name} "
        f"header map: {sorted(CORE_PUBLIC_HEADERS - documented)}"
    )


def test_supported_headers_appear_in_the_sdk_header_map() -> None:
    """The SDK guide's header map is the human index of the same surface."""
    guide = SDK_GUIDE.read_text(encoding="utf-8")
    # The map groups related headers on one row, so match anywhere in the guide
    # rather than trying to parse the table structure.
    documented = set(re.findall(r"`([\w/]+\.h)`", guide))
    missing = sorted(facade_reachable_header_names() - documented)
    assert not missing, (
        f"headers reachable from {FACADE.name} are missing from the {SDK_GUIDE.name} "
        f"header map: {missing}"
    )

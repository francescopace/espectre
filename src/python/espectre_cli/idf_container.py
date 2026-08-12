# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Docker execution support for ESP-IDF firmware builds."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Callable


IDF_DOCKER_IMAGE = (
    "espressif/idf:release-v5.5@sha256:"
    "0c439ea923cd42700f9bbbe82542749d980712edb0ead0ea6db7eef35619b812"
)
DOCKER_PULL_POLICIES = ("ask", "missing", "never")


class DockerBackendError(RuntimeError):
    """Report that the Docker build backend cannot be prepared."""


def docker_executable() -> str | None:
    """Return the Docker CLI path when it is installed."""
    return shutil.which("docker")


def docker_daemon_is_running(docker: str) -> bool:
    """Return whether the Docker CLI can reach a running engine."""
    try:
        result = subprocess.run(
            [docker, "info", "--format", "{{.ServerVersion}}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


def docker_image_is_present(docker: str, image: str = IDF_DOCKER_IMAGE) -> bool:
    """Return whether the pinned ESP-IDF image is already cached locally."""
    try:
        result = subprocess.run(
            [docker, "image", "inspect", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


def _interactive_terminal() -> bool:
    return sys.stdin.isatty()


def ensure_docker_backend(
    pull_policy: str,
    *,
    image: str = IDF_DOCKER_IMAGE,
    input_fn: Callable[[str], str] = input,
) -> str:
    """Prepare Docker and the pinned image, prompting only for required user actions."""
    if pull_policy not in DOCKER_PULL_POLICIES:
        raise ValueError(f"Unsupported Docker pull policy: {pull_policy}")

    docker = docker_executable()
    if docker is None:
        raise DockerBackendError(
            "Docker is not installed. Install Docker, or install ESP-IDF 5.5.4 for local builds."
        )

    while not docker_daemon_is_running(docker):
        if not _interactive_terminal():
            raise DockerBackendError(
                "Docker is installed, but its engine is not running. Start Docker and rerun the command."
            )
        try:
            response = input_fn(
                "Docker is installed, but its engine is not running. Start Docker, then press Enter "
                "to retry, or enter q to quit: "
            )
        except EOFError as exc:
            raise DockerBackendError("Docker must be running to use the container build backend.") from exc
        if response.strip().lower() in {"q", "quit", "n", "no"}:
            raise DockerBackendError("Docker must be running to use the container build backend.")

    if docker_image_is_present(docker, image):
        return docker

    if pull_policy == "never":
        raise DockerBackendError(
            "The pinned ESP-IDF Docker image is not cached. Rerun with --pull missing to download it."
        )

    if pull_policy == "ask":
        if not _interactive_terminal():
            raise DockerBackendError(
                "The pinned ESP-IDF Docker image is not cached. Rerun with --pull missing to download it."
            )
        try:
            response = input_fn(
                "The ESP-IDF build image is not cached and may require a multi-gigabyte download. "
                "Download it now? [y/N] "
            )
        except EOFError as exc:
            raise DockerBackendError("The ESP-IDF Docker image download was not approved.") from exc
        if response.strip().lower() not in {"y", "yes"}:
            raise DockerBackendError("The ESP-IDF Docker image download was not approved.")

    try:
        subprocess.run([docker, "pull", image], check=True)
    except OSError as exc:
        raise DockerBackendError("The Docker CLI could not be started.") from exc
    except subprocess.CalledProcessError as exc:
        raise DockerBackendError(f"Docker could not download the ESP-IDF build image (exit {exc.returncode}).") from exc
    return docker


def build_docker_command(
    docker: str,
    *,
    frontend: str,
    app_path: Path,
    commands: list[list[str]],
    repo_root: Path,
    sdkconfig_defaults: str,
    image: str = IDF_DOCKER_IMAGE,
) -> list[str]:
    """Build the Docker CLI command that executes an ESP-IDF command sequence."""
    resolved_root = repo_root.resolve()
    resolved_app = app_path.resolve()
    try:
        app_relative = resolved_app.relative_to(resolved_root)
    except ValueError as exc:
        raise DockerBackendError(f"ESP-IDF app directory is outside the repository: {resolved_app}") from exc

    container_home_relative = Path(".github") / ".cache" / f"{frontend}-home"
    container_home = resolved_root / container_home_relative
    root_managed_components = container_home / "root_managed_components"
    container_home.mkdir(parents=True, exist_ok=True)
    root_managed_components.mkdir(parents=True, exist_ok=True)

    command = [docker, "run", "--rm"]
    if hasattr(os, "getuid") and hasattr(os, "getgid"):
        command.extend(["--user", f"{os.getuid()}:{os.getgid()}"])
    command.extend(
        [
            "-e",
            f"HOME=/work/{container_home_relative.as_posix()}",
            "-e",
            f"SDKCONFIG_DEFAULTS={sdkconfig_defaults}",
            "-v",
            f"{root_managed_components}:/opt/esp/root_managed_components",
            "-v",
            f"{resolved_root}:/work",
            "-w",
            f"/work/{app_relative.as_posix()}",
            image,
            "bash",
            "-lc",
            " && ".join(shlex.join(item) for item in commands),
        ]
    )
    return command


def run_idf_container(
    *,
    frontend: str,
    app_path: Path,
    commands: list[list[str]],
    repo_root: Path,
    sdkconfig_defaults: str,
    pull_policy: str,
    docker: str | None = None,
) -> None:
    """Run ESP-IDF build commands in the pinned Docker image."""
    docker = docker or ensure_docker_backend(pull_policy)
    command = build_docker_command(
        docker,
        frontend=frontend,
        app_path=app_path,
        commands=commands,
        repo_root=repo_root,
        sdkconfig_defaults=sdkconfig_defaults,
    )
    try:
        subprocess.run(command, check=True)
    except OSError as exc:
        raise DockerBackendError("The Docker CLI could not be started.") from exc
    except subprocess.CalledProcessError as exc:
        raise DockerBackendError(f"Docker ESP-IDF build failed with exit code {exc.returncode}.") from exc

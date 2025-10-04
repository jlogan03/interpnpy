#!/usr/bin/env python3
"""Build an optimised extension using cargo-pgo and a Python workload."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCH = ROOT / "scripts" / "profile_workload.py"
DEFAULT_WORKDIR = ROOT / "target" / "pgo"
ARTIFACT_NAMES = {"libinterpn.so", "libinterpn.dylib", "interpn.dll", "interpn.pyd"}


class CommandError(RuntimeError):
    """Raised when a subprocess exits with a non-zero status."""


def run(cmd: Sequence[str], *, env: dict[str, str] | None = None, cwd: Path | None = None) -> None:
    """Execute a command, echoing it before running."""
    print("+", " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, check=True, env=env, cwd=cwd)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - aids debugging
        raise CommandError(f"command failed with exit code {exc.returncode}: {' '.join(cmd)}") from exc


def ensure_bench_dependencies(bench_script: Path) -> None:
    """Import required Python modules before kicking off PGO."""
    required = {"numpy"}
    if bench_script.name == "bench_cpu.py":
        required.update({"scipy", "matplotlib"})

    missing: list[str] = []
    for module in sorted(required):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)

    if missing:
        deps = ", ".join(missing)
        hint = "uv pip install '.[bench]'"
        raise SystemExit(f"Missing Python dependencies ({deps}). Install them via `{hint}` before running PGO.")


def ensure_cargo_pgo() -> None:
    """Verify that cargo-pgo is available."""
    try:
        subprocess.run(["cargo", "pgo", "--version"], check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:  # pragma: no cover - trivial guard
        raise SystemExit("`cargo-pgo` is required. Install it via `cargo install cargo-pgo`.") from exc
    except subprocess.CalledProcessError as exc:  # pragma: no cover - environment issue
        raise SystemExit("Failed to execute `cargo pgo`. Ensure the tool is installed and functional.") from exc


def cargo_pgo(args: Sequence[str], *, env: dict[str, str] | None = None) -> None:
    """Run a cargo-pgo subcommand from the project root."""
    run(["cargo", "pgo", *args], env=env, cwd=ROOT)


def find_cdylib(target_dir: Path) -> Path:
    """Locate the built shared library under the given target directory."""
    for name in ARTIFACT_NAMES:
        candidates = sorted(target_dir.rglob(name))
        if candidates:
            return min(candidates, key=lambda path: len(path.parts))
    raise SystemExit(f"Could not locate compiled cdylib in {target_dir}.")


def extension_destination() -> Path:
    """Derive the expected extension filename inside the package."""
    suffix = next((s for s in EXTENSION_SUFFIXES if "abi3" in s), EXTENSION_SUFFIXES[0])
    return ROOT / "src" / "interpn" / f"interpn{suffix}"


def install_artifact(target_dir: Path) -> Path:
    """Copy the compiled library into the Python package."""
    artifact = find_cdylib(target_dir)
    destination = extension_destination()
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(artifact, destination)
    return destination


def run_benchmark(bench_script: Path, profiles_dir: Path) -> None:
    """Execute the Python workload while directing LLVM profiles to profiles_dir."""
    env = os.environ.copy()
    env["LLVM_PROFILE_FILE"] = str(profiles_dir / "interpn-%p-%m.profraw")
    env.setdefault("MPLBACKEND", "Agg")
    run([sys.executable, str(bench_script)], env=env, cwd=ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run profile-guided optimisation for interpn.")
    parser.add_argument("--bench", type=Path, default=DEFAULT_BENCH, help="Path to the benchmark workload to execute")
    parser.add_argument(
        "--workdir",
        type=Path,
        default=DEFAULT_WORKDIR,
        help="Directory used as CARGO_TARGET_DIR for instrumented and optimised builds",
    )
    parser.add_argument(
        "--skip-final-build",
        action="store_true",
        help="Only gather profiles and skip the final optimised build",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path)


def main() -> None:
    args = parse_args()
    bench_script = resolve_path(args.bench)
    if not bench_script.exists():
        raise SystemExit(f"Benchmark script not found: {bench_script}")

    ensure_cargo_pgo()
    ensure_bench_dependencies(bench_script)

    workdir = resolve_path(args.workdir)
    profiles_dir = workdir / "pgo-profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    cargo_env = os.environ.copy()
    cargo_env["CARGO_TARGET_DIR"] = str(workdir)

    print("Cleaning previous cargo-pgo artifacts...", flush=True)
    cargo_pgo(["clean"], env=cargo_env)

    print("Building instrumented extension with cargo-pgo...", flush=True)
    cargo_pgo(["instrument", "build", "--", "--features=python"], env=cargo_env)
    instrumented_path = install_artifact(workdir)
    print(f"Instrumented library copied to {instrumented_path}", flush=True)

    print("Running benchmark workload...", flush=True)
    run_benchmark(bench_script, profiles_dir)

    if args.skip_final_build:
        print(
            f"Skipping final build. Profiles available in {profiles_dir}; the instrumented library remains installed.",
            flush=True,
        )
        return

    print("Building optimised extension with cargo-pgo...", flush=True)
    cargo_pgo(["optimize", "build", "--", "--features=python"], env=cargo_env)
    optimised_path = install_artifact(workdir)

    print("PGO build complete. Optimised extension installed at", optimised_path, flush=True)


if __name__ == "__main__":
    try:
        main()
    except CommandError as error:
        raise SystemExit(str(error)) from error

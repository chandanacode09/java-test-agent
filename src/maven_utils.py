"""
Maven execution utilities for faster test runs.

Optimizations:
1. Use mvnd (Maven Daemon) if available - eliminates JVM startup overhead
2. Skip Spring formatting during iteration - saves 3-5s per compile
3. Parallel test forks - 4x speedup for multi-class runs
"""

import shutil
import subprocess
from pathlib import Path
from typing import Optional


def get_maven_cmd(project_path: Path) -> str:
    """
    Get the best Maven command for the project.

    Priority:
    1. mvnd (Maven Daemon) - fastest, keeps JVM warm
    2. ./mvnw (Maven Wrapper) - project-specific version
    3. mvn (System Maven) - fallback

    Returns:
        Command string to use for Maven
    """
    # Check for mvnd first (fastest option)
    if shutil.which("mvnd"):
        return "mvnd"

    # Check for project's Maven wrapper
    mvnw = project_path / "mvnw"
    if mvnw.exists():
        return "./mvnw"

    # Fallback to system Maven
    if shutil.which("mvn"):
        return "mvn"

    # Last resort
    return "./mvnw"


# Common flags to skip slow operations during iteration
FAST_FLAGS = [
    "-Dspring-javaformat.skip=true",  # Skip Spring format check (saves 3-5s)
    "-Dcheckstyle.skip=true",          # Skip checkstyle if present
    "-Dspotbugs.skip=true",            # Skip spotbugs if present
    "-Denforcer.skip=true",            # Skip enforcer plugin
]

# Parallel execution flags for test runs
PARALLEL_FLAGS = [
    "-DforkCount=4",      # Use 4 parallel JVM forks
    "-DreuseForks=true",  # Reuse forks between tests
]


def build_compile_cmd(
    project_path: Path,
    quiet: bool = True,
    skip_tests: bool = False,
    fast: bool = True
) -> list[str]:
    """
    Build Maven compile command with optimizations.

    Args:
        project_path: Path to the Maven project
        quiet: Suppress Maven output
        skip_tests: Skip test execution (just compile)
        fast: Apply speed optimizations (skip formatting, etc.)

    Returns:
        Command as list of strings for subprocess
    """
    cmd = [get_maven_cmd(project_path), "test-compile"]

    if quiet:
        cmd.append("-q")

    if skip_tests:
        cmd.append("-DskipTests")

    if fast:
        cmd.extend(FAST_FLAGS)

    return cmd


def build_test_cmd(
    project_path: Path,
    test_class: Optional[str] = None,
    quiet: bool = True,
    fast: bool = True,
    parallel: bool = False
) -> list[str]:
    """
    Build Maven test command with optimizations.

    Args:
        project_path: Path to the Maven project
        test_class: Specific test class to run (e.g., "VetControllerTest")
        quiet: Suppress Maven output
        fast: Apply speed optimizations
        parallel: Enable parallel test execution

    Returns:
        Command as list of strings for subprocess
    """
    cmd = [get_maven_cmd(project_path), "test"]

    if quiet:
        cmd.append("-q")

    if test_class:
        cmd.append(f"-Dtest={test_class}")

    if fast:
        cmd.extend(FAST_FLAGS)

    if parallel:
        cmd.extend(PARALLEL_FLAGS)

    return cmd


def run_compile(
    project_path: Path,
    timeout: int = 120,
    fast: bool = True
) -> tuple[bool, str, str]:
    """
    Run Maven compile with optimizations.

    Returns:
        Tuple of (success, stdout, stderr)
    """
    cmd = build_compile_cmd(project_path, fast=fast)

    try:
        result = subprocess.run(
            cmd,
            cwd=project_path,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return (
            result.returncode == 0,
            result.stdout,
            result.stderr
        )
    except subprocess.TimeoutExpired:
        return False, "", "Compilation timed out"
    except Exception as e:
        return False, "", str(e)


def run_tests(
    project_path: Path,
    test_class: Optional[str] = None,
    timeout: int = 180,
    fast: bool = True,
    parallel: bool = False
) -> tuple[bool, str, str]:
    """
    Run Maven tests with optimizations.

    Returns:
        Tuple of (success, stdout, stderr)
    """
    cmd = build_test_cmd(
        project_path,
        test_class=test_class,
        fast=fast,
        parallel=parallel
    )

    try:
        result = subprocess.run(
            cmd,
            cwd=project_path,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return (
            result.returncode == 0,
            result.stdout,
            result.stderr
        )
    except subprocess.TimeoutExpired:
        return False, "", "Tests timed out"
    except Exception as e:
        return False, "", str(e)


def check_mvnd_available() -> bool:
    """Check if mvnd (Maven Daemon) is available."""
    return shutil.which("mvnd") is not None


def install_mvnd_hint() -> str:
    """Return installation hint for mvnd."""
    return """
Maven Daemon (mvnd) not found. Install for 30-40% faster builds:

  macOS:   brew install mvndaemon/homebrew-mvnd/mvnd
  Linux:   sdk install mvnd
  Windows: choco install mvndaemon

Or download from: https://github.com/apache/maven-mvnd
"""

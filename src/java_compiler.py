"""
FastJavaCompiler - Direct Java compilation without Maven overhead.

Replaces `mvn test-compile` (20-40s) with direct `javac` calls (1-3s).

Key optimizations:
1. Classpath is built once and cached
2. Only compiles the specific test file, not entire project
3. No Maven plugin loading, POM parsing, etc.
"""

import hashlib
import subprocess
from pathlib import Path
from typing import Optional


class FastJavaCompiler:
    """
    Fast Java compilation using javac directly.

    Usage:
        compiler = FastJavaCompiler(project_path)
        success, error = compiler.compile_test(test_file)
    """

    def __init__(self, project_path: Path, verbose: bool = False):
        self.project_path = project_path
        self.verbose = verbose
        self._classpath: Optional[str] = None
        self._cache_file = project_path / ".java-test-agent-classpath"
        self._pom_hash_file = project_path / ".java-test-agent-pom-hash"

    def compile_test(self, test_file: Path) -> tuple[bool, str]:
        """
        Compile a single test file directly with javac.

        Args:
            test_file: Path to the .java test file

        Returns:
            (success, error_message)
        """
        classpath = self._get_classpath()
        if not classpath:
            return False, "Failed to build classpath"

        # Ensure output directory exists
        output_dir = self.project_path / "target" / "test-classes"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build javac command
        cmd = [
            "javac",
            "-cp", classpath,
            "-d", str(output_dir),
            "-encoding", "UTF-8",
            "-source", "17",
            "-target", "17",
            "-Xlint:none",  # Suppress warnings for speed
            str(test_file)
        ]

        if self.verbose:
            print(f"  javac: {test_file.name}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30  # Should be very fast
            )

            if result.returncode == 0:
                return True, ""
            else:
                return False, result.stderr

        except subprocess.TimeoutExpired:
            return False, "Compilation timed out"
        except Exception as e:
            return False, str(e)

    def _get_classpath(self) -> Optional[str]:
        """Get classpath, using cache if valid."""
        if self._classpath:
            return self._classpath

        # Check if cache is valid
        if self._is_cache_valid():
            self._classpath = self._cache_file.read_text().strip()
            if self.verbose:
                print("  Using cached classpath")
            return self._classpath

        # Build fresh classpath
        if self.verbose:
            print("  Building classpath (one-time)...")

        self._classpath = self._build_classpath()

        if self._classpath:
            # Cache it
            self._cache_file.write_text(self._classpath)
            self._save_pom_hash()

        return self._classpath

    def _is_cache_valid(self) -> bool:
        """Check if cached classpath is still valid."""
        if not self._cache_file.exists():
            return False

        if not self._pom_hash_file.exists():
            return False

        # Check if pom.xml has changed
        pom_file = self.project_path / "pom.xml"
        if not pom_file.exists():
            return False

        current_hash = self._hash_file(pom_file)
        cached_hash = self._pom_hash_file.read_text().strip()

        return current_hash == cached_hash

    def _save_pom_hash(self):
        """Save hash of pom.xml for cache invalidation."""
        pom_file = self.project_path / "pom.xml"
        if pom_file.exists():
            hash_value = self._hash_file(pom_file)
            self._pom_hash_file.write_text(hash_value)

    def _hash_file(self, path: Path) -> str:
        """Compute MD5 hash of a file."""
        return hashlib.md5(path.read_bytes()).hexdigest()

    def _build_classpath(self) -> Optional[str]:
        """Build classpath from Maven dependencies."""
        try:
            # Use Maven to get dependency classpath
            result = subprocess.run(
                [
                    "./mvnw",
                    "dependency:build-classpath",
                    "-DincludeScope=test",
                    "-Dmdep.outputFile=/dev/stdout",
                    "-q"
                ],
                cwd=self.project_path,
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode != 0:
                if self.verbose:
                    print(f"  Failed to build classpath: {result.stderr}")
                return None

            # Maven outputs classpath to stdout
            deps_classpath = result.stdout.strip()

            # Add project's own classes
            classpath_parts = [
                deps_classpath,
                str(self.project_path / "target" / "classes"),
                str(self.project_path / "target" / "test-classes"),
            ]

            return ":".join(filter(None, classpath_parts))

        except subprocess.TimeoutExpired:
            if self.verbose:
                print("  Classpath build timed out")
            return None
        except Exception as e:
            if self.verbose:
                print(f"  Classpath build error: {e}")
            return None

    def invalidate_cache(self):
        """Force cache invalidation."""
        if self._cache_file.exists():
            self._cache_file.unlink()
        if self._pom_hash_file.exists():
            self._pom_hash_file.unlink()
        self._classpath = None

    def ensure_project_compiled(self) -> bool:
        """
        Ensure main project classes are compiled.

        Run this once before compiling tests.
        """
        target_classes = self.project_path / "target" / "classes"

        if target_classes.exists() and any(target_classes.rglob("*.class")):
            return True

        if self.verbose:
            print("  Compiling main project (one-time)...")

        try:
            result = subprocess.run(
                ["./mvnw", "compile", "-q", "-DskipTests"],
                cwd=self.project_path,
                capture_output=True,
                text=True,
                timeout=120
            )
            return result.returncode == 0

        except Exception:
            return False


class FastTestRunner:
    """
    Fast test execution using JUnit ConsoleLauncher.

    Skips Maven overhead for test execution.
    """

    JUNIT_LAUNCHER_VERSION = "1.10.0"

    def __init__(self, project_path: Path, classpath: str, verbose: bool = False):
        self.project_path = project_path
        self.classpath = classpath
        self.verbose = verbose
        self._launcher_jar: Optional[Path] = None

    def run_test(self, test_class: str) -> tuple[bool, str, int, int]:
        """
        Run a test class using JUnit ConsoleLauncher.

        Args:
            test_class: Fully qualified class name (e.g., "com.example.FooTest")

        Returns:
            (success, output, tests_passed, tests_total)
        """
        launcher = self._get_launcher_jar()

        if not launcher:
            # Fall back to Maven if launcher not available
            return self._run_with_maven(test_class)

        cmd = [
            "java",
            "-jar", str(launcher),
            "--class-path", self.classpath,
            "--select-class", test_class,
            "--fail-if-no-tests",
        ]

        if self.verbose:
            print(f"  JUnit: {test_class}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )

            output = result.stdout + result.stderr

            # Parse JUnit output
            tests_passed, tests_total = self._parse_junit_output(output)

            success = result.returncode == 0 and tests_total > 0

            return success, output, tests_passed, tests_total

        except subprocess.TimeoutExpired:
            return False, "Test execution timed out", 0, 0
        except Exception as e:
            return False, str(e), 0, 0

    def _get_launcher_jar(self) -> Optional[Path]:
        """Get JUnit ConsoleLauncher JAR, downloading if needed."""
        if self._launcher_jar and self._launcher_jar.exists():
            return self._launcher_jar

        # Check common locations
        cache_dir = Path.home() / ".cache" / "java-test-agent"
        launcher_name = f"junit-platform-console-standalone-{self.JUNIT_LAUNCHER_VERSION}.jar"
        launcher_path = cache_dir / launcher_name

        if launcher_path.exists():
            self._launcher_jar = launcher_path
            return launcher_path

        # Try to download
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)

            url = (
                f"https://repo1.maven.org/maven2/org/junit/platform/"
                f"junit-platform-console-standalone/{self.JUNIT_LAUNCHER_VERSION}/"
                f"{launcher_name}"
            )

            if self.verbose:
                print(f"  Downloading JUnit launcher...")

            import urllib.request
            urllib.request.urlretrieve(url, launcher_path)

            self._launcher_jar = launcher_path
            return launcher_path

        except Exception as e:
            if self.verbose:
                print(f"  Failed to download JUnit launcher: {e}")
            return None

    def _parse_junit_output(self, output: str) -> tuple[int, int]:
        """Parse JUnit ConsoleLauncher output for test counts."""
        import re

        # Look for summary line like: "4 tests successful"
        match = re.search(r"(\d+)\s+tests?\s+successful", output)
        if match:
            passed = int(match.group(1))
        else:
            passed = 0

        # Look for total: "4 tests found"
        match = re.search(r"(\d+)\s+tests?\s+found", output)
        if match:
            total = int(match.group(1))
        else:
            total = passed

        return passed, total

    def _run_with_maven(self, test_class: str) -> tuple[bool, str, int, int]:
        """Fallback to Maven for test execution."""
        from .maven_utils import run_tests
        import re

        success, stdout, stderr = run_tests(
            self.project_path,
            test_class=test_class,
            timeout=180
        )

        output = stdout + stderr

        # Parse Maven/Surefire output
        matches = re.findall(
            r"Tests run:\s*(\d+),\s*Failures:\s*(\d+),\s*Errors:\s*(\d+)",
            output
        )

        if matches:
            match = matches[-1]
            total = int(match[0])
            failures = int(match[1])
            errors = int(match[2])
            passed = total - failures - errors
            return failures == 0 and errors == 0, output, passed, total

        return success, output, 0, 0

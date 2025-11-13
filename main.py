#!/usr/bin/env python3
import os
import re
import subprocess
import sys
from pathlib import Path

TEST_FILE_PATTERN = re.compile(r"^test(\d+)\.py$")


def find_test_files():
    """Find all test*.py files where * is a positive integer."""
    test_files = []
    current_dir = Path(__file__).parent

    for file in current_dir.glob("test*.py"):
        # Extract the number from testX.py
        match = TEST_FILE_PATTERN.match(file.name)
        if not match:
            continue
        test_files.append((int(match.group(1)), file))

    # Sort by test number
    test_files.sort(key=lambda x: x[0])
    return [file for _, file in test_files]


def parse_expected_output(test_file):
    """Parse the expected output from a test file."""
    with open(test_file, "r") as f:
        lines = f.readlines()

    expected_output = []
    in_expected_section = False

    for line in lines:
        if line.strip() == "# Expected output:":
            in_expected_section = True
            continue

        if not in_expected_section or len(line) < 2:
            continue
        # Remove first 2 characters (should be "# ")
        parsed_line = line[2:].strip()
        # Only add non-empty lines
        if parsed_line:
            expected_output.append(parsed_line)

    return expected_output


def parse_actual_output(stdout):
    """Parse actual stdout by stripping lines and removing empty ones."""
    return [line.strip() for line in stdout.split("\n") if line.strip()]


def run_test(test_file):
    """Run a test file and return the result."""
    try:
        # important: use -u to ensure that stdout is flushed immediately
        # for every print statement in python, ensuring that the output
        # order w.r.t. the C/C++ extension is correct.
        result = subprocess.run(
            f"python -u {test_file}",
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )

        return result
    except subprocess.TimeoutExpired:
        return None


def main():
    """Main function to run all tests."""
    test_files = find_test_files()

    if not test_files:
        print("No test files found.")
        return 1

    print(f"Found {len(test_files)} test files.")
    print("=" * 80)

    passed = 0
    failed = 0

    for test_file in test_files:
        test_name = test_file.name
        print(f"\nRunning {test_name}...")

        # Parse expected output
        try:
            expected_output = parse_expected_output(test_file)
        except Exception as e:
            print(f"  ❌ FAILED: Could not parse expected output: {e}")
            failed += 1
            continue

        # Run the test
        result = run_test(test_file)

        if result is None:
            print(f"  ❌ FAILED: Test timed out")
            failed += 1
            continue

        # Check return code
        if result.returncode != 0 or result.stderr.strip():
            print(f"  ❌ FAILED: Non-zero return code ({result.returncode})")
            print(f"  stderr: {result.stderr}")
            failed += 1
            continue

        # Parse actual output
        actual_output = parse_actual_output(result.stdout)

        # Compare outputs
        if actual_output == expected_output:
            print(f"  ✅ PASSED")
            passed += 1
        else:
            print(f"  ❌ FAILED: Output mismatch")
            print(f"  Expected ({len(expected_output)} lines):")
            for line in expected_output:
                print(f"    {line}")
            print(f"  Actual ({len(actual_output)} lines):")
            for line in actual_output:
                print(f"    {line}")
            failed += 1

    # Summary
    print("\n" + "=" * 80)
    print(f"Summary: {passed} passed, {failed} failed out of {len(test_files)} tests")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

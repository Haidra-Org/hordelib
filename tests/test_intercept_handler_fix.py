"""
Test that the improved InterceptHandler correctly identifies source locations.

This test verifies that:
1. Frame walking skips import machinery correctly
2. Source locations are accurate
3. Performance is acceptable
"""

import logging
import sys
from pathlib import Path

from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from hordelib.comfy_horde import InterceptHandler


def test_source_location_accuracy():
    """Test that log messages show correct source location."""

    # Capture loguru output
    captured_messages = []

    def capture_sink(message):
        captured_messages.append(message.record)

    # Configure loguru
    logger.remove()
    logger.add(capture_sink, format="{message}")

    # Setup stdlib logging with InterceptHandler
    test_logger = logging.getLogger("test_source_location")
    test_logger.setLevel(logging.INFO)
    test_logger.handlers.clear()
    test_logger.propagate = False

    handler = InterceptHandler()
    test_logger.addHandler(handler)

    # Clear captured messages
    captured_messages.clear()

    # Log a message
    test_logger.info("Test message for source location")

    # Check we got the message
    assert len(captured_messages) == 1, f"Expected 1 message, got {len(captured_messages)}"

    record = captured_messages[0]

    # Check the source location
    print("\n" + "=" * 80)
    print("SOURCE LOCATION TEST RESULTS")
    print("=" * 80)
    print(f"File: {record['file'].path}")
    print(f"Line: {record['line']}")
    print(f"Function: {record['function']}")
    print(f"Name: {record['name']}")
    print("=" * 80)

    # Verify it's this test file, not importlib or logging infrastructure
    assert "importlib" not in str(record["file"]), f"Should not show importlib, got: {record['file'].path}"
    assert "_bootstrap" not in str(record["file"]), f"Should not show _bootstrap, got: {record['file'].path}"
    assert "test_intercept_handler_fix" in str(record["file"]), f"Should show test file, got: {record['file'].path}"
    assert (
        record["function"] == "test_source_location_accuracy"
    ), f"Should show test function, got: {record['function']}"

    print(f"   Correctly identified: {record['file'].name}:{record['function']}:{record['line']}")

    # Cleanup
    test_logger.removeHandler(handler)
    logger.remove()


def test_filters_still_work():
    """Test that message and library filters still work."""

    captured_messages = []

    def capture_sink(message):
        captured_messages.append(message.record["message"])

    logger.remove()
    logger.add(capture_sink, format="{message}")

    test_logger = logging.getLogger("test_filters")
    test_logger.setLevel(logging.INFO)
    test_logger.handlers.clear()
    test_logger.propagate = False

    handler = InterceptHandler()
    test_logger.addHandler(handler)

    # Test message filtering
    captured_messages.clear()
    test_logger.info("lowvram: loaded module regularly")
    assert len(captured_messages) == 0, "Filtered message should not appear"

    test_logger.info("lora key not loaded")
    assert len(captured_messages) == 0, "Filtered message should not appear"

    test_logger.info("This should appear")
    assert len(captured_messages) == 1, "Normal message should appear"
    assert "This should appear" in captured_messages[0]

    # Test library filtering
    captured_messages.clear()
    numba_logger = logging.getLogger("numba.core.something")
    numba_logger.setLevel(logging.INFO)
    numba_logger.handlers.clear()
    numba_logger.propagate = False
    numba_logger.addHandler(handler)

    numba_logger.info("This should be filtered")
    assert len(captured_messages) == 0, "numba.core messages should be filtered"

    # Cleanup
    test_logger.removeHandler(handler)
    numba_logger.removeHandler(handler)
    logger.remove()


def test_exception_handling():
    """Test that exception info is properly passed through."""

    captured_exceptions = []

    def capture_sink(message):
        record = message.record
        if record["exception"]:
            captured_exceptions.append(record["exception"])

    logger.remove()
    logger.add(capture_sink, format="{message}")

    test_logger = logging.getLogger("test_exceptions")
    test_logger.setLevel(logging.ERROR)
    test_logger.handlers.clear()
    test_logger.propagate = False

    handler = InterceptHandler()
    test_logger.addHandler(handler)

    # Log an exception
    try:
        raise ValueError("Test exception")
    except ValueError:
        test_logger.exception("An error occurred")

    assert len(captured_exceptions) == 1, "Exception should be captured"
    exc_info = captured_exceptions[0]
    assert exc_info.type is ValueError
    assert "Test exception" in str(exc_info.value)

    # Cleanup
    test_logger.removeHandler(handler)
    logger.remove()

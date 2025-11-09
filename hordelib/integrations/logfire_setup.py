"""Logfire instrumentation setup and configuration for hordelib.

This module centralizes all logfire configuration, including:
- OTLP endpoint configuration for local Jaeger/Prometheus
- Sensitive data scrubbing
- Auto-instrumentation for Pydantic, HTTP libraries, and system metrics
- Loguru bridge for existing logging infrastructure
"""

import os

import logfire
from logfire import ScrubbingOptions, ScrubMatch


def scrub_sensitive_data(match: ScrubMatch) -> str | None:
    """Custom scrubbing callback for fine-grained control of sensitive data.

    This callback is invoked for each potential match of sensitive patterns.
    Return a string to replace the value, or None to use default scrubbing behavior.

    Default patterns automatically handle: password, token, key, secret, authorization
    """
    # Scrub CivitAI API tokens
    if match.path == ("attributes", "civit_api_token"):
        return "[REDACTED]"

    # Truncate long prompts to prevent span size issues (>500 chars)
    if match.path == ("attributes", "prompt") and isinstance(match.value, str) and len(match.value) > 500:
        return match.value[:500] + "...[TRUNCATED]"

    # Return None to use default scrubbing behavior for other matches
    return None


def initialize_logfire() -> None:
    """Initialize logfire with local OTLP configuration and auto-instrumentation.

    This function should be called as early as possible in the application lifecycle,
    ideally in hordelib/__init__.py, to ensure all subsequent operations are traced.

    Configuration:
        - Sends traces to local Jaeger via OTLP (http://localhost:4318/v1/traces)
        - Sends metrics to local Prometheus via OTLP (http://localhost:9090/api/v1/otlp/v1/metrics)
        - Scrubs sensitive data (tokens, passwords, keys)
        - Bridges loguru logs to logfire
        - Enables auto-instrumentation for Pydantic, HTTP libraries, and system metrics
        - Uses 100% sampling (no sampling config) for maximum initial visibility

    Environment Variables:
        OTEL_EXPORTER_OTLP_TRACES_ENDPOINT: Jaeger endpoint (default: http://localhost:4318/v1/traces)
        OTEL_EXPORTER_OTLP_METRICS_ENDPOINT: Prometheus endpoint (default: http://localhost:9090/api/v1/otlp/v1/metrics)
        OTEL_EXPORTER_OTLP_PROTOCOL: Protocol (default: http/protobuf)
    """
    # Read OTLP endpoints from environment (with defaults for local development)
    traces_endpoint = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://localhost:4318/v1/traces")
    metrics_endpoint = os.getenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", "http://localhost:9090/api/v1/otlp/v1/metrics")

    # Configure scrubbing options
    scrubbing_options = ScrubbingOptions(
        callback=scrub_sensitive_data,
        # Extra patterns are added to default patterns (password, token, key, secret, authorization)
        extra_patterns=["civit.*token", "api[_-]?key"],
    )

    # Initialize logfire with local OTLP configuration
    logfire.configure(
        send_to_logfire=False,  # Local deployment - do not send to Logfire cloud
        service_name="hordelib",
        scrubbing=scrubbing_options,
        # Future optimization: Enable sampling after collecting baseline performance data
        #
        # Option 1: Simple head sampling (10% of all traces)
        # from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
        # sampling=logfire.SamplingOptions(
        #     head=ParentBased(TraceIdRatioBased(0.1))
        # )
        #
        # Option 2: Level-based tail sampling (keep slow traces and errors)
        # sampling=logfire.SamplingOptions.level_or_duration(
        #     duration_threshold=1.0,  # Always keep traces >1 second
        # )
        #
        # Current approach: 100% sampling for maximum initial visibility
    )

    # Auto-instrumentation: Pydantic validation
    # Captures validation events for all Pydantic models (ImagePayload, ModelGenerationInputStable, etc.)
    # Uses record='all' by default to capture all validation events
    logfire.instrument_pydantic()

    # Auto-instrumentation: HTTP libraries
    # Captures CivitAI downloads, model metadata fetches, etc.
    try:
        logfire.instrument_httpx(
            capture_headers=True,  # Capture request/response headers
            # Note: capture_request_body and capture_response_body can be enabled if needed
            # but may significantly increase span size for large payloads (e.g., model downloads)
        )
    except Exception as e:
        # httpx may not be installed in all environments
        logfire.warn("Failed to instrument httpx", error=str(e))

    try:
        logfire.instrument_requests()
    except Exception as e:
        logfire.warn("Failed to instrument requests", error=str(e))

    try:
        logfire.instrument_aiohttp_client(
            capture_headers=True,
            # Avoid capturing large response bodies (model files, images)
            # capture_response_body=False,
        )
    except Exception as e:
        logfire.warn("Failed to instrument aiohttp", error=str(e))

    # Auto-instrumentation: System metrics
    # Collects CPU, memory, disk I/O, network metrics
    try:
        logfire.instrument_system_metrics()
    except Exception as e:
        logfire.warn("Failed to instrument system metrics", error=str(e))

    # Bridge loguru to logfire
    # This connects existing logger.info(), logger.debug() calls to logfire
    # WARNING: Logfire does not scrub loguru-formatted messages, so avoid logging sensitive data via loguru
    try:
        from loguru import logger

        logger.configure(handlers=[{"sink": logfire.loguru_handler()}])
    except Exception as e:
        # loguru may not be installed or configuration may fail
        logfire.warn("Failed to configure loguru bridge", error=str(e))

    logfire.info(
        "Logfire instrumentation initialized",
        traces_endpoint=traces_endpoint,
        metrics_endpoint=metrics_endpoint,
        scrubbing_enabled=True,
        sampling_rate=1.0,
    )

    # Apply deep ComfyUI internals instrumentation
    _initialize_comfy_internals()


def _initialize_comfy_internals() -> None:
    """Initialize deep ComfyUI internals instrumentation.

    This applies monkey-patches to ComfyUI internal functions for comprehensive
    visibility into node execution, model loading, sampling, and memory management.
    """
    try:
        from hordelib.integrations.logfire_comfy_internals import instrument_comfy_internals

        instrument_comfy_internals()
        logfire.info("Deep ComfyUI instrumentation applied")
    except Exception as e:
        logfire.warn("Failed to apply deep ComfyUI instrumentation", error=str(e))

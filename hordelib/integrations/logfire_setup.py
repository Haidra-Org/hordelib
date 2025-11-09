"""Logfire instrumentation setup and configuration for hordelib.

This module centralizes all logfire configuration, including:
- OTLP endpoint configuration for local Jaeger/Prometheus
- Sensitive data scrubbing
- Auto-instrumentation for Pydantic, HTTP libraries, and system metrics
- Loguru bridge for unified logging infrastructure

UNIFIED LOGGING APPROACH:
- Use loguru's `logger` for ALL logging events (info, debug, error, warning)
- Use logfire for spans (@logfire.instrument, with logfire.span) and metrics
- Loguru logs automatically flow to logfire via loguru_handler()
- See LOGURU_LOGFIRE_BEST_PRACTICES.md for detailed guidelines
"""

import os

import logfire
from logfire import ScrubbingOptions, ScrubMatch

try:
    import httpx as httpx  # type: ignore # Ensure httpx is imported for instrumentation
except ImportError:
    pass

import aiohttp as aiohttp  # Ensure aiohttp is imported for instrumentation
import requests as requests  # Ensure requests is imported for instrumentation


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
    """Initialize logfire with local OTLP configuration or Logfire cloud platform.

    This function should be called as early as possible in the application lifecycle,
    ideally in hordelib/__init__.py, to ensure all subsequent operations are traced.

    Configuration:
        Mode 1 - Logfire Cloud Platform (when LOGFIRE_TOKEN is set):
            - Sends data to Logfire cloud (https://logfire-api.pydantic.dev)
            - Uses token-based authentication
            - Recommended for production deployments

        Mode 2 - Local OTLP Endpoints (default when no LOGFIRE_TOKEN):
            - Sends traces to local Jaeger via OTLP (http://localhost:4318/v1/traces)
            - Sends metrics to local Prometheus via OTLP (http://localhost:9090/api/v1/otlp/v1/metrics)
            - Recommended for local development and testing

        Both modes:
            - Scrub sensitive data (tokens, passwords, keys)
            - Bridge loguru logs to logfire
            - Enable auto-instrumentation for Pydantic, HTTP libraries, and system metrics
            - Use 100% sampling (no sampling config) for maximum initial visibility

    Environment Variables:
        LOGFIRE_TOKEN: Logfire write token for cloud platform (optional)
            If set, enables sending to Logfire cloud instead of local OTLP endpoints.
            Get your token from: https://logfire.pydantic.dev/

        LOGFIRE_BASE_URL: Custom Logfire base URL (optional, for self-hosted instances)
            Default: https://logfire-api.pydantic.dev

        OTEL_EXPORTER_OTLP_TRACES_ENDPOINT: Jaeger endpoint (default: http://localhost:4318/v1/traces)
            Only used when LOGFIRE_TOKEN is not set.

        OTEL_EXPORTER_OTLP_METRICS_ENDPOINT: Prometheus endpoint (default: http://localhost:9090/api/v1/otlp/v1/metrics)
            Only used when LOGFIRE_TOKEN is not set.

        OTEL_EXPORTER_OTLP_PROTOCOL: Protocol (default: http/protobuf)
    """
    # Check if we should use Logfire cloud platform or local OTLP endpoints
    logfire_token = os.getenv("LOGFIRE_TOKEN")
    use_logfire_cloud = logfire_token is not None

    # Read OTLP endpoints from environment (with defaults for local development)
    # These are only used when not using Logfire cloud
    traces_endpoint = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://localhost:4318/v1/traces")
    metrics_endpoint = os.getenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", "http://localhost:9090/api/v1/otlp/v1/metrics")

    # Configure scrubbing options
    scrubbing_options = ScrubbingOptions(
        callback=scrub_sensitive_data,
        # Extra patterns are added to default patterns (password, token, key, secret, authorization)
        extra_patterns=["civit.*token", "api[_-]?key"],
    )

    # Initialize logfire with appropriate configuration based on deployment mode
    # Future optimization: Enable sampling after collecting baseline performance data
    #
    # Option 1: Simple head sampling (10% of all traces)
    # from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
    # sampling = logfire.SamplingOptions(
    #     head=ParentBased(TraceIdRatioBased(0.1))
    # )
    #
    # Option 2: Level-based tail sampling (keep slow traces and errors)
    # sampling = logfire.SamplingOptions.level_or_duration(
    #     duration_threshold=1.0,  # Always keep traces >1 second
    # )
    #
    # Current approach: 100% sampling for maximum initial visibility

    if use_logfire_cloud:
        # Mode 1: Send to Logfire cloud platform
        # Support for custom base URL (e.g., self-hosted or EU region)
        logfire_base_url = os.getenv("LOGFIRE_BASE_URL")
        if logfire_base_url:
            logfire.configure(
                send_to_logfire=True,
                token=logfire_token,
                service_name="hordelib",
                scrubbing=scrubbing_options,
                advanced=logfire.AdvancedOptions(base_url=logfire_base_url),
            )
        else:
            logfire.configure(
                send_to_logfire=True,
                token=logfire_token,
                service_name="hordelib",
                scrubbing=scrubbing_options,
            )
    else:
        # Mode 2: Local OTLP deployment - do not send to Logfire cloud
        logfire.configure(
            send_to_logfire=False,
            service_name="hordelib",
            scrubbing=scrubbing_options,
        )

    # Auto-instrumentation: Pydantic validation
    # Captures validation events for all Pydantic models (ImagePayload, ModelGenerationInputStable, etc.)
    # Only records validation failures to reduce noise from library initialization
    logfire.instrument_pydantic(record="failure")

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
    # ALL application logging should use loguru's logger, NOT logfire.info/error/debug
    # Logfire is reserved for spans (@logfire.instrument, with logfire.span) and metrics
    # WARNING: Logfire does not scrub loguru-formatted messages, so avoid logging sensitive data via loguru
    from loguru import logger

    logger.configure(handlers=[logfire.loguru_handler()])

    # Log initialization with appropriate context
    if use_logfire_cloud:
        base_url = os.getenv("LOGFIRE_BASE_URL", "https://logfire-api.pydantic.dev")
        logfire.info(
            "Logfire instrumentation initialized (Cloud Mode)",
            mode="cloud",
            base_url=base_url,
            scrubbing_enabled=True,
            sampling_rate=1.0,
        )
    else:
        logfire.info(
            "Logfire instrumentation initialized (Local OTLP Mode)",
            mode="local",
            traces_endpoint=traces_endpoint,
            metrics_endpoint=metrics_endpoint,
            scrubbing_enabled=True,
            sampling_rate=1.0,
        )

    # Note: Deep ComfyUI instrumentation is deferred to do_comfy_import()
    # in comfy_horde.py after ComfyUI modules are loaded into sys.path


def initialize_comfy_internals() -> None:
    """Initialize deep ComfyUI internals instrumentation.

    This applies monkey-patches to ComfyUI internal functions for comprehensive
    visibility into node execution, model loading, sampling, and memory management.

    This function should be called AFTER ComfyUI has been imported and added to sys.path,
    typically from do_comfy_import() in comfy_horde.py.
    """
    try:
        from hordelib.integrations.logfire_comfy_internals import instrument_comfy_internals

        instrument_comfy_internals()
        logfire.info("Deep ComfyUI instrumentation applied")
    except Exception as e:
        logfire.warn("Failed to apply deep ComfyUI instrumentation", error=str(e))

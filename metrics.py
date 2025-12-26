"""Prometheus metrics for monitoring TTS performance."""
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST


# Request metrics
request_counter = Counter(
    'tts_requests_total',
    'Total number of TTS requests',
    ['endpoint', 'status']
)

request_duration = Histogram(
    'tts_request_duration_seconds',
    'Request duration in seconds',
    ['endpoint'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

# Time to first byte (TTFB) metric
ttfb_histogram = Histogram(
    'tts_ttfb_seconds',
    'Time to first byte in seconds',
    buckets=[0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
)

# Audio generation metrics
audio_duration_generated = Counter(
    'tts_audio_duration_seconds_total',
    'Total duration of audio generated in seconds'
)

generation_speed = Histogram(
    'tts_generation_speed_ratio',
    'Ratio of audio duration to generation time (realtime factor)',
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
)

# Active requests
active_requests = Gauge(
    'tts_active_requests',
    'Number of currently active requests'
)

# Performance instrumentation metrics
tokens_per_second = Histogram(
    'tts_tokens_per_second',
    'Token generation rate (tokens/second)',
    buckets=[50, 100, 200, 300, 400, 500, 750, 1000]
)

snac_decode_seconds = Histogram(
    'tts_snac_decode_seconds',
    'SNAC decode latency per chunk (seconds)',
    buckets=[0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2]
)

time_to_first_audio = Histogram(
    'tts_time_to_first_audio_seconds',
    'Time from request start to first audio chunk (seconds)',
    buckets=[0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
)

# GPU metrics
gpu_memory_used = Gauge(
    'tts_gpu_memory_bytes',
    'GPU memory used in bytes'
)

gpu_utilization = Gauge(
    'tts_gpu_utilization_percent',
    'GPU utilization percentage'
)

# Error metrics
error_counter = Counter(
    'tts_errors_total',
    'Total number of errors',
    ['error_type']
)


def get_metrics():
    """Return current metrics in Prometheus format."""
    return generate_latest()


def get_metrics_content_type():
    """Return the content type for Prometheus metrics."""
    return CONTENT_TYPE_LATEST

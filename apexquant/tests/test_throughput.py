"""Throughput harness and the memory-traffic roofline model."""

from apexquant import ApexQuantConfig, benchmark_throughput, memory_traffic_model


def test_traffic_ratio_tracks_bit_width():
    # Memory-bound speedup ceiling ~ 16/bits (minus small scale/codebook overhead).
    m4 = memory_traffic_model(4096, 4096, bits=4)
    m2 = memory_traffic_model(4096, 4096, bits=2)
    assert 3.5 < m4.traffic_ratio < 4.0
    assert 7.0 < m2.traffic_ratio < 8.0
    assert m2.traffic_ratio > m4.traffic_ratio


def test_decode_is_memory_bound_prefill_less_so():
    decode = memory_traffic_model(4096, 4096, bits=4, batch=1)
    prefill = memory_traffic_model(4096, 4096, bits=4, batch=4096)
    assert decode.memory_bound
    # Arithmetic intensity rises with batch (more compute per weight byte).
    assert prefill.arithmetic_intensity > decode.arithmetic_intensity


def test_benchmark_runs_and_reports_positive_numbers():
    r = benchmark_throughput(512, 512, ApexQuantConfig(), batch=1, iters=5, warmup=2)
    assert r.latency_ms > 0
    assert r.tokens_per_s > 0
    assert r.compression > 1.0

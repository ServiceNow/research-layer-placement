# Quick Start Guide: Likelihood REST Backend Tests

## 🚀 Run Tests

### Unit Tests (Fast - Recommended)

```bash
# Run all unit tests (8 tests, ~9 seconds)
pytest tests/test_likelihood_rest_unit.py -v

# Expected: 8 passed in ~9s
```

### Performance Benchmarks

```bash
# Run benchmarks with detailed output
pytest tests/benchmark_likelihood_rest.py -v -s

# Expected: 5-6 passed with performance metrics printed
```

### Integration Tests (Requires GPU)

```bash
# Skip by default - requires environment variable
RUN_INTEGRATION_TESTS=1 pytest tests/test_likelihood_rest_integration.py -v

# Requirements:
# - GPU (A100/H100 recommended)
# - vLLM installed
# - Model downloaded (default: Qwen/Qwen2-0.5B-Instruct)
```

## 📊 Expected Performance

### Mocked Tests (Unit + Benchmarks)

| Metric | Value |
|--------|-------|
| Problems/sec | 10,000-14,000 |
| Tokens/sec | 35,000-41,000 |
| Latency | <1ms per request |
| Optimal batch | 16-32 chats |

### Real Servers (Integration)

| Configuration | Expected Problems/sec |
|--------------|---------------------|
| 1 GPU (A100) | 10-20 |
| 2 GPUs (A100) | 18-35 |
| 4 GPUs (A100) | 35-76 |

## 📁 Files Overview

```
tests/
├── test_likelihood_rest_unit.py          [8 unit tests ✓]
├── test_likelihood_rest_integration.py   [3 integration tests, GPU required]
├── benchmark_likelihood_rest.py          [6 benchmarks]
├── README_LIKELIHOOD_REST.md             [Full documentation]
├── PERFORMANCE_RESULTS.md                [Detailed performance analysis]
├── SUMMARY.md                            [High-level overview]
├── TEST_RESULTS.md                       [Test execution results]
└── QUICK_START.md                        [This file]
```

## 🎯 What Gets Tested

### Unit Tests
- ✅ Single/multi-choice scoring
- ✅ Round-robin distribution (2-3 servers)
- ✅ Failover when servers fail
- ✅ Concurrent request handling
- ✅ Placement configuration

### Benchmarks
- ✅ Throughput measurement
- ✅ Multi-server speedup analysis
- ✅ Performance scaling (1-4 servers)
- ✅ Batch size optimization
- ✅ Concurrent request performance

### Integration Tests (with real servers)
- ⏭️ Single server evaluation
- ⏭️ Distributed evaluation
- ⏭️ Performance measurement

## 💡 Common Tasks

### Check if tests pass

```bash
pytest tests/test_likelihood_rest_unit.py -v
# Look for: 8 passed
```

### Measure performance

```bash
pytest tests/benchmark_likelihood_rest.py -v -s | grep "problems/sec"
```

### Test with real servers

```bash
# Set up environment
export RUN_INTEGRATION_TESTS=1
export TEST_MODEL_PATH="Qwen/Qwen2-0.5B-Instruct"

# Run integration tests
pytest tests/test_likelihood_rest_integration.py -v
```

### Run all tests (unit only, skip integration)

```bash
pytest tests/test_likelihood_rest_unit.py tests/benchmark_likelihood_rest.py -v
```

## 🔍 Troubleshooting

### Tests fail with "Connection refused"

This is expected - unit tests use mocked HTTP, so no real servers needed.

### Integration tests always skip

Set environment variable:
```bash
RUN_INTEGRATION_TESTS=1 pytest tests/test_likelihood_rest_integration.py -v
```

### Benchmarks show unexpected performance

Mocked tests measure distribution overhead only. Real-world performance with actual inference is much different (see PERFORMANCE_RESULTS.md).

### Import errors

Make sure you're using the project's venv:
```bash
/home/toolkit/code/research-layer-placement/.venv/bin/python -m pytest tests/test_likelihood_rest_unit.py -v
```

## 📖 More Information

- **Detailed docs**: [README_LIKELIHOOD_REST.md](README_LIKELIHOOD_REST.md)
- **Performance**: [PERFORMANCE_RESULTS.md](PERFORMANCE_RESULTS.md)
- **Overview**: [SUMMARY.md](SUMMARY.md)
- **Test results**: [TEST_RESULTS.md](TEST_RESULTS.md)

## ✅ Success Criteria

After running tests, you should see:

```
tests/test_likelihood_rest_unit.py::TestChatLogProbsSingleServer::test_single_choice_scoring PASSED
tests/test_likelihood_rest_unit.py::TestChatLogProbsSingleServer::test_batch_of_problems PASSED
tests/test_likelihood_rest_unit.py::TestChatLogProbsDistribution::test_round_robin_distribution PASSED
tests/test_likelihood_rest_unit.py::TestChatLogProbsDistribution::test_three_server_distribution PASSED
tests/test_likelihood_rest_unit.py::TestChatLogProbsFailover::test_failover_to_healthy_server PASSED
tests/test_likelihood_rest_unit.py::TestChatLogProbsFailover::test_all_servers_fail PASSED
tests/test_likelihood_rest_unit.py::TestChatLogProbsPerformance::test_concurrent_requests PASSED
tests/test_likelihood_rest_unit.py::TestChatLogProbsPlacement::test_placement_in_request PASSED

======================== 8 passed in 8.91s ==========================
```

**Coverage**: 80% of REST backend code

**Performance**: ~12K problems/sec (mocked), 10-50 problems/sec (real)

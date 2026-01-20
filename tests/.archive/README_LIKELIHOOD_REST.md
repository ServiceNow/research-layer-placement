# Likelihood Evaluator REST Backend Tests

This directory contains comprehensive tests for the likelihood evaluator with REST backend integration. The tests are split into two categories:

**📊 [View Performance Results](PERFORMANCE_RESULTS.md)** - Detailed benchmarks with throughput, speedup, and optimization analysis.

## 1. Unit Tests ([test_likelihood_rest_unit.py](test_likelihood_rest_unit.py))

Fast, mocked tests that verify REST backend integration without requiring actual servers or task loading.

### What's Tested

#### Single Server Operations
- `test_single_choice_scoring`: Basic scoring of 4-choice multiple-choice questions
- `test_batch_of_problems`: Scoring multiple problems in one batch

#### Multi-Server Distribution
- `test_round_robin_distribution`: Verifies requests distribute across 2 servers
- `test_three_server_distribution`: Verifies distribution across 3 servers

#### Failover Behavior
- `test_failover_to_healthy_server`: When one server fails, requests failover to healthy server
- `test_all_servers_fail`: Proper error handling when all servers fail

#### Performance
- `test_concurrent_requests`: Multiple concurrent requests handled correctly

#### Placement
- `test_placement_in_request`: Placement configuration sent with requests

### Running Unit Tests

```bash
# Run all unit tests
pytest tests/test_likelihood_rest_unit.py -v

# Run specific test class
pytest tests/test_likelihood_rest_unit.py::TestChatLogProbsFailover -v

# Run with coverage
pytest tests/test_likelihood_rest_unit.py --cov=src/place_layers/backend/rest
```

### Test Architecture

Tests use:
- **aioresponses**: Mock HTTP responses from REST servers
- **Mocked `/info` endpoint**: Server validation during backend initialization
- **Mocked `/compute_chat_log_probs`**: Likelihood scoring endpoint

Key patterns:
```python
def mock_server_info() -> dict:
    """Standard mock /info response."""
    return {
        "model_path": "/models/qwen2-0.5b",
        "batch_size": 8,
        "supports_placement": True,
        "num_layers": 24,
        "mixer_names": ["attention", "gdn"],
    }

def mock_chat_log_probs_response(num_choices: int = 4, correct_idx: int = 0) -> dict:
    """Mock /compute_chat_log_probs response."""
    # Returns ChatLogProbResult format with token_ids, logprobs, answer_range, total_log_prob
```

## 2. Integration Tests ([test_likelihood_rest_integration.py](test_likelihood_rest_integration.py))

Real end-to-end tests that spin up actual local Transformers backend servers and run evaluations.

### What's Tested

#### Real Server Evaluation
- `test_single_server_evaluation`: Eval with one local server on MMLU subset
- `test_distributed_evaluation`: Eval distributed across two local servers
- `test_performance_measurement`: Measure throughput with different batch sizes

### Running Integration Tests

Integration tests are **skipped by default** and require the `RUN_INTEGRATION_TESTS=1` environment variable:

```bash
# Run integration tests (CPU is fine, takes ~1-2 min per test)
RUN_INTEGRATION_TESTS=1 pytest tests/test_likelihood_rest_integration.py -v -s

# Optionally use GPU for faster inference
RUN_INTEGRATION_TESTS=1 TEST_DEVICE_MAP=auto pytest tests/test_likelihood_rest_integration.py -v -s

# Set custom model for testing
RUN_INTEGRATION_TESTS=1 TEST_MODEL_PATH="Qwen/Qwen2-0.5B-Instruct" pytest tests/test_likelihood_rest_integration.py -v -s
```

### Requirements for Integration Tests

1. **CPU or GPU**: Tests work on CPU (slower) or GPU (faster)
2. **Model downloaded**: Default is `Qwen/Qwen2-0.5B-Instruct` (set via `TEST_MODEL_PATH` env var)
3. **Available ports**: 8080, 8081 must be free
4. **Startup time**: ~30 seconds per server for model loading on CPU

### Server Management

The `ServerManager` class handles:
- Starting multiple vLLM servers on different ports
- Waiting for servers to be ready via `/info` endpoint
- Graceful shutdown after tests complete

Example usage:
```python
manager = ServerManager("Qwen/Qwen2-0.5B-Instruct", ports=[8080, 8081])
urls = await manager.start()  # Blocks until servers ready
# ... run tests ...
manager.stop()  # Clean shutdown
```

### Performance Measurement

The integration tests measure:
- **Problems/sec**: How many MMLU problems evaluated per second
- **Accuracy**: Actual model accuracy on task
- **Throughput**: Total tokens processed per second
- **Speedup**: Performance gain from multiple servers

Example output:
```
=== Performance Optimization Results ===
Model: Qwen/Qwen2-0.5B-Instruct
Servers: 2
Problems: 50 each

Batch Size | Time (s) | Problems/s | Chats/s | Accuracy
----------------------------------------------------------
         8 |    12.34 |        4.1 |    16.2 | 0.420
        16 |     8.56 |        5.8 |    23.4 | 0.420
        32 |     7.21 |        6.9 |    27.7 | 0.420

Optimal batch size: 32 (6.9 problems/sec)
```

## Test Coverage

The tests achieve **80% coverage** of the REST backend code (`src/place_layers/backend/rest.py`), covering:
- Server health tracking and recovery
- Round-robin distribution across servers
- Failover and retry logic
- Chat log-prob computation
- Placement handling

## Quick Start

### Run just the fast unit tests:
```bash
pytest tests/test_likelihood_rest_unit.py -v
```

### Run everything including integration tests:
```bash
# Set up environment
export TEST_MODEL_PATH="Qwen/Qwen2-0.5B-Instruct"

# Run unit tests first (fast)
pytest tests/test_likelihood_rest_unit.py -v

# Run integration tests (requires GPU, takes ~5-10 minutes)
RUN_INTEGRATION_TESTS=1 pytest tests/test_likelihood_rest_integration.py -v -s
```

## Debugging

### Enable verbose logging:
```bash
pytest tests/test_likelihood_rest_unit.py -v -s --log-cli-level=DEBUG
```

### Check server health during integration tests:
```python
# In test, after backend initialization:
for server, health in backend._server_pool._health.items():
    print(f"{server}: healthy={health.healthy}, failures={health.consecutive_failures}")
```

### Profile performance:
```bash
RUN_INTEGRATION_TESTS=1 pytest tests/test_likelihood_rest_integration.py -v -s --profile
```

## Architecture Notes

### REST Backend Distribution

The REST backend (`src/place_layers/backend/rest.py`) distributes work via:

1. **`_distribute_batch()`**: Splits items across servers round-robin
2. **Server health tracking**: Failed servers excluded until recovery backoff expires
3. **Exponential backoff**: `wait = min(base * 2^(failures-1), max_interval)`
4. **Retry logic**: Uses tenacity for request retries with failover

### Likelihood Evaluator Flow

1. **Load tasks**: Load MMLU/ARC tasks via lm-eval
2. **Build chats**: For each problem, create `(context, completion)` tuples for each choice
3. **Compute log-probs**: Send chats to REST backend
4. **Select argmax**: Choose answer with highest log-prob
5. **Compare to gold**: Compute accuracy

### Chat Log-Prob Format

Request:
```json
{
  "chats": [
    [
      [{"role": "user", "content": "Question..."}],
      [{"role": "assistant", "content": "A"}]
    ],
    ...
  ],
  "placement": ["attention", "gdn", ...]
}
```

Response:
```json
{
  "results": [
    {
      "token_ids": [128006, 78191, 198, 65, 128007],
      "logprobs": [null, -0.01, -0.02, -0.1, -0.01],
      "answer_range": [3, 4],
      "total_log_prob": -0.1
    },
    ...
  ]
}
```

## Contributing

When adding new tests:

1. **Unit tests**: Add to `test_likelihood_rest_unit.py` for fast feedback
2. **Integration tests**: Add to `test_likelihood_rest_integration.py` for end-to-end validation
3. **Mock responses**: Use `mock_chat_log_probs_response()` helper
4. **Server setup**: Use `backend_with_mocks()` context manager for unit tests
5. **Performance tests**: Measure and print metrics for optimization

## Related Files

- [src/place_layers/backend/rest.py](../src/place_layers/backend/rest.py): REST backend implementation
- [src/place_layers/evaluator/likelihood.py](../src/place_layers/evaluator/likelihood.py): Likelihood evaluator
- [tests/test_rest_backend.py](test_rest_backend.py): Lower-level REST backend tests
- [CLAUDE.md](../CLAUDE.md): Project documentation

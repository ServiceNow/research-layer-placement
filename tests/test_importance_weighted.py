"""Tests for importance-weighted accuracy evaluator edge cases."""

from place_layers.evaluator.importance_weighted import (
    TraceDataset,
    TraceSource,
)


class TestTraceDataset:
    """Tests for TraceDataset edge cases."""

    def test_accuracy_empty_dataset(self):
        """Test accuracy with empty dataset returns 0, not division by zero."""
        empty_dataset = TraceDataset(
            traces=(),
            teacher_model_id="test",
            task_names=(),
            samples_per_problem=0,
        )
        assert empty_dataset.accuracy == 0.0

    def test_avg_completion_length_empty(self):
        """Test avg_completion_length with empty dataset returns 0, not division by zero."""
        empty_dataset = TraceDataset(
            traces=(),
            teacher_model_id="test",
            task_names=(),
            samples_per_problem=0,
        )
        assert empty_dataset.avg_completion_length == 0.0


class TestTraceSource:
    """Tests for TraceSource custom logic."""

    def test_display_name_defaults_to_path_basename(self):
        """Test that display_name derives from path when name not provided."""
        source = TraceSource(
            path="/path/to/math_traces.json",
            weight=1.0,
        )
        assert source.name is None
        assert source.display_name == "math_traces"

    def test_display_name_uses_explicit_name(self):
        """Test that display_name uses explicit name when provided."""
        source = TraceSource(
            path="/path/to/traces.json",
            weight=2.0,
            name="MATH",
        )
        assert source.display_name == "MATH"

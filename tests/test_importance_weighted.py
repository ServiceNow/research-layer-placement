"""Tests for importance-weighted accuracy evaluator edge cases."""

from place_layers.evaluator.traces import TraceDataset


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

"""Utilities for prompt-conditioned reward datasets and models."""

from place_layers.router.data import (
    PromptTraceGroup,
    RewardDatasetSplit,
    load_reward_dataset_split_from_score_log,
    load_prompt_trace_groups,
    load_sampled_placements,
    sample_random_placements,
    save_sampled_placements,
    score_problem_trace_matrix,
)
from place_layers.router.embeddings import (
    extract_prompt_embeddings,
    load_prompt_embedding_artifact,
    save_prompt_embedding_artifact,
)
from place_layers.router.loss import (
    ListCrossEntropyRewardLossConfig,
    MeanSquaredErrorRewardLossConfig,
    RewardLossConfig,
)
from place_layers.router.metrics import (
    RewardModelMetrics,
    evaluate_reward_model,
)
from place_layers.router.model import (
    RewardModelConfig,
    RewardModelFitResult,
    RewardOptimizerConfig,
    build_checkpoint_payload,
    build_placement_features,
    placement_feature_dim,
)

__all__ = [
    "ListCrossEntropyRewardLossConfig",
    "MeanSquaredErrorRewardLossConfig",
    "PromptTraceGroup",
    "RewardDatasetSplit",
    "RewardModelConfig",
    "RewardModelFitResult",
    "RewardModelMetrics",
    "RewardLossConfig",
    "RewardOptimizerConfig",
    "build_checkpoint_payload",
    "build_placement_features",
    "evaluate_reward_model",
    "extract_prompt_embeddings",
    "load_prompt_embedding_artifact",
    "load_reward_dataset_split_from_score_log",
    "load_prompt_trace_groups",
    "load_sampled_placements",
    "placement_feature_dim",
    "sample_random_placements",
    "save_prompt_embedding_artifact",
    "save_sampled_placements",
    "score_problem_trace_matrix",
]

"""Pure functions for hierarchical beam search.

This module contains stateless, reusable functions that implement the core
mechanics of beam search. These functions are shared between:
- HierarchicalBeamSearch (standalone, for count-budget problems)
- HierarchicalSearch (inner loop for cost-constrained problems)

All functions are pure - they take explicit inputs and return outputs without
modifying any state or relying on class attributes.
"""


def generate_beam_candidates(
    beam: list[dict],
    unassigned: list[int],
) -> list[dict]:
    """Generate new candidates by expanding each beam entry with one additional layer.

    For each entry in the beam, tries adding each unassigned layer that isn't
    already in that entry. Deduplicates candidates using sorted layer tuples.

    Args:
        beam: Current beam entries, each with a "layers" key containing list[int].
        unassigned: Layer indices available for assignment.

    Returns:
        List of candidate dicts with "layers" (sorted list) and "score" (None).
    """
    candidates = []
    seen: set[tuple[int, ...]] = set()

    for beam_entry in beam:
        existing = set(beam_entry["layers"])

        for layer_idx in unassigned:
            if layer_idx in existing:
                continue

            new_layers = tuple(sorted(beam_entry["layers"] + [layer_idx]))

            if new_layers in seen:
                continue
            seen.add(new_layers)

            candidates.append({"layers": list(new_layers), "score": None})

    return candidates


def create_layer_assignment(
    fixed_assignments: dict[int, int],
    target_layers: list[int],
    target_mixer_idx: int,
    unassigned: list[int],
    fallback_mixer_idx: int,
    num_layers: int,
) -> tuple[int, ...]:
    """Create a complete layer-to-mixer assignment tuple.

    Combines three sources of assignments:
    1. Fixed assignments from previous phases (already decided)
    2. Target layers assigned to target mixer (being evaluated)
    3. Remaining unassigned layers assigned to fallback mixer

    Args:
        fixed_assignments: Layer -> mixer mapping from previous phases.
        target_layers: Layers to assign to target_mixer_idx for this evaluation.
        target_mixer_idx: Mixer index for target layers.
        unassigned: All layers not yet permanently assigned.
        fallback_mixer_idx: Mixer for unassigned layers not in target_layers.
        num_layers: Total number of layers (determines tuple length).

    Returns:
        Tuple of mixer indices, one per layer.
    """
    assignment = dict(fixed_assignments)

    # Assign target layers to target mixer
    for layer_idx in target_layers:
        assignment[layer_idx] = target_mixer_idx

    # Assign remaining unassigned layers to fallback
    for layer_idx in unassigned:
        if layer_idx not in assignment:
            assignment[layer_idx] = fallback_mixer_idx

    return tuple(assignment[i] for i in range(num_layers))


def select_top_beam(
    candidates: list[dict],
    beam_width: int,
    higher_is_better: bool = True,
) -> list[dict]:
    """Sort candidates by score and return top beam_width entries.

    Args:
        candidates: List of dicts with "score" key (must not be None).
        beam_width: Number of top candidates to keep.
        higher_is_better: If True, higher scores are better.

    Returns:
        Top beam_width candidates, sorted by score.
    """
    candidates.sort(key=lambda x: x["score"], reverse=higher_is_better)
    return candidates[:beam_width]

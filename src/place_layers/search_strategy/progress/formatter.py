"""Placement formatting utilities for search progress observers.

This module provides the PlacementFormatterMixin class that encapsulates all
placement formatting logic, enabling consistent display across different
observer implementations (RichSearchDisplay, PlainObserver, etc.).
"""

from __future__ import annotations

from collections.abc import Sequence


class PlacementFormatterMixin:
    """Mixin providing placement formatting methods.

    Subclasses must set _mixer_names and _num_layers before calling these methods.
    Typically done in on_search_started() from event data.
    """

    _mixer_names: tuple[str, ...]
    _num_layers: int

    def _format_layer_list(self, layers: Sequence[int], *, max_show: int = 10) -> str:
        """Format a list of layer indices compactly.

        Args:
            layers: Layer indices to format
            max_show: Maximum layers to show before truncating

        Returns:
            Comma-separated string like "1,2,3" or "1,2,...,9" if truncated
        """
        if len(layers) <= max_show:
            return ",".join(str(layer) for layer in layers)
        first_half = max_show // 2
        return ",".join(str(layer) for layer in layers[:first_half]) + f",...,{layers[-1]}"

    def _format_layer_assignment(
        self,
        mixer: str,
        layers: Sequence[int],
        *,
        max_show: int = 10,
        escape_brackets: bool = False,
    ) -> str:
        """Format a single mixer's layer assignment.

        Args:
            mixer: Mixer name (e.g., "FA", "SWA")
            layers: Layer indices assigned to this mixer
            max_show: Maximum layers to show before truncating
            escape_brackets: Whether to escape brackets for Rich markup

        Returns:
            String like "FA@[1,2,3]" or "FA@\\[1,2,3]" if escaped
        """
        layers_str = self._format_layer_list(layers, max_show=max_show)
        bracket = "\\[" if escape_brackets else "["
        return f"{mixer}@{bracket}{layers_str}]"

    def _group_layers_by_mixer(
        self,
        assignments: tuple[str | None, ...],
    ) -> dict[str, list[int]]:
        """Group layer indices by their assigned mixer.

        Args:
            assignments: Per-layer tuple where index=layer, value=mixer or None.
                        None entries (unassigned layers) are skipped.

        Returns:
            Dict mapping mixer name to sorted list of layer indices.
        """
        by_mixer: dict[str, list[int]] = {}

        for layer_idx, mixer in enumerate(assignments):
            if mixer is not None:
                by_mixer.setdefault(mixer, []).append(layer_idx)

        # Sort layers within each mixer
        for layers in by_mixer.values():
            layers.sort()

        return by_mixer

    def _format_placement(
        self,
        by_mixer: dict[str, Sequence[int]],
        *,
        max_layers_per_mixer: int = 10,
        escape_brackets: bool = False,
    ) -> str:
        """Format pre-grouped placement as string.

        Args:
            by_mixer: Dict mapping mixer name to layer indices.
                     Use _group_layers_by_mixer to normalize tuples first.
            max_layers_per_mixer: Max layers to show per mixer before truncating
            escape_brackets: Whether to escape brackets for Rich markup

        Returns:
            String like "FA@[0,2], GDN@[1], SWA@[3,4,5]"
        """
        # Format in canonical mixer_names order
        parts = []
        for mixer in self._mixer_names:
            if mixer in by_mixer:
                parts.append(
                    self._format_layer_assignment(
                        mixer,
                        by_mixer[mixer],
                        max_show=max_layers_per_mixer,
                        escape_brackets=escape_brackets,
                    )
                )
        return ", ".join(parts)

    def _format_allocation(self, allocation: tuple[int, ...]) -> str:
        """Format allocation as 'FA:2 SWA:1 GDN:7'."""
        if not self._mixer_names:
            return str(allocation)
        parts = [f"{name}:{count}" for name, count in zip(self._mixer_names, allocation)]
        return " ".join(parts)

    def _format_placement_diff(
        self,
        base: tuple[str | None, ...],
        new: tuple[str | None, ...],
        *,
        max_layers_per_mixer: int = 6,
    ) -> str:
        """Format the diff between two placements.

        Returns layers that changed from None→assigned in base→new,
        grouped by their new mixer. Separator: ' + '.

        Args:
            base: Base placement (partial, may have None)
            new: New placement (partial or complete)
            max_layers_per_mixer: Max layers before truncation

        Returns:
            String like "FA@0 + SWA@[1,2,...,47]"
        """
        # Find changed layers grouped by new mixer
        changed_by_mixer: dict[str, list[int]] = {}
        for layer_idx, (old, current) in enumerate(zip(base, new)):
            if old is None and current is not None:
                changed_by_mixer.setdefault(current, []).append(layer_idx)

        # Format in canonical order, joined by " + "
        parts = []
        for mixer in self._mixer_names:
            if mixer in changed_by_mixer:
                layers = changed_by_mixer[mixer]
                if len(layers) == 1:
                    # Single layer: no brackets
                    parts.append(f"{mixer}@{layers[0]}")
                else:
                    # Multiple layers: with brackets
                    layers_str = self._format_layer_list(layers, max_show=max_layers_per_mixer)
                    parts.append(f"{mixer}@[{layers_str}]")

        return " + ".join(parts)

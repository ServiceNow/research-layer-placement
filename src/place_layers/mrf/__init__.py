"""MRF analysis module for chain Markov Random Fields.

This module provides tools for analyzing chain-structured MRFs, including:
- ChainMRFModel: Represents a chain MRF with unary and pairwise potentials
- ExactMRFSolver: Viterbi-style DP solver for finding optimal placements
"""

from place_layers.mrf.model import ChainMRFModel
from place_layers.mrf.solver import ExactMRFSolver, ExactSolution

__all__ = ["ChainMRFModel", "ExactMRFSolver", "ExactSolution"]

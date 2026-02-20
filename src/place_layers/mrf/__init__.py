"""MRF analysis module for chain Markov Random Fields.

This module provides tools for analyzing chain-structured MRFs, including:
- ChainMRFModel: Represents a chain MRF with unary and pairwise potentials
- MRFModelProtocol: Protocol interface used by ExactMRFSolver
- SurrogateMRFModel: MRF model backed by ridge regression coefficients
- ExactMRFSolver: Viterbi-style DP solver for finding optimal placements
"""

from place_layers.mrf.model import ChainMRFModel, MRFModelProtocol
from place_layers.mrf.pareto_lcb_solver import ParetoLCBSolver
from place_layers.mrf.solver import ExactMRFSolver, ExactSolution
from place_layers.mrf.surrogate_model import SurrogateMRFModel

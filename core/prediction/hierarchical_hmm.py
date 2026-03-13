import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from core.prediction.hmm_engine import HMMEngine, HMMState


class StrategicGoal(Enum):
    DEBUGGING = 0
    DEVELOPING = 1
    REFACTORING = 2
    SYNCING = 3
    IDLE = 4


@dataclass
class HierarchicalPredictionResult:
    tactical_state: HMMState
    strategic_goal: StrategicGoal
    tactical_confidence: float
    strategic_confidence: float


class HierarchicalHMMEngine:
    """
    Hierarchical HMM for multi-layer strategic goal estimation.
    L2 (Strategic) states supervise L1 (Tactical) state transitions.
    """

    def __init__(self):
        self.logger = logging.getLogger("HierarchicalHMMEngine")

        # L2: Strategic Layer
        self.strategic_states = [g for g in StrategicGoal]
        self.num_strategic = len(self.strategic_states)
        self.strategic_transitions = np.full(
            (self.num_strategic, self.num_strategic), 1.0 / self.num_strategic
        )
        self.strategic_priors = np.full(self.num_strategic, 1.0 / self.num_strategic)

        # L1: Tactical Layer (Conditioned on L2)
        # For each strategic goal, we have a specific tactical transition matrix
        self.tactical_engines: Dict[StrategicGoal, HMMEngine] = {
            goal: HMMEngine() for goal in StrategicGoal
        }

        # Current belief state
        self.current_strategic_belief = np.copy(self.strategic_priors)
        self.last_strategic_goal = StrategicGoal.IDLE

    def predict(self, observations: List[str]) -> HierarchicalPredictionResult:
        """
        Jointly estimates strategic goal and next tactical state.
        Uses a Bayesian update for the strategic layer based on tactical likelihoods.
        """
        if not observations:
            return HierarchicalPredictionResult(
                HMMState.IDLE, StrategicGoal.IDLE, 0.5, 0.5
            )

        # 1. Get tactical likelihoods for each strategic context
        tactical_likelihoods = []
        best_tactical_per_goal = {}

        for goal in StrategicGoal:
            engine = self.tactical_engines[goal]
            decoded_states = engine.decode(observations)
            pred = decoded_states[-1]
            # Use the engine's internal confidence as a proxy for likelihood

            likelihood = self._estimate_likelihood(engine, observations)
            tactical_likelihoods.append(likelihood)
            best_tactical_per_goal[goal] = pred

        # 2. Update Strategic Belief (L2)
        # P(S | O) \propto P(O | S) * P(S)
        unnormalized_belief = (
            np.array(tactical_likelihoods) * self.current_strategic_belief
        )
        total = np.sum(unnormalized_belief)

        if total > 0:
            self.current_strategic_belief = unnormalized_belief / total
        else:
            self.current_strategic_belief = np.full(
                self.num_strategic, 1.0 / self.num_strategic
            )

        # 3. Decision
        strategic_idx = np.argmax(self.current_strategic_belief)
        best_goal = self.strategic_states[strategic_idx]
        best_tactical = best_tactical_per_goal[best_goal]

        return HierarchicalPredictionResult(
            tactical_state=best_tactical,
            strategic_goal=best_goal,
            tactical_confidence=float(
                np.max(self.current_strategic_belief)
            ),  # Simplified
            strategic_confidence=float(np.max(self.current_strategic_belief)),
        )

    def learn(
        self, observations: List[str], assumed_goal: Optional[StrategicGoal] = None
    ):
        """
        Learns both layers from observation data.
        """
        # If goal is provided, train that specific tactical engine
        if assumed_goal:
            self.tactical_engines[assumed_goal].learn(observations)
            # Update strategic transitions towards this goal
            idx = assumed_goal.value
            self.strategic_transitions[:, idx] += 0.05
            self._normalize_matrices()
        else:
            # Autonomous assignment based on current belief
            pred = self.predict(observations)
            self.tactical_engines[pred.strategic_goal].learn(observations)

    def _estimate_likelihood(self, engine: HMMEngine, observations: List[str]) -> float:
        """
        Computes the log-likelihood of observations under a specific engine.
        Using a simplified sum-of-path probabilities.
        """
        # This is a placeholder for the full forward-algorithm likelihood
        # In HMMEngine, predict_state uses Viterbi. For likelihood, we'd want the Alpha sum.
        # For now, we use a mock likelihood proportional to the 'learnability'
        # or consistency with the existing tactical priors.
        return 1.0  # Implement full log-likelihood in production version

    def _normalize_matrices(self):
        self.strategic_transitions /= self.strategic_transitions.sum(axis=1)[
            :, np.newaxis
        ]
        self.strategic_priors /= self.strategic_priors.sum()

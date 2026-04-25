import os
import logging
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from typing import Iterable, Any, Dict, Optional
from datetime import datetime

from factor_engines.base import MiningEngine
from factor_core.candidate import FactorCandidate
from factor_io.specs import MiningExperimentSpec
from mining.context import FactorMiningContext

# Imports from alpha_gfn
from alpha_gfn.modules import SequenceEncoder, SimpleNeuralNet
from alpha_gfn.gflownet import EntropyTBGFlowNet
from alpha_gfn.config import HIDDEN_DIM
from alpha_gfn.env.core import GFNEnvCore
from gfn.samplers import Sampler
from gfn.modules import DiscretePolicyEstimator

logger = logging.getLogger(__name__)

class GFNRewardAdapter:
    """Adapts GFN reward to use the unified FactorInspector."""
    def __init__(self, context: FactorMiningContext, engine: 'GFNEngine'):
        self.context = context
        self.engine = engine
        self.seen_exprs = {} # Cache reports to avoid re-evaluation in same episode

    def __call__(self, expr: Any) -> float:
        expr_str = str(expr)
        if expr_str in self.seen_exprs:
            return self.seen_exprs[expr_str]
        
        # We don't have the full candidate yet, create a temporary one
        candidate = FactorCandidate(
            factor_id="",
            expression=expr_str,
            engine="gfn",
            search_space_id=self.context.spec.search_space.family_id if self.context.spec.search_space else "unknown"
        )
        
        # Use orchestrator's inspector logic (but we are inside engine, so we need to access it)
        # For GFN reward, we typically only need TRAIN IC or similar
        # Since we want to decouple, the engine shouldn't ideally know about the inspector.
        # But for GFN, it NEEDS the reward.
        # 425-20.md suggests using the unified inspector.
        
        # Temporary solution: Engine will have its own lightweight evaluator for reward
        # or it can be passed the inspector during prepare.
        if hasattr(self.engine, 'inspector'):
            report = self.engine.inspector.inspect(candidate, self.context, splits=["train"])
            if report.decision == "rejected":
                # Give a small positive reward for "not crashing" but failing criteria
                reward = 1e-4 
            else:
                m = report.metrics.get("train").metrics
                # Use a more sensitive reward: abs(IC) + stability or something
                ic = abs(m.get("ic_mean", 0))
                reward = ic + 1e-3 # Ensure strictly positive
        else:
            reward = 1e-3
            
        self.seen_exprs[expr_str] = float(reward)
        return float(reward)

class GFNEngine:
    """
    GFlowNet Mining Engine aligned with 425-20.md.
    """
    engine_name = "gfn"
    engine_version = "0.2.0"

    def __init__(self):
        self.spec = None
        self.context = None
        self.env = None
        self.gfn = None
        self.sampler = None
        self.optimizer = None
        self.inspector = None

    def prepare(self, spec: MiningExperimentSpec, context: FactorMiningContext) -> None:
        self.spec = spec
        self.context = context
        
        # 1. Setup Inspector for reward (if not passed, create one)
        from factor_core.inspector import FactorInspector
        from factor_core.expression_quality import ExpressionQualityValidator
        from factor_eval.factor_metrics import FactorMetricsEvaluator
        
        self.inspector = FactorInspector(
            quality_validator=ExpressionQualityValidator(
                min_complexity=spec.screening.min_complexity if spec.screening else 6,
                min_ts_operators=spec.screening.min_ts_operators if spec.screening else 1,
                min_operators=spec.screening.min_operators if spec.screening else 3,
                min_features=spec.screening.min_features if spec.screening else 1
            ),
            metrics_evaluator=FactorMetricsEvaluator(device=str(context.device)),
            feature_map={str(f): f for f in context.features},
            screening_spec=spec.screening
        )

        # 2. Build Env
        self.env = GFNEnvCore(
            device=context.device,
            custom_features=context.features,
            operators=context.operators,
            delta_times=context.delta_times,
            constants=context.constants,
            max_expr_length=spec.search_space.max_expr_length if spec.search_space else 20,
            min_expr_length=spec.search_space.min_expr_length if spec.search_space else 6,
            reward_adapter=GFNRewardAdapter(context, self)
        )
        
        # 3. Build GFN Chain
        n_tokens = self.env.n_tokens
        encoder = SequenceEncoder(n_tokens, spec.engine.params.get("encoder_type", "gnn"), tokens=self.env._tokens)
        pf_head = SimpleNeuralNet(input_dim=HIDDEN_DIM, output_dim=self.env.n_actions)
        pb_head = SimpleNeuralNet(input_dim=HIDDEN_DIM, output_dim=self.env.n_actions - 1)
        
        pf = DiscretePolicyEstimator(nn.Sequential(encoder, pf_head), n_actions=self.env.n_actions, preprocessor=self.env.preprocessor)
        pb = DiscretePolicyEstimator(nn.Sequential(encoder, pb_head), n_actions=self.env.n_actions, preprocessor=self.env.preprocessor, is_backward=True)
        
        self.gfn = EntropyTBGFlowNet(pf=pf, pb=pb, entropy_coef=spec.engine.params.get("entropy_coef", 0.01))
        self.gfn.to(context.device)
        
        self.sampler = Sampler(estimator=pf)
        self.optimizer = Adam(list(encoder.parameters()) + list(pf_head.parameters()) + list(pb_head.parameters()) + [self.gfn.logZ], 
                             lr=spec.engine.params.get("learning_rate", 1e-4))

    def discover(self, context: FactorMiningContext) -> Iterable[FactorCandidate]:
        """Training loop that yields candidates."""
        n_episodes = self.spec.engine.params.get("n_episodes", 1000)
        batch_size = self.spec.engine.params.get("batch_size", 32)
        
        logger.info(f"Starting GFN discovery for {n_episodes} episodes...")
        
        pbar = tqdm(range(n_episodes))
        for episode in pbar:
            # Sample trajectories
            trajectories = self.sampler.sample_trajectories(env=self.env, n=batch_size)
            
            # Compute loss and update
            loss = self.gfn.loss(env=self.env, trajectories=trajectories)
            if loss is not None and torch.isfinite(loss):
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # Yield candidates from trajectories
            # trajectories.states is a States object, trajectories.states.tensor shape: [max_steps, batch_size, state_dim]
            # We want the terminal states. 
            for i in range(batch_size):
                # Try to get terminal state more robustly
                if hasattr(trajectories, 'last_states'):
                     terminal_state_tensor = trajectories.last_states.tensor[i]
                elif hasattr(trajectories, 'when_is_done'):
                     # Index into states using when_is_done
                     done_idx = trajectories.when_is_done[i]
                     terminal_state_tensor = trajectories.states.tensor[done_idx, i]
                else:
                     # Fallback: Search for the state that has the longest valid sequence
                     state_seq = trajectories.states.tensor[:, i] # (steps, state_dim)
                     # In GFN, states are cumulative, so the one with most non -1 is the "most terminal"
                     # or just the last one if it didn't finish early.
                     terminal_state_tensor = state_seq[-1]
                
                expr = self.env.state_to_expression(terminal_state_tensor)
                if expr is not None:
                    yield FactorCandidate(
                        factor_id="",
                        expression=str(expr),
                        engine=self.engine_name,
                        engine_version=self.engine_version,
                        metadata={"episode": episode, "batch_idx": i}
                    )
            
            if episode % 10 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

    def close(self) -> None:
        pass

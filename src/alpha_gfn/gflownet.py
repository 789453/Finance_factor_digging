import torch

# GFN library compatibility (torchgfn 2.4.0+)
from gfn.containers import Trajectories
from gfn.env import Env
from gfn.gflownet import TBGFlowNet
from gfn.modules import DiscretePolicyEstimator


class EntropyTBGFlowNet(TBGFlowNet):
    def __init__(
        self,
        pf: DiscretePolicyEstimator,
        pb: DiscretePolicyEstimator,
        entropy_coef: float = 0.0,
        entropy_temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(pf, pb, **kwargs)
        self.entropy_coef = entropy_coef
        self.entropy_temperature = entropy_temperature

    def _get_entropy_bonus(self, trajectories: Trajectories) -> torch.Tensor:
        n = trajectories.n_trajectories
        if self.entropy_coef <= 0 or trajectories.estimator_outputs is None:
            return torch.zeros(n, device=trajectories.states.device)

        logits = trajectories.estimator_outputs
        temp_logits = logits / self.entropy_temperature
        probs = torch.softmax(temp_logits, dim=-1)
        entropy_per_step = -(probs * torch.log(probs.clamp(min=1e-9))).sum(dim=-1)

        max_len = logits.shape[0]
        step_indices = torch.arange(max_len, device=logits.device).unsqueeze(1)
        valid_steps_mask = step_indices < trajectories.terminating_idx.unsqueeze(0)
        entropy_sum_per_trajectory = (entropy_per_step * valid_steps_mask).sum(dim=0)
        return entropy_sum_per_trajectory * self.entropy_coef
    
    def loss(
        self,
        env: Env,
        trajectories: Trajectories,
        recalculate_all_logprobs: bool = True,
        reduction: str = "mean",
    ) -> torch.Tensor:
        base_loss = super().loss(
            env,
            trajectories,
            recalculate_all_logprobs=recalculate_all_logprobs,
            reduction=reduction,
        )
        entropy_bonus = self._get_entropy_bonus(trajectories)

        if reduction == "none":
            return base_loss + entropy_bonus
        if reduction == "sum":
            return base_loss + entropy_bonus.sum()
        return base_loss + entropy_bonus.mean()

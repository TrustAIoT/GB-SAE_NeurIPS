from dataclasses import dataclass


@dataclass
class TrainingScheduleConfig:
    # gate warmup
    warmup_pct: float = 0.2     
    steady_pct: float = 0.6     


class TrainingPhaseController:
    """Scale gate regularization from zero through a linear ramp to one.
    
    warmup_pct marks the end of the zero-scale phase; steady_pct marks the
    end of the ramp, as fractions of total_steps. get returns aux_enable=True
    for compatibility and the current gate_scale. Auxiliary loss is not ramped.
    """

    def __init__(self, total_steps: int, cfg: TrainingScheduleConfig):
        assert 0.0 <= cfg.warmup_pct < cfg.steady_pct <= 1.0

        self.total_steps = total_steps
        self.cfg = cfg

    def _ratio(self, step: int) -> float:
        return min(step / self.total_steps, 1.0)

    def get(self, step: int):
        r = self._ratio(step)

        # gate regularization warmup
        if r < self.cfg.warmup_pct:
            gate_scale = 0.0
        elif r < self.cfg.steady_pct:
            t = (r - self.cfg.warmup_pct) / (self.cfg.steady_pct - self.cfg.warmup_pct)
            gate_scale = float(t)
        else:
            gate_scale = 1.0

        return {
            # aux is always enabled now
            "aux_enable": True,

            # scale for lambda_gate_bin / lambda_gate_ratio
            "gate_scale": gate_scale,
        }

from pathlib import Path

code = r"""
# self_parking_ai/controller/pure_pursuit.py
from dataclasses import dataclass
import numpy as np

@dataclass
class State:
    x: float
    y: float
    yaw: float   # radians

class PurePursuit:
    def __init__(self, lookahead=2.5, wheelbase=2.5, max_steer=np.deg2rad(35)):
        self.Ld = float(lookahead)
        self.L  = float(wheelbase)
        self.max_steer = float(max_steer)

    def steer(self, state: State, target_xy):
        tx, ty = target_xy
        dx = tx - state.x
        dy = ty - state.y

        # transform target to vehicle frame
        cos_y = np.cos(state.yaw); sin_y = np.sin(state.yaw)
        local_x =  cos_y * dx + sin_y * dy
        local_y = -sin_y * dx + cos_y * dy

        # lookahead distance and curvature
        Ld = max(1e-3, np.hypot(local_x, local_y))
        kappa = 2.0 * local_y / (Ld**2)  # simple pure-pursuit curvature
        delta = np.arctan(self.L * kappa)

        # clamp to steering limits
        return float(np.clip(delta, -self.max_steer, self.max_steer))
"""
p = Path("/content/self_parking_ai/controller/pure_pursuit.py")
p.write_text(code)
print("Wrote:", p, p.exists())

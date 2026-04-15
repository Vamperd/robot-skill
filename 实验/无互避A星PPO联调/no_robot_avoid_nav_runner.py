from __future__ import annotations

import math
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
NAV_DIR = REPO_ROOT / "导航结合RL运动"
if str(NAV_DIR) not in sys.path:
    sys.path.insert(0, str(NAV_DIR))

from scheduler_nav_runner import SchedulerNavRunner  # noqa: E402


class NoRobotAvoidNavRunner(SchedulerNavRunner):
    """Experimental nav runner that ignores robot-robot avoidance entirely.

    This path is intentionally isolated from the formal validation route.
    A* still uses static walls from the scenario, while robots do not perceive
    each other as neighbors and do not collide with each other.
    """

    def _neighbors(self, robot_id: str) -> list[dict]:
        return []

    def _check_collision(self, x: float, y: float, robot_id: str) -> bool:
        for ox, oy, ow, oh in self.scenario["obstacles"]:
            closest_x = max(ox, min(x, ox + ow))
            closest_y = max(oy, min(y, oy + oh))
            if math.hypot(x - closest_x, y - closest_y) < self.robot_states[robot_id]["radius"]:
                return True
        return False

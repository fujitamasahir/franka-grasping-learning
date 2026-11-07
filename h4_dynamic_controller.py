"""Observation-only, phase-limited rolling replanning rule.

The rule sees a sequence of noisy XY samples. It has no simulator, noise
profile, true sigma, evaluation seed, or transition-time argument.
"""
import numpy as np
from evaluate_h4_adaptive_replanning import estimate_sigma, choose_replanning_interval


class RollingReplanningRule:
    def __init__(self, online):
        self.online = bool(online)
        self.reset()

    def reset(self):
        self.xy = []
        self.interval = 4
        self.events = []

    def observe(self, noisy_xy):
        xy = np.asarray(noisy_xy, dtype=np.float64)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError('Rule requires a finite noisy XY observation')
        self.xy.append(xy.copy())

    def decide(self):
        """Before observing sample t, decide from samples [t-20:t]."""
        t = len(self.xy)
        if t < 20 or t > 80 or t % 4 or (not self.online and t != 20):
            return None
        if self.events and self.events[-1]['step'] == t:
            raise RuntimeError('Duplicate decision before recording next observation')
        estimate = estimate_sigma(self.xy[t-20:t])
        before = self.interval
        if t == 20:
            self.interval = choose_replanning_interval(estimate, .0075)
        elif before == 4 and estimate > .0085:
            self.interval = 1
        elif before == 1 and estimate < .0065:
            self.interval = 4
        event = {'step': t, 'window_start': t-20, 'window_end': t-1,
                 'estimated_sigma_m': estimate, 'before': before,
                 'after': self.interval, 'switched': before != self.interval}
        self.events.append(event)
        return event.copy()


def advance_at_boundary(controller, observation, event):
    """A new interval starts with a new H4 prediction at action index zero.

    In particular, simply changing interval 1->4 and calling advance would
    wrongly use action 1 of the preceding R1 prediction at the boundary.
    """
    if event is None:
        controller.advance(observation)
        return
    expected_index = controller.replan_interval - 1
    if controller.chunk_index != expected_index:
        raise RuntimeError('Decision attempted before consuming the old chunk')
    controller.replan_interval = event['after']
    controller.plan(observation)

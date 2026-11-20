"""Observation-only frozen Kalman decisions; no true state/noise/profile input."""
from h4_dynamic_controller import RollingReplanningRule
from kalman_noise_estimator import kalman_noise


class KalmanReplanningRule(RollingReplanningRule):
    def decide(self):
        t=len(self.xy)
        if t<20 or t>80 or t%4 or (not self.online and t!=20):return None
        if self.events and self.events[-1]['step']==t:raise RuntimeError('Duplicate decision')
        estimate=kalman_noise(self.xy[t-20:t])['sigma_m']
        before=self.interval
        if t==20:self.interval=4 if estimate<=.0075 else 1
        elif before==4 and estimate>.0085:self.interval=1
        elif before==1 and estimate<.0065:self.interval=4
        event={'step':t,'window_start':t-20,'window_end':t-1,'estimated_sigma_m':estimate,
               'before':before,'after':self.interval,'switched':before!=self.interval}
        self.events.append(event)
        return event.copy()

from dataclasses import dataclass

@dataclass
class GainTuner:
    """Edge-triggered D-Pad tuning: ◀/▶ -> Kp -/+ ; ▲/▼ -> Kd -/+ ; L1+◀/▶ -> Ki -/+"""
    def __init__(self, pid_x, pid_y, kp_step=0.2, ki_step=0.02, kd_step=0.05):
        self.pid_x = pid_x
        self.pid_y = pid_y
        self.kp_step = kp_step
        self.ki_step = ki_step
        self.kd_step = kd_step
        self.prev = {"dpad_left":False, "dpad_right":False, "dpad_up":False, "dpad_down":False, "L1":False}

    def _edge(self, name, cur):
        was = self.prev.get(name, False)
        self.prev[name] = cur
        return (not was) and bool(cur)

    def _apply(self, kp=None, ki=None, kd=None):
        if kp is not None:
            self.pid_x.kp = self.pid_y.kp = max(0.0, kp)
        if ki is not None:
            self.pid_x.ki = self.pid_y.ki = max(0.0, ki)
        if kd is not None:
            self.pid_x.kd = self.pid_y.kd = max(0.0, kd)
        print(f"[GAINS] Kp={self.pid_x.kp:.3f} Ki={self.pid_x.ki:.3f} Kd={self.pid_x.kd:.3f}")

    def step(self, signals: dict):
        # read with safe defaults
        left  = bool(signals.get("dpad_left",  False))
        right = bool(signals.get("dpad_right", False))
        up    = bool(signals.get("dpad_up",    False))
        down  = bool(signals.get("dpad_down",  False))
        L1    = bool(signals.get("L1",         False))

        if self._edge("left", left):
            if L1: self._apply(ki=self.pid_x.ki - self.ki_step)
            else:  self._apply(kp=self.pid_x.kp - self.kp_step)
        if self._edge("right", right):
            if L1: self._apply(ki=self.pid_x.ki + self.ki_step)
            else:  self._apply(kp=self.pid_x.kp + self.kp_step)
        if self._edge("up", up):
            self._apply(kd=self.pid_x.kd + self.kd_step)
        if self._edge("down", down):
            self._apply(kd=self.pid_x.kd - self.kd_step)
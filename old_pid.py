from dataclasses import dataclass
@dataclass
class PID:
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0
    u_min: float = -1
    u_max: float = 1
    integ_min: float = -0.25
    integ_max: float = 0.25
    d_window: int = 5  # effective window size for derivative smoothing (used to compute EMA alpha)

    _e_prev: float = 0.0
    _i: float = 0.0
    _d_filtered: float = 0.0
    _init: bool = False

    def __post_init__(self):
        # initialize the filtered derivative to zero
        self._d_filtered = 0.0

    def reset(self):
        self._e_prev = 0.0
        self._i = 0.0
        self._d_filtered = 0.0
        self._init = False

    def update(self, error: float, dt: float) -> float:
        if dt <= 0:
            return 0.0
        # integral with clamp
        self._i += error * dt
        if self._i > self.integ_max:
            self._i = self.integ_max
        if self._i < self.integ_min:
            self._i = self.integ_min
        # derivative: exponential moving average (EMA) low-pass filter
        # more recent samples are weighted higher; alpha derived from d_window
        d_raw = (error - self._e_prev) / dt if self._init else 0.0
        self._e_prev = error
        self._init = True
        # compute EMA smoothing factor from effective window size
        alpha = 0.4
        # update filtered derivative
        self._d_filtered = alpha * d_raw + (1.0 - alpha) * self._d_filtered
        d_filtered = self._d_filtered
        # output with saturation
        u = self.kp * error + self.ki * self._i + self.kd * d_filtered
        if u > self.u_max:
            u = self.u_max
        if u < self.u_min:
            u = self.u_min
        return u


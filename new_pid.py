from dataclasses import dataclass


@dataclass
class NEW_PID:
    """
    Lightweight PID controller with:
      - output saturation (u_min / u_max)
      - integral anti-windup (integ_min / integ_max)
      - optional exponential smoothing of derivative (via d_window)

    Usage:
        pid = PID(kp=10.0, ki=0.5, kd=0.1, u_min=-1.0, u_max=1.0)
        u = pid.update(error, dt)
    """
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0

    # output saturation
    u_min: float = -1.0
    u_max: float = 1.0

    # integral state saturation
    integ_min: float = -0.25
    integ_max: float = 0.25

    # effective window size for derivative smoothing (EMA)
    # d_window <= 1 disables smoothing
    d_window: int = 5

    # internal state (do not set directly)
    _e_prev: float = 0.0
    _i: float = 0.0
    _d_filtered: float = 0.0
    _initialized: bool = False

    def reset(self) -> None:
        """Reset integral and derivative state."""
        self._e_prev = 0.0
        self._i = 0.0
        self._d_filtered = 0.0
        self._initialized = False

    def update(self, error: float, dt: float) -> float:
        """Update PID with new error and timestep, return control effort."""
        if dt <= 0.0:
            dt = 1e-6

        if not self._initialized:
            self._e_prev = error
            self._d_filtered = 0.0
            self._initialized = True

        # --- Integral term with clamping ---
        self._i += error * dt
        if self._i > self.integ_max:
            self._i = self.integ_max
        elif self._i < self.integ_min:
            self._i = self.integ_min

        # --- Derivative term (on error) with optional EMA smoothing ---
        d_raw = (error - self._e_prev) / dt
        self._e_prev = error

        if self.d_window is not None and self.d_window > 1:
            # approximate relationship between window size and EMA alpha
            alpha = 2.0 / (self.d_window + 1.0)
        else:
            alpha = 1.0

        self._d_filtered = alpha * d_raw + (1.0 - alpha) * self._d_filtered

        # --- PID output ---
        u = self.kp * error + self.ki * self._i + self.kd * self._d_filtered

        # --- Output saturation ---
        if u > self.u_max:
            u = self.u_max
        elif u < self.u_min:
            u = self.u_min

        return u

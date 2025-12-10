from dataclasses import dataclass
import numpy as np

@dataclass
class PID:
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0
    u_min: float = -1
    u_max: float = 1
    integ_min: float = -0.025  # Tight limits to keep integral oscillating
    integ_max: float = 0.025   # Not accumulating on one side
    d_window: int = 5
    
    # Anti-windup: stop integrating when saturated
    enable_antiwindup: bool = True

    # Integral auto-reset: reset integral when error is small for a duration
    enable_integral_reset: bool = True
    integral_reset_threshold: float = 2.0  # degrees - reset when error small
    integral_reset_time: float = 1.0  # seconds - reset quickly to keep oscillating

    # Integral reset on direction change: reset when error changes sign significantly
    enable_direction_reset: bool = True
    direction_reset_threshold: float = 1.0  # degrees - reset quickly when oscillating

    _e_prev: float = 0.0
    _i: float = 0.0
    _d_filtered: float = 0.0
    _init: bool = False

    # For derivative filtering - track last few error derivatives
    _d_history: list = None
    _d_history_max: int = 5

    # For integral reset - track how long error has been small
    _low_error_timer: float = 0.0

    def __post_init__(self):
        self._d_filtered = 0.0
        self._d_history = []

    def reset(self):
        self._e_prev = 0.0
        self._i = 0.0
        self._d_filtered = 0.0
        self._init = False
        self._d_history = []
        self._low_error_timer = 0.0

    def reset_integral(self):
        """Reset only the integral term (useful for clearing windup)"""
        self._i = 0.0
        self._low_error_timer = 0.0

    def update(self, error: float, dt: float) -> float:
        if dt <= 0:
            return 0.0

        # === DIRECTION CHANGE RESET ===
        # Reset integral if error changes direction significantly (prevents oscillations)
        if self.enable_direction_reset and self._init:
            error_change_deg = np.rad2deg(abs(error - self._e_prev))
            # If error changed sign AND magnitude changed significantly, reset integral
            if (np.sign(error) != np.sign(self._e_prev) and
                error_change_deg > self.direction_reset_threshold):
                self._i = 0.0

        # === PROPORTIONAL ===
        p_term = self.kp * error

        # === INTEGRAL with anti-windup ===
        # Calculate what output would be without integral
        d_raw = (error - self._e_prev) / dt if self._init else 0.0
        u_no_i = p_term + self.kd * self._d_filtered
        
        # Only integrate if we're not saturated OR if integral would help unsaturate
        if self.enable_antiwindup:
            if self.u_min < u_no_i < self.u_max:
                # Not saturated, integrate normally
                self._i += error * dt
            elif (u_no_i >= self.u_max and error < 0) or (u_no_i <= self.u_min and error > 0):
                # Saturated but error is reducing saturation, allow integration
                self._i += error * dt
            # else: saturated and error would increase saturation - don't integrate
        else:
            # Standard integration
            self._i += error * dt
        
        # Clamp integral
        self._i = np.clip(self._i, self.integ_min, self.integ_max)
        i_term = self.ki * self._i

        # === INTEGRAL AUTO-RESET ===
        # Reset integral if error has been small for a sustained period
        if self.enable_integral_reset:
            # Convert error to degrees for threshold comparison
            error_deg = np.rad2deg(abs(error))

            if error_deg < self.integral_reset_threshold:
                # Error is small, increment timer
                self._low_error_timer += dt

                # If error has been small long enough, reset integral
                if self._low_error_timer >= self.integral_reset_time:
                    self._i = 0.0
                    i_term = 0.0
                    self._low_error_timer = 0.0  # Reset timer after reset
            else:
                # Error is not small, reset timer
                self._low_error_timer = 0.0

        # === DERIVATIVE with improved filtering ===
        if self._init:
            # Use moving average of derivatives to reduce noise spikes
            self._d_history.append(d_raw)
            if len(self._d_history) > self._d_history_max:
                self._d_history.pop(0)
            
            # Median filter to reject outliers, then EMA
            if len(self._d_history) >= 3:
                d_median = np.median(self._d_history)
            else:
                d_median = d_raw
            
            # EMA on median-filtered derivative
            alpha = 2.0 / (self.d_window + 1)  # Standard EMA formula
            self._d_filtered = alpha * d_median + (1.0 - alpha) * self._d_filtered
        else:
            self._d_filtered = 0.0
        
        d_term = self.kd * self._d_filtered
        
        # Update state
        self._e_prev = error
        self._init = True
        
        # === OUTPUT with saturation ===
        u = p_term + i_term + d_term
        u = np.clip(u, self.u_min, self.u_max)
        
        return u
    
    def get_terms(self, error: float) -> tuple:
        """Return individual P, I, D terms for debugging"""
        p = self.kp * error
        i = self.ki * self._i
        d = self.kd * self._d_filtered
        return p, i, d
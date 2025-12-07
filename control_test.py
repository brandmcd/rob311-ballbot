import time
import lcm
import threading
import numpy as np
from collections import deque

from mbot_lcm_msgs.mbot_motor_pwm_t import mbot_motor_pwm_t
from mbot_lcm_msgs.mbot_balbot_feedback_t import mbot_balbot_feedback_t
from DataLogger3 import dataLogger
from ps4_controller_api import PS4InputHandler
from pid import PID

# ===== CONTROL LOOP CONSTANTS =====
FREQ = 100
DT = 1 / FREQ
PWM_MAX = 0.96
PWM_USER_CAP = 0.96

# ===== ROBOT GEOMETRY =====
N_GEARBOX = 70
N_ENC = 64
R_W = 0.048  # Wheel radius [m]
R_K = 0.121  # Ball radius [m]
alpha = 45 * (np.pi / 180.0)

# ===== BALANCE PARAMETERS (base gains) =====
BALANCE_KP = 8.7
BALANCE_KI = 0.13
BALANCE_KD = 0.018

# ===== STEERING PARAMETERS (MUCH WEAKER THAN BALANCE) =====
# Steering gains intentionally smaller + scaled torque
STEER_KP = 7.0
STEER_KI = 0.04
STEER_KD = 0.015

STEER_SCALE = 0.65  # steering torque contribution multiplier

# Steering limits (lean angle caps)
THETA_MAX_DEG = 3.0     # small lean for steering (was 3.0)
THETA_MAX_RAD = np.deg2rad(THETA_MAX_DEG)
TH_STEER_PRIOR_DEG = 3.0
TH_STEER_PRIOR_RAD = np.deg2rad(TH_STEER_PRIOR_DEG)
VEL_CMD_EPS = 1e-3

# Safety
TH_ABORT_DEG = 25.0

# ===== AUTO-TUNING PARAMETERS =====
AUTOTUNING_ENABLED = False  # toggled with Square in BALANCE mode
TUNE_WINDOW = 50           # number of samples for cost window
TUNE_UPDATE_INTERVAL = 100   # iterations between gain updates

# Gradient-ish step sizes (more conservative to avoid wander)
LEARNING_RATE_KP = 0.050
LEARNING_RATE_KD = 0.003
LEARNING_RATE_KI = 0.0005

# ---- Physics-informed cost weights ----
# Big picture:
#  - Going off balance (large |theta|) is terrible.
#  - Going fast along the floor at low angles (high |dphi|) leads to slip.
#  - Drift and huge torques still matter, but less than tilt/low-angle speed.

# Angle penalties
W_ANGLE_CORE    = 1.4    # base lean penalty
W_ANGLE_BARRIER = 35.0   # extra penalty as we approach dangerous tilt

# Angular rate penalty (discourage oscillations)
W_ANGLE_RATE    = 0.8

# Control effort penalty
W_CONTROL       = 0.005

# Drift penalty (rolling away over time)
W_DRIFT         = 6.0

# Low-angle speed penalty (slip risk when nearly upright)
W_LOWANG_SPEED  = 3.0

# Angle thresholds (radians)
TH_WARN_DEG          = 3.0   # start treating as "dangerously tilted"
TH_WARN_RAD          = np.deg2rad(TH_WARN_DEG)

TH_LOWANG_SPEED_DEG  = 2.0   # below this, high surface speed = likely slip
TH_LOWANG_SPEED_RAD  = np.deg2rad(TH_LOWANG_SPEED_DEG)

# ===== GLOBAL STATE =====
listening = False
msg = mbot_balbot_feedback_t()
last_time = 0.0
last_seen = {"MBOT_BALBOT_FEEDBACK": 0.0}
BALANCE_MODE = True


def feedback_handler(channel, data):
    global msg, last_seen, last_time
    last_time = time.time()
    last_seen[channel] = time.time()
    msg = mbot_balbot_feedback_t.decode(data)


def lcm_listener(lc):
    global listening
    while listening:
        try:
            lc.handle_timeout(100)
            if time.time() - last_time > 2.0:
                print("LCM Publisher seems inactive...")
            elif time.time() - last_seen["MBOT_BALBOT_FEEDBACK"] > 2.0:
                print("LCM MBOT_BALBOT_FEEDBACK node seems inactive...")
        except Exception as e:
            print(f"LCM listening error: {e}")
            break


def calc_enc2rad(ticks):
    return (2.0 * np.pi * ticks) / (N_ENC * N_GEARBOX)


def calc_torque_conv(Tx, Ty, Tz):
    """EXACT conversion from your working code."""
    u1 = (1.0 / 3.0) * (Tz - ((2.0 * Ty) / np.cos(alpha)))
    u2 = (1.0 / 3.0) * (Tz + (1.0 / np.cos(alpha)) * (-np.sqrt(3.0) * Tx + Ty))
    u3 = (1.0 / 3.0) * (Tz + (1.0 / np.cos(alpha)) * (np.sqrt(3.0) * Tx + Ty))
    return u1, u2, u3


def calc_kinematic_conv(psi1, psi2, psi3):
    phi_x = (np.sqrt(2.0 / 3.0)) * (R_W / R_K) * (psi2 - psi3)
    phi_y = (np.sqrt(2.0) / 3.0) * (R_W / R_K) * (-2.0 * psi1 + psi2 + psi3)
    phi_z = (np.sqrt(2.0) / 3.0) * (R_W / R_K) * (psi1 + psi2 + psi3)
    return phi_x, phi_y, phi_z


def func_clip(x, lim_lo, lim_hi):
    if x > lim_hi:
        x = lim_hi
    elif x < lim_lo:
        x = lim_lo
    return x


def too_lean(theta_x, theta_y, deg_limit):
    return (abs(theta_x) > np.deg2rad(deg_limit)) or (abs(theta_y) > np.deg2rad(deg_limit))


# ===== AUTO-TUNER =====
class AutoTuner:
    """
    Auto-tunes ONLY the *balance* PIDs (pid_x, pid_y).

    Cost penalizes:
      - Angle error (upright)          -> W_ANGLE_CORE, W_ANGLE_BARRIER
      - Angular velocity (oscillation) -> W_ANGLE_RATE
      - Control effort (huge torques)  -> W_CONTROL
      - Position drift (phi_x, phi_y)  -> W_DRIFT
      - High surface speed at low lean -> W_LOWANG_SPEED
    """

    def __init__(self, window_size=TUNE_WINDOW):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        self.cost_history = deque(maxlen=100)
        self.best_cost = float("inf")
        self.best_gains = {
            "kp": BALANCE_KP,
            "ki": BALANCE_KI,
            "kd": BALANCE_KD,
        }
        self.phi_x_ref = 0.0
        self.phi_y_ref = 0.0

    def reset(self, pid_x, pid_y, phi_x=0.0, phi_y=0.0):
        """Call when enabling autotune or resetting IMU."""
        self.history.clear()
        self.cost_history.clear()
        self.best_cost = float("inf")
        self.best_gains = {
            "kp": pid_x.kp,
            "ki": pid_x.ki,
            "kd": pid_x.kd,
        }
        self.phi_x_ref = phi_x
        self.phi_y_ref = phi_y

    def add_sample(
        self,
        theta_x,
        theta_y,
        theta_dot_x,
        theta_dot_y,
        Tx,
        Ty,
        phi_x,
        phi_y,
        dphi_x,
        dphi_y,
    ):
        self.history.append(
            {
                "theta_x": theta_x,
                "theta_y": theta_y,
                "theta_dot_x": theta_dot_x,
                "theta_dot_y": theta_dot_y,
                "Tx": Tx,
                "Ty": Ty,
                "phi_x": phi_x,
                "phi_y": phi_y,
                "dphi_x": dphi_x,
                "dphi_y": dphi_y,
            }
        )

    def calculate_cost(self):
        if len(self.history) < self.window_size // 2:
            return float("inf")

        cost_angle_core = 0.0
        cost_angle_barrier = 0.0
        cost_rate = 0.0
        cost_control = 0.0
        cost_drift = 0.0
        cost_low_angle_speed = 0.0
        n = len(self.history)

        for s in self.history:
            theta_x = s["theta_x"]
            theta_y = s["theta_y"]
            theta_mag = np.sqrt(theta_x**2 + theta_y**2)

            # 1) Core angle cost: keep lean near zero
            cost_angle_core += theta_mag**2

            # 2) Barrier around warning angle: punish approaching dangerous tilt
            if theta_mag > TH_WARN_RAD:
                ratio = theta_mag / TH_WARN_RAD
                cost_angle_barrier += ratio**4  # grows fast near/above threshold

            # 3) Angular rate (oscillation)
            cost_rate += s["theta_dot_x"]**2 + s["theta_dot_y"]**2

            # 4) Control effort
            cost_control += s["Tx"]**2 + s["Ty"]**2

            # 5) Drift from reference
            dx = s["phi_x"] - self.phi_x_ref
            dy = s["phi_y"] - self.phi_y_ref
            cost_drift += dx**2 + dy**2

            # 6) Low-angle speed (slip risk)
            dphi_x = s["dphi_x"]
            dphi_y = s["dphi_y"]
            surf_speed_sq = dphi_x**2 + dphi_y**2

            if theta_mag < TH_LOWANG_SPEED_RAD:
                # Closer to 0 angle → stronger penalty on surface speed
                weight_factor = 1.0 - (theta_mag / TH_LOWANG_SPEED_RAD)
                weight_factor = max(0.0, min(1.0, weight_factor))
                cost_low_angle_speed += weight_factor * surf_speed_sq

        total_cost = (
            W_ANGLE_CORE * cost_angle_core / n
            + W_ANGLE_BARRIER * cost_angle_barrier / max(n, 1)
            + W_ANGLE_RATE * cost_rate / n
            + W_CONTROL * cost_control / n
            + W_DRIFT * cost_drift / n
            + W_LOWANG_SPEED * cost_low_angle_speed / n
        )

        return total_cost

    def tune_gains(self, pid_x, pid_y):
        """Update gains based on cost trends."""
        current_cost = self.calculate_cost()
        self.cost_history.append(current_cost)

        # Track best
        if current_cost < self.best_cost:
            self.best_cost = current_cost
            self.best_gains = {
                "kp": pid_x.kp,
                "ki": pid_x.ki,
                "kd": pid_x.kd,
            }

        if len(self.cost_history) >= 3:
            r = list(self.cost_history)[-3:]
            # If clearly diverging, snap back to best
            if r[-1] > r[-2] > r[-3]:
                print("[AUTOTUNE] Cost diverging, reverting to best gains")
                pid_x.kp = self.best_gains["kp"]
                pid_x.ki = self.best_gains["ki"]
                pid_x.kd = self.best_gains["kd"]
                pid_y.kp = self.best_gains["kp"]
                pid_y.ki = self.best_gains["ki"]
                pid_y.kd = self.best_gains["kd"]
                return current_cost

            cost_trend = (r[-1] - r[-3]) / 2.0

            if cost_trend > 0:
                # Cost increasing → see if it's oscillatory or just large angle
                if len(self.history) >= 20:
                    last = list(self.history)[-20:]
                    rate_cost = np.mean(
                        [x["theta_dot_x"]**2 + x["theta_dot_y"]**2 for x in last]
                    )
                    angle_cost = np.mean(
                        [np.sqrt(x["theta_x"]**2 + x["theta_y"]**2)**2 for x in last]
                    )

                    if rate_cost > angle_cost:
                        # Oscillation: slightly reduce Kp, bump Kd
                        pid_x.kp *= 0.985
                        pid_y.kp *= 0.985
                        pid_x.kd += LEARNING_RATE_KD
                        pid_y.kd += LEARNING_RATE_KD
                    else:
                        # Sluggish: increase Kp a bit
                        pid_x.kp += LEARNING_RATE_KP
                        pid_y.kp += LEARNING_RATE_KP
            else:
                # Improving → gentle co-tuning of Kp, Kd
                if abs(cost_trend) > 0.01:
                    pid_x.kp += 0.5 * LEARNING_RATE_KP
                    pid_y.kp += 0.5 * LEARNING_RATE_KP
                    pid_x.kd += 0.5 * LEARNING_RATE_KD
                    pid_y.kd += 0.5 * LEARNING_RATE_KD

        # Clamp to sane ranges
        pid_x.kp = func_clip(pid_x.kp, 1.0, 20.0)
        pid_y.kp = func_clip(pid_y.kp, 1.0, 20.0)
        pid_x.kd = func_clip(pid_x.kd, 0.0, 0.5)
        pid_y.kd = func_clip(pid_y.kd, 0.0, 0.5)
        pid_x.ki = func_clip(pid_x.ki, 0.0, 1.0)
        pid_y.ki = func_clip(pid_y.ki, 0.0, 1.0)

        print(
            f"[AUTOTUNE] Cost={current_cost:.4f} best={self.best_cost:.4f} | "
            f"Kp={pid_x.kp:.3f}, Ki={pid_x.ki:.4f}, Kd={pid_x.kd:.4f}"
        )
        return current_cost


def main():
    global listening, msg, BALANCE_MODE, AUTOTUNING_ENABLED

    # === Data Logging ===
    trial_num = int(input("Test Number? "))
    filename = f"main_control_autotune_{trial_num}.txt"
    dl = dataLogger(filename)

    # === LCM Setup ===
    lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=0")
    lc.subscribe("MBOT_BALBOT_FEEDBACK", feedback_handler)
    listening = True
    listener_thread = threading.Thread(target=lcm_listener, args=(lc,), daemon=True)
    listener_thread.start()
    print("Started LCM listener...")

    # === Controller Setup ===
    controller = PS4InputHandler(interface="/dev/input/js0", connecting_using_ds4drv=False)
    controller_thread = threading.Thread(target=controller.listen, args=(10,))
    controller_thread.daemon = True
    controller_thread.start()
    print("PS4 Controller active...")
    print("Controls:")
    print("  Triangle = Emergency Stop")
    print("  Circle   = Reset IMU & PIDs")
    print("  X        = Toggle Balance/Steering Mode")
    print("  Square   = Toggle Auto-tune (BALANCE mode only)")
    print("  Right Stick = World-frame steering (when not in balance mode)")
    print("  D-Pad    = Tune PID gains (disabled while autotuning)")

    # === PID Controllers ===
    pid_x = PID(
        kp=BALANCE_KP,
        ki=BALANCE_KI,
        kd=BALANCE_KD,
        u_min=-PWM_USER_CAP,
        u_max=PWM_USER_CAP,
        integ_min=-0.25,
        integ_max=0.25,
        d_window=5,
    )
    pid_y = PID(
        kp=BALANCE_KP,
        ki=BALANCE_KI,
        kd=BALANCE_KD,
        u_min=-PWM_USER_CAP,
        u_max=PWM_USER_CAP,
        integ_min=-0.25,
        integ_max=0.25,
        d_window=5,
    )
    pid_x_steer = PID(
        kp=STEER_KP,
        ki=STEER_KI,
        kd=STEER_KD,
        u_min=-PWM_USER_CAP,
        u_max=PWM_USER_CAP,
        integ_min=-0.25,
        integ_max=0.25,
        d_window=5,
    )
    pid_y_steer = PID(
        kp=STEER_KP,
        ki=STEER_KI,
        kd=STEER_KD,
        u_min=-PWM_USER_CAP,
        u_max=PWM_USER_CAP,
        integ_min=-0.25,
        integ_max=0.25,
        d_window=5,
    )

    tuner = AutoTuner()

    try:
        command = mbot_motor_pwm_t()
        print("Starting control loop...")
        time.sleep(0.5)

        # Logging header
        header = [
            "i",
            "t_now",
            "Tx",
            "Ty",
            "Tz",
            "u1",
            "u2",
            "u3",
            "theta_x",
            "theta_y",
            "theta_z",
            "psi_1",
            "psi_2",
            "psi_3",
            "dpsi_1",
            "dpsi_2",
            "dpsi_3",
            "e_theta_x",
            "e_theta_y",
            "abort",
            "kp_x",
            "ki_x",
            "kd_x",
            "kp_y",
            "ki_y",
            "kd_y",
            "cost",
            "autotune",
        ]
        dl.appendData(header)

        i = 0
        t_start = time.time()
        prev_t = time.time()

        # Wait for first message
        time.sleep(0.1)

        # Zero encoders and IMU
        enc_pos_1_start = msg.enc_ticks[0]
        enc_pos_2_start = msg.enc_ticks[1]
        enc_pos_3_start = msg.enc_ticks[2]
        theta_x_0 = msg.imu_angles_rpy[0]
        theta_y_0 = msg.imu_angles_rpy[1]
        theta_z_0 = msg.imu_angles_rpy[2]

        print(
            f"IMU offsets: theta_x_0={theta_x_0:.4f}, "
            f"theta_y_0={theta_y_0:.4f}, theta_z_0={theta_z_0:.4f}"
        )

        # Button states
        prev_tri = prev_cir = prev_x = prev_sqr = 0
        prev_dpad_h = prev_dpad_vert = 0

        # D-pad tuning
        step = 0.1
        cur_tune = 0  # 0=Kp, 1=Kd, 2=Ki

        # For angular velocity estimate
        prev_theta_x = 0.0
        prev_theta_y = 0.0

        last_cost = float("inf")

        print("\n=== Starting in BALANCE mode ===")

        while True:
            time.sleep(DT)
            t_now = time.time() - t_start
            i += 1

            try:
                # =============================
                # GAMEPAD INPUT
                # =============================
                bt_signals = controller.get_signals()
                js_R_x = bt_signals["js_R_x"]
                js_R_y = bt_signals["js_R_y"]

                dpad_h = bt_signals["dpad_horiz"]
                dpad_vert = bt_signals["dpad_vert"]

                # Select which gain to tune (only when not autotuning)
                if not AUTOTUNING_ENABLED:
                    if dpad_h > prev_dpad_h:
                        cur_tune = (cur_tune + 1) % 3
                    elif dpad_h < prev_dpad_h:
                        cur_tune = (cur_tune - 1) % 3

                    if BALANCE_MODE:
                        # Tune BALANCE PIDs
                        if dpad_vert > prev_dpad_vert:
                            if cur_tune == 0:
                                pid_x.kp += step
                                pid_y.kp += step
                            elif cur_tune == 1:
                                pid_x.kd += step / 10.0
                                pid_y.kd += step / 10.0
                            else:
                                pid_x.ki += step / 10.0
                                pid_y.ki += step / 10.0
                        elif dpad_vert < prev_dpad_vert:
                            if cur_tune == 0:
                                pid_x.kp -= step
                                pid_y.kp -= step
                            elif cur_tune == 1:
                                pid_x.kd -= step / 10.0
                                pid_y.kd -= step / 10.0
                            else:
                                pid_x.ki -= step / 10.0
                                pid_y.ki -= step / 10.0
                    else:
                        # Tune STEERING PIDs
                        if dpad_vert > prev_dpad_vert:
                            if cur_tune == 0:
                                pid_x_steer.kp += step
                                pid_y_steer.kp += step
                            elif cur_tune == 1:
                                pid_x_steer.kd += step / 10.0
                                pid_y_steer.kd += step / 10.0
                            else:
                                pid_x_steer.ki += step / 10.0
                                pid_y_steer.ki += step / 10.0
                        elif dpad_vert < prev_dpad_vert:
                            if cur_tune == 0:
                                pid_x_steer.kp -= step
                                pid_y_steer.kp -= step
                            elif cur_tune == 1:
                                pid_x_steer.kd -= step / 10.0
                                pid_y_steer.kd -= step / 10.0
                            else:
                                pid_x_steer.ki -= step / 10.0
                                pid_y_steer.ki -= step / 10.0

                prev_dpad_h = dpad_h
                prev_dpad_vert = dpad_vert

                # Buttons
                tri = bt_signals.get("but_tri", 0)
                cir = bt_signals.get("but_cir", 0)
                but_x = bt_signals.get("but_x", 0)
                sqr = bt_signals.get("but_sq", 0)

                # Triangle: Emergency stop
                if tri and not prev_tri:
                    print("PS4 KILL (Triangle) pressed - stopping motors and exiting.")
                    command.utime = int(time.time() * 1e6)
                    command.pwm[:] = [0.0, 0.0, 0.0]
                    lc.publish("MBOT_MOTOR_PWM_CMD", command.encode())
                    break

                # Circle: Reset IMU & PIDs (also reset tuner reference)
                if cir and not prev_cir:
                    theta_x_0 = msg.imu_angles_rpy[0]
                    theta_y_0 = msg.imu_angles_rpy[1]
                    theta_z_0 = msg.imu_angles_rpy[2]
                    pid_x.reset()
                    pid_y.reset()
                    pid_x_steer.reset()
                    pid_y_steer.reset()

                    # Recompute current phi for drift reference
                    enc_pos_1_tmp = msg.enc_ticks[0] - enc_pos_1_start
                    enc_pos_2_tmp = msg.enc_ticks[1] - enc_pos_2_start
                    enc_pos_3_tmp = msg.enc_ticks[2] - enc_pos_3_start
                    psi_1_tmp = calc_enc2rad(enc_pos_1_tmp)
                    psi_2_tmp = calc_enc2rad(enc_pos_2_tmp)
                    psi_3_tmp = calc_enc2rad(enc_pos_3_tmp)
                    phi_x_tmp, phi_y_tmp, _ = calc_kinematic_conv(
                        psi_1_tmp, psi_2_tmp, psi_3_tmp
                    )
                    tuner.reset(pid_x, pid_y, phi_x_tmp, phi_y_tmp)

                    print(
                        f"IMU reset (Circle): theta_x_0={theta_x_0:.4f}, "
                        f"theta_y_0={theta_y_0:.4f}, theta_z_0={theta_z_0:.4f}"
                    )

                # X: Toggle BALANCE / STEERING
                if but_x and not prev_x:
                    BALANCE_MODE = not BALANCE_MODE
                    mode_str = "BALANCE" if BALANCE_MODE else "STEERING"
                    print(f"Mode toggled (X): {mode_str}")
                    if not BALANCE_MODE and AUTOTUNING_ENABLED:
                        AUTOTUNING_ENABLED = False
                        print("[AUTOTUNE] Disabled (STEERING mode)")

                # Square: Toggle autotuning (BALANCE mode only)
                if sqr and not prev_sqr:
                    if BALANCE_MODE:
                        AUTOTUNING_ENABLED = not AUTOTUNING_ENABLED
                        if AUTOTUNING_ENABLED:
                            # Reset tuner around current position
                            enc_pos_1_tmp = msg.enc_ticks[0] - enc_pos_1_start
                            enc_pos_2_tmp = msg.enc_ticks[1] - enc_pos_2_start
                            enc_pos_3_tmp = msg.enc_ticks[2] - enc_pos_3_start
                            psi_1_tmp = calc_enc2rad(enc_pos_1_tmp)
                            psi_2_tmp = calc_enc2rad(enc_pos_2_tmp)
                            psi_3_tmp = calc_enc2rad(enc_pos_3_tmp)
                            phi_x_tmp, phi_y_tmp, _ = calc_kinematic_conv(
                                psi_1_tmp, psi_2_tmp, psi_3_tmp
                            )
                            tuner.reset(pid_x, pid_y, phi_x_tmp, phi_y_tmp)
                            print("[AUTOTUNE] ENABLED in BALANCE mode")
                        else:
                            print("[AUTOTUNE] DISABLED")
                    else:
                        print("Auto-tuning only available in BALANCE mode")

                prev_tri = tri
                prev_cir = cir
                prev_x = but_x
                prev_sqr = sqr

                # =============================
                # SENSORS
                # =============================
                theta_x = msg.imu_angles_rpy[0] - theta_x_0
                theta_y = msg.imu_angles_rpy[1] - theta_y_0
                theta_z = msg.imu_angles_rpy[2] - theta_z_0  # yaw

                enc_pos_1 = msg.enc_ticks[0] - enc_pos_1_start
                enc_pos_2 = msg.enc_ticks[1] - enc_pos_2_start
                enc_pos_3 = msg.enc_ticks[2] - enc_pos_3_start
                enc_dtick_1 = msg.enc_delta_ticks[0]
                enc_dtick_2 = msg.enc_delta_ticks[1]
                enc_dtick_3 = msg.enc_delta_ticks[2]
                enc_dt = msg.enc_delta_time

                psi_1 = calc_enc2rad(enc_pos_1)
                psi_2 = calc_enc2rad(enc_pos_2)
                psi_3 = calc_enc2rad(enc_pos_3)
                enc_dt = max(enc_dt, 1e-6)
                dpsi_1 = calc_enc2rad(enc_dtick_1) / (enc_dt * 1e-6)
                dpsi_2 = calc_enc2rad(enc_dtick_2) / (enc_dt * 1e-6)
                dpsi_3 = calc_enc2rad(enc_dtick_3) / (enc_dt * 1e-6)
                phi_x, phi_y, phi_z = calc_kinematic_conv(psi_1, psi_2, psi_3)
                dphi_x, dphi_y, dphi_z = calc_kinematic_conv(dpsi_1, dpsi_2, dpsi_3)

                abort = too_lean(theta_x, theta_y, TH_ABORT_DEG)

                now_t = time.time()
                dt = max(1e-6, now_t - prev_t)
                prev_t = now_t

                # angular velocities for cost
                theta_dot_x = (theta_x - prev_theta_x) / dt
                theta_dot_y = (theta_y - prev_theta_y) / dt
                prev_theta_x = theta_x
                prev_theta_y = theta_y

                # =============================
                # CONTROL LOGIC (direct lean steering, no yaw mapping)
                # =============================
                DEADZONE = 0.08
                js_fwd = -js_R_y if abs(js_R_y) > DEADZONE else 0.0
                js_strafe = js_R_x if abs(js_R_x) > DEADZONE else 0.0

                # Nonlinear stick shaping
                js_fwd = np.sign(js_fwd) * (js_fwd**2)
                js_strafe = np.sign(js_strafe) * (js_strafe**2)

                theta_d_x = 0.0
                theta_d_y = 0.0
                if not BALANCE_MODE and (abs(js_fwd) > VEL_CMD_EPS or abs(js_strafe) > VEL_CMD_EPS):
                    theta_d_y = -THETA_MAX_RAD * js_fwd  # pitch for forward/back
                    theta_d_x =  THETA_MAX_RAD * js_strafe  # roll for strafe

                e_bal_x = -theta_x
                e_bal_y = -theta_y
                e_str_x = theta_d_x - theta_x
                e_str_y = theta_d_y - theta_y

                Tx_balance = Ty_balance = 0.0
                Tx_steer = Ty_steer = 0.0
                Tx = Ty = Tz = 0.0

                if not abort:
                    # Always compute balance torques
                    Tx_balance = pid_y.update(e_bal_y, dt)  # pitch -> Tx
                    Ty_balance = pid_x.update(e_bal_x, dt)  # roll  -> Ty

                    # Feed to autotuner (BALANCE + autotuning on)
                    if BALANCE_MODE and AUTOTUNING_ENABLED:
                        tuner.add_sample(
                            theta_x,
                            theta_y,
                            theta_dot_x,
                            theta_dot_y,
                            Tx_balance,
                            Ty_balance,
                            phi_x,
                            phi_y,
                            dphi_x,
                            dphi_y,
                        )
                        if i % TUNE_UPDATE_INTERVAL == 0:
                            last_cost = tuner.tune_gains(pid_x, pid_y)

                    if BALANCE_MODE:
                        pid_x_steer.reset()
                        pid_y_steer.reset()
                        Tx = func_clip(Tx_balance, -PWM_USER_CAP, PWM_USER_CAP)
                        Ty = func_clip(Ty_balance, -PWM_USER_CAP, PWM_USER_CAP)
                        Tz = 0.0
                    else:
                        # Add steering only if commanded and not already leaning hard
                        if (
                            (abs(js_fwd) > VEL_CMD_EPS or abs(js_strafe) > VEL_CMD_EPS)
                            and (abs(theta_x) < TH_STEER_PRIOR_RAD and abs(theta_y) < TH_STEER_PRIOR_RAD)
                        ):
                            Tx_steer = pid_y_steer.update(e_str_y, dt)
                            Ty_steer = pid_x_steer.update(e_str_x, dt)
                        else:
                            pid_x_steer.reset()
                            pid_y_steer.reset()
                            Tx_steer = 0.0
                            Ty_steer = 0.0

                        Tx = func_clip(
                            Tx_balance + STEER_SCALE * Tx_steer,
                            -PWM_USER_CAP,
                            PWM_USER_CAP,
                        )
                        Ty = func_clip(
                            Ty_balance + STEER_SCALE * Ty_steer,
                            -PWM_USER_CAP,
                            PWM_USER_CAP,
                        )
                        Tz = 0.0
                else:
                    # Emergency stop on large lean
                    Tx = Ty = Tz = 0.0
                    pid_x.reset()
                    pid_y.reset()
                    pid_x_steer.reset()
                    pid_y_steer.reset()
                    AUTOTUNING_ENABLED = False
                    print(
                        f"ABORT: Lean angle too high! "
                        f"theta_x={np.rad2deg(theta_x):.1f}°, "
                        f"theta_y={np.rad2deg(theta_y):.1f}°"
                    )

                # =============================
                # MOTOR COMMANDS (directions preserved)
                # =============================
                u3, u1, u2 = calc_torque_conv(Tx, Ty, Tz)
                u1 = -func_clip(u1, -PWM_MAX, PWM_MAX)
                u2 = -func_clip(u2, -PWM_MAX, PWM_MAX)
                u3 = -func_clip(u3, -PWM_MAX, PWM_MAX)

                command.utime = int(time.time() * 1e6)
                command.pwm[0] = u1
                command.pwm[1] = u2
                command.pwm[2] = u3
                lc.publish("MBOT_MOTOR_PWM_CMD", command.encode())

                # =============================
                # LOGGING
                # =============================
                data = [
                    i,
                    t_now,
                    float(Tx),
                    float(Ty),
                    float(Tz),
                    float(u1),
                    float(u2),
                    float(u3),
                    float(np.rad2deg(theta_x)),
                    float(np.rad2deg(theta_y)),
                    float(np.rad2deg(theta_z)),
                    float(psi_1),
                    float(psi_2),
                    float(psi_3),
                    float(dpsi_1),
                    float(dpsi_2),
                    float(dpsi_3),
                    float(np.rad2deg(e_str_x)),
                    float(np.rad2deg(e_str_y)),
                    float(abort),
                    float(pid_x.kp),
                    float(pid_x.ki),
                    float(pid_x.kd),
                    float(pid_y.kp),
                    float(pid_y.ki),
                    float(pid_y.kd),
                    float(last_cost),
                    float(AUTOTUNING_ENABLED),
                ]
                dl.appendData(data)

                # =============================
                # CONSOLE OUTPUT
                # =============================
                if BALANCE_MODE:
                    abort_str = " [ABORT]" if abort else ""
                    tune_names = ["Kp", "Kd", "Ki"]
                    cur_name = tune_names[cur_tune]
                    at_str = "ON" if AUTOTUNING_ENABLED else "OFF"
                    print(
                        f"[BALANCE] t={t_now:6.3f}s{abort_str} | "
                        f"θx={np.rad2deg(theta_x):+6.2f}°, θy={np.rad2deg(theta_y):+6.2f}° | "
                        f"Kx(Kp,Ki,Kd)=({pid_x.kp:.3f},{pid_x.ki:.4f},{pid_x.kd:.4f}) | "
                        f"Ky(Kp,Ki,Kd)=({pid_y.kp:.3f},{pid_y.ki:.4f},{pid_y.kd:.4f}) | "
                        f"Tx={Tx:+6.3f}, Ty={Ty:+6.3f} | "
                        f"u=[{u1:+5.3f},{u2:+5.3f},{u3:+5.3f}] | "
                        f"AUTOTUNE={at_str}, cost={last_cost:8.4f}"
                    )
                    if not AUTOTUNING_ENABLED:
                        print(f"    D-Pad tuning BALANCE {cur_name}")
                else:
                    abort_str = " [ABORT]" if abort else ""
                    print(
                        f"[STEERING] t={t_now:6.3f}s{abort_str} | "
                        f"θx={np.rad2deg(theta_x):+6.2f}° (d={np.rad2deg(theta_d_x):+6.2f}°) | "
                        f"θy={np.rad2deg(theta_y):+6.2f}° (d={np.rad2deg(theta_d_y):+6.2f}°) | "
                        f"cmd_fwd={js_fwd:+.2f}, cmd_strf={js_strafe:+.2f} | "
                        f"Tx={Tx:+6.3f}, Ty={Ty:+6.3f} | "
                        f"Kx(Kp,Ki,Kd)=({pid_x.kp:.3f},{pid_x.ki:.4f},{pid_x.kd:.4f}) | "
                        f"Ky(Kp,Ki,Kd)=({pid_y.kp:.3f},{pid_y.ki:.4f},{pid_y.kd:.4f})"
                    )
                    if not AUTOTUNING_ENABLED:
                        tune_names = ["Kp", "Kd", "Ki"]
                        cur_name = tune_names[cur_tune]
                        print(f"    D-Pad tuning STEERING {cur_name}")

            except KeyError:
                print("Waiting for sensor data...")

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Stopping motors...")
        command = mbot_motor_pwm_t()
        command.utime = int(time.time() * 1e6)
        command.pwm[:] = [0.0, 0.0, 0.0]
        lc.publish("MBOT_MOTOR_PWM_CMD", command.encode())

    finally:
        print(f"Saving data as {filename}...")
        dl.writeOut()
        listening = False
        print("Stopping LCM listener...")
        listener_thread.join(timeout=1)
        controller_thread.join(timeout=1)
        controller.on_options_press()
        print("Shutting down motors...")
        command = mbot_motor_pwm_t()
        command.utime = int(time.time() * 1e6)
        command.pwm[:] = [0.0, 0.0, 0.0]
        lc.publish("MBOT_MOTOR_PWM_CMD", command.encode())


if __name__ == "__main__":
    main()

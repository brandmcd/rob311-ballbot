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
DT = 1.0 / FREQ
PWM_MAX = 0.96
PWM_USER_CAP = 0.96

# ===== ROBOT GEOMETRY =====
N_GEARBOX = 70
N_ENC = 64
R_W = 0.048  # Wheel radius [m]
R_K = 0.121  # Ball radius [m]
alpha = 45.0 * (np.pi / 180.0)

# ===== BALANCE PARAMETERS =====
BALANCE_KP = 8.6
BALANCE_KI = 0.17
BALANCE_KD = 0.02

# ===== STEERING PARAMETERS (WEAK, ADDED ON TOP OF BALANCE) =====
STEER_KP = 4.0
STEER_KI = 0.03
STEER_KD = 0.0
STEER_WEIGHT = 0.2          # steering torque scaled vs balance

# Steering limits
THETA_MAX_DEG = 2.0         # small fixed lean for steering
THETA_MAX_RAD = np.deg2rad(THETA_MAX_DEG)
TH_STEER_PRIOR_DEG = 3.0
TH_STEER_PRIOR_RAD = np.deg2rad(TH_STEER_PRIOR_DEG)

# Safety
TH_ABORT_DEG = 25.0

# ===== AUTOTUNE PARAMETERS (BALANCE ONLY) =====
AUTOTUNING_ENABLED = False      # toggled with Square (in BALANCE mode)
TUNE_WINDOW = 200
TUNE_UPDATE_INTERVAL = 50

LEARNING_RATE_KP = 0.075
LEARNING_RATE_KD = 0.0015
LEARNING_RATE_KI = 0.0005

# Cost weights (physics-aware)
W_ANGLE_CORE    = 6.0
W_ANGLE_BARRIER = 25.0
W_ANGLE_RATE    = 0.8
W_CONTROL       = 0.006
W_DRIFT         = 1.75
W_LOWANG_SPEED  = 4.0

TH_WARN_RAD         = np.deg2rad(8.0)   # "dangerous" tilt
TH_LOWANG_SPEED_RAD = np.deg2rad(2.0)   # near-upright zone → penalize speed

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


# ===== AUTOTUNER (BALANCE ONLY) =====
class AutoTuner:
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

            # Upright cost
            cost_angle_core += theta_mag**2

            # Barrier near dangerous tilt
            if theta_mag > TH_WARN_RAD:
                ratio = theta_mag / TH_WARN_RAD
                cost_angle_barrier += ratio**4

            # Angular rate cost
            cost_rate += s["theta_dot_x"]**2 + s["theta_dot_y"]**2

            # Torque cost
            cost_control += s["Tx"]**2 + s["Ty"]**2

            # Drift cost
            dx = s["phi_x"] - self.phi_x_ref
            dy = s["phi_y"] - self.phi_y_ref
            cost_drift += dx**2 + dy**2

            # Low-angle speed cost (slip risk)
            dphi_x = s["dphi_x"]
            dphi_y = s["dphi_y"]
            surf_speed_sq = dphi_x**2 + dphi_y**2
            if theta_mag < TH_LOWANG_SPEED_RAD:
                w = 1.0 - (theta_mag / TH_LOWANG_SPEED_RAD)
                w = max(0.0, min(1.0, w))
                cost_low_angle_speed += w * surf_speed_sq

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
        current_cost = self.calculate_cost()
        self.cost_history.append(current_cost)

        if current_cost < self.best_cost:
            self.best_cost = current_cost
            self.best_gains = {
                "kp": pid_x.kp,
                "ki": pid_x.ki,
                "kd": pid_x.kd,
            }

        if len(self.cost_history) >= 3:
            r = list(self.cost_history)[-3:]
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
                # getting worse
                if len(self.history) >= 20:
                    last = list(self.history)[-20:]
                    rate_cost = np.mean(
                        [x["theta_dot_x"]**2 + x["theta_dot_y"]**2 for x in last]
                    )
                    angle_cost = np.mean(
                        [np.sqrt(x["theta_x"]**2 + x["theta_y"]**2)**2 for x in last]
                    )
                    if rate_cost > angle_cost:
                        # oscillating -> reduce Kp, bump Kd
                        pid_x.kp *= 0.985
                        pid_y.kp *= 0.985
                        pid_x.kd += LEARNING_RATE_KD
                        pid_y.kd += LEARNING_RATE_KD
                    else:
                        # sluggish -> increase Kp
                        pid_x.kp += LEARNING_RATE_KP
                        pid_y.kp += LEARNING_RATE_KP
            else:
                # improving -> gentle co-tune
                if abs(cost_trend) > 0.01:
                    pid_x.kp += 0.5 * LEARNING_RATE_KP
                    pid_y.kp += 0.5 * LEARNING_RATE_KP
                    pid_x.kd += 0.5 * LEARNING_RATE_KD
                    pid_y.kd += 0.5 * LEARNING_RATE_KD

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

    trial_num = int(input("Test Number? "))
    filename = f"main_control_autotune_dpad_{trial_num}.txt"
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
    print("  Circle   = Reset IMU & PIDs (and autotune ref)")
    print("  X        = Toggle Balance / Steering mode")
    print("  Square   = Toggle Auto-tune (BALANCE mode only)")
    print("  D-Pad    = Tune BALANCE gains (balance mode, autotune OFF)")
    print("  D-Pad DOWN (STEERING mode) = drive forward (fixed lean)")

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

        header = [
            "i", "t_now",
            "Tx", "Ty", "Tz",
            "u1", "u2", "u3",
            "theta_x", "theta_y", "theta_z",
            "psi_1", "psi_2", "psi_3",
            "dpsi_1", "dpsi_2", "dpsi_3",
            "e_x", "e_y",
            "abort",
            "kp_x", "ki_x", "kd_x",
            "kp_y", "ki_y", "kd_y",
            "cost", "autotune", "steer_active",
        ]
        dl.appendData(header)

        i = 0
        t_start = time.time()
        prev_t = time.time()

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

        prev_tri = prev_cir = prev_x = prev_sqr = 0
        prev_dpad_h = prev_dpad_vert = 0

        step = 0.1
        cur_tune = 0  # 0=Kp, 1=Kd, 2=Ki

        prev_theta_x = 0.0
        prev_theta_y = 0.0
        last_cost = float("inf")

        print("\n=== Starting in BALANCE mode ===")

        while True:
            time.sleep(DT)
            t_now = time.time() - t_start
            i += 1

            try:
                bt_signals = controller.get_signals()
                dpad_h = bt_signals["dpad_horiz"]
                dpad_vert = bt_signals["dpad_vert"]   # down is usually -1, up is +1

                # ---- D-Pad for tuning ONLY in BALANCE mode and AUTOTUNE OFF ----
                if BALANCE_MODE and not AUTOTUNING_ENABLED:
                    if dpad_h > prev_dpad_h:
                        cur_tune = (cur_tune + 1) % 3
                    elif dpad_h < prev_dpad_h:
                        cur_tune = (cur_tune - 1) % 3

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

                prev_dpad_h = dpad_h
                prev_dpad_vert = dpad_vert

                # Buttons
                tri = bt_signals.get("but_tri", 0)
                cir = bt_signals.get("but_cir", 0)
                but_x = bt_signals.get("but_x", 0)
                sqr = bt_signals.get("but_sq", 0)

                # Emergency stop
                if tri and not prev_tri:
                    print("PS4 KILL (Triangle) pressed - stopping motors and exiting.")
                    command.utime = int(time.time() * 1e6)
                    command.pwm[:] = [0.0, 0.0, 0.0]
                    lc.publish("MBOT_MOTOR_PWM_CMD", command.encode())
                    break

                # Reset IMU + PIDs + autotune reference
                if cir and not prev_cir:
                    theta_x_0 = msg.imu_angles_rpy[0]
                    theta_y_0 = msg.imu_angles_rpy[1]
                    theta_z_0 = msg.imu_angles_rpy[2]
                    pid_x.reset()
                    pid_y.reset()
                    pid_x_steer.reset()
                    pid_y_steer.reset()

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

                # Toggle mode
                if but_x and not prev_x:
                    BALANCE_MODE = not BALANCE_MODE
                    mode_str = "BALANCE" if BALANCE_MODE else "STEERING"
                    print(f"Mode toggled (X): {mode_str}")
                    if not BALANCE_MODE and AUTOTUNING_ENABLED:
                        AUTOTUNING_ENABLED = False
                        print("[AUTOTUNE] Disabled (STEERING mode)")

                # Toggle autotune (BALANCE only)
                if sqr and not prev_sqr:
                    if BALANCE_MODE:
                        AUTOTUNING_ENABLED = not AUTOTUNING_ENABLED
                        if AUTOTUNING_ENABLED:
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

                # SENSORS
                theta_x = msg.imu_angles_rpy[0] - theta_x_0
                theta_y = msg.imu_angles_rpy[1] - theta_y_0
                theta_z = msg.imu_angles_rpy[2] - theta_z_0

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

                theta_dot_x = (theta_x - prev_theta_x) / dt
                theta_dot_y = (theta_y - prev_theta_y) / dt
                prev_theta_x = theta_x
                prev_theta_y = theta_y

                # ===== CONTROL LOGIC =====
                theta_d_x = 0.0
                theta_d_y = 0.0
                steer_active = False

                # In STEERING mode: D-Pad down -> fixed forward lean, else 0 (pure balance)
                if not BALANCE_MODE:
                    if dpad_vert < 0:  # down pressed
                        theta_d_y = -THETA_MAX_RAD
                        theta_d_x = 0.0
                        steer_active = True
                    else:
                        theta_d_y = 0.0
                        theta_d_x = 0.0
                        steer_active = False

                e_x = theta_d_x - theta_x
                e_y = theta_d_y - theta_y
                e_bal_x = -theta_x
                e_bal_y = -theta_y
                e_str_x = e_x
                e_str_y = e_y

                Tx_balance = Ty_balance = 0.0
                Tx_steer = Ty_steer = 0.0
                Tx = Ty = Tz = 0.0

                if not abort:
                    Tx_balance = pid_y.update(e_bal_y, dt)  # pitch -> Tx
                    Ty_balance = pid_x.update(e_bal_x, dt)  # roll  -> Ty

                    # Autotune (BALANCE only)
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
                        # STEERING MODE: only add steering if D-Pad down is held
                        if steer_active and (
                            abs(theta_x) < TH_STEER_PRIOR_RAD
                            and abs(theta_y) < TH_STEER_PRIOR_RAD
                        ):
                            # Only forward steering (Ty_steer unused here)
                            Tx_steer = pid_y_steer.update(e_str_y, dt)
                            Ty_steer = 0.0
                        else:
                            pid_x_steer.reset()
                            pid_y_steer.reset()
                            Tx_steer = 0.0
                            Ty_steer = 0.0

                        Tx = func_clip(
                            Tx_balance + STEER_WEIGHT * Tx_steer,
                            -PWM_USER_CAP,
                            PWM_USER_CAP,
                        )
                        Ty = func_clip(
                            Ty_balance + STEER_WEIGHT * Ty_steer,
                            -PWM_USER_CAP,
                            PWM_USER_CAP,
                        )
                        Tz = 0.0
                else:
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

                # ===== MOTOR COMMANDS =====
                u3, u1, u2 = calc_torque_conv(Tx, Ty, Tz)
                u1 = -func_clip(u1, -PWM_MAX, PWM_MAX)
                u2 = -func_clip(u2, -PWM_MAX, PWM_MAX)
                u3 = -func_clip(u3, -PWM_MAX, PWM_MAX)

                command.utime = int(time.time() * 1e6)
                command.pwm[0] = u1
                command.pwm[1] = u2
                command.pwm[2] = u3
                lc.publish("MBOT_MOTOR_PWM_CMD", command.encode())

                # ===== LOGGING =====
                data = [
                    i, t_now,
                    float(Tx), float(Ty), float(Tz),
                    float(u1), float(u2), float(u3),
                    float(np.rad2deg(theta_x)),
                    float(np.rad2deg(theta_y)),
                    float(np.rad2deg(theta_z)),
                    float(psi_1), float(psi_2), float(psi_3),
                    float(dpsi_1), float(dpsi_2), float(dpsi_3),
                    float(np.rad2deg(e_x)),
                    float(np.rad2deg(e_y)),
                    float(abort),
                    float(pid_x.kp), float(pid_x.ki), float(pid_x.kd),
                    float(pid_y.kp), float(pid_y.ki), float(pid_y.kd),
                    float(last_cost),
                    float(AUTOTUNING_ENABLED),
                    float(steer_active),
                ]
                dl.appendData(data)

                # ===== PRINT =====
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
                        f"D-Pad DOWN active={steer_active} | "
                        f"Tx={Tx:+6.3f}, Ty={Ty:+6.3f}"
                    )
                    print("    (Steer PID not tunable in this mode; D-Pad only drives)")

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

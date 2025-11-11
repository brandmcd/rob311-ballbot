import time
import threading
import lcm
import numpy as np

from mbot_lcm_msgs.mbot_motor_pwm_t import mbot_motor_pwm_t
from mbot_lcm_msgs.mbot_balbot_feedback_t import mbot_balbot_feedback_t
from DataLogger import dataLogger

# --- Control loop constants -------------------------------------------------
FREQ = 200                  # Hz
DT = 1.0 / FREQ             # s per step
PWM_MAX = 1.0               # absolute motor command bound

# --- Controller tuning parameters ------------------------------------------
PWM_USER_CAP = 0.50         # actuator saturation for tuning (±50%)
SOFTSTART_SEC = 2.0         # seconds to ramp soft-start
DEADBAND_TH_DEG = 0.25      # deadband on small angle errors (degrees)
TH_ABORT_DEG = 18.0         # hard abort if tilt exceeds this (degrees)

# --- Robot geometry --------------------------------------------------------
N_GEARBOX = 70              # gearbox ratio
N_ENC = 64                  # encoder ticks per motor rev
R_W = 0.048                 # omni-wheel radius [m]
R_K = 0.121                 # ball radius [m]
alpha = 45 * (np.pi / 180)  # omni-wheel angle from vertical [rad]

# --- PID initial gains (tuning) -------------------------------------------
Kp_x, Ki_x, Kd_x = 8.0, 0.0, 0.0
Kp_y, Ki_y, Kd_y = 8.0, 0.0, 0.0

# --- Filters ---------------------------------------------------------------
MA_TH_WIN = 11              # samples for theta moving average
MA_DTH_WIN = 7              # samples for dtheta moving average

# --- LCM / runtime globals ------------------------------------------------
listening = False
msg = mbot_balbot_feedback_t()
last_time = 0.0
last_seen = {"MBOT_BALBOT_FEEDBACK": 0.0}


def feedback_handler(channel, data):
    """LCM callback to decode incoming feedback messages."""
    global msg, last_seen, last_time
    last_time = time.time()
    last_seen[channel] = last_time
    msg = mbot_balbot_feedback_t.decode(data)


def lcm_listener(lc):
    """Background thread: handle LCM with a timeout and print warnings.

    This mirrors the behavior in `ballbot_control.py` so users see similar
    console diagnostics when the LCM publisher is inactive.
    """
    global listening
    while listening:
        try:
            lc.handle_timeout(100)  # 100 ms
            if time.time() - last_time > 2.0:
                print("LCM Publisher seems inactive...")
            elif time.time() - last_seen["MBOT_BALBOT_FEEDBACK"] > 2.0:
                print("LCM MBOT_BALBOT_FEEDBACK node seems inactive...")
        except Exception as e:
            print(f"LCM listening error: {e}")
            break


# ------------------------- Helper functions ---------------------------------
def calc_enc2rad(ticks):
    """Convert encoder ticks (absolute) to motor shaft radians."""
    return (2.0 * np.pi * ticks) / (N_ENC * N_GEARBOX)


def calc_torque_conv(Tx, Ty, Tz):
    """Convert virtual torques (Tx,Ty,Tz) to individual motor torques u1,u2,u3."""
    u1 = (1.0 / 3.0) * (Tz - ((2.0 * Ty) / np.cos(alpha)))
    u2 = (1.0 / 3.0) * (Tz + (1.0 / np.cos(alpha)) * (-np.sqrt(3) * Tx + Ty))
    u3 = (1.0 / 3.0) * (Tz + (1.0 / np.cos(alpha)) * ( np.sqrt(3) * Tx + Ty))
    return u1, u2, u3


def calc_kinematic_conv(psi1, psi2, psi3):
    """Motor angles -> ball rotation angles (phi_x, phi_y, phi_z)."""
    phi_x = (np.sqrt(2.0 / 3.0)) * (R_W / R_K) * (psi2 - psi3)
    phi_y = (np.sqrt(2.0) / 3.0) * (R_W / R_K) * (-2.0 * psi1 + psi2 + psi3)
    phi_z = (np.sqrt(2.0) / 3.0) * (R_W / R_K) * (psi1 + psi2 + psi3)
    return phi_x, phi_y, phi_z


def func_clip(x, lo, hi):
    """Clip x to the inclusive range [lo, hi]."""
    return max(lo, min(hi, x))


class MovingAverage:
    """Simple fixed-window moving average with O(1) update."""

    def __init__(self, win):
        self.win = max(1, int(win))
        self.buf = np.zeros(self.win, dtype=float)
        self.sum = 0.0
        self.idx = 0
        self.count = 0

    def reset(self, x0=0.0):
        self.buf[:] = x0
        self.sum = float(x0) * self.win
        self.idx = 0
        self.count = self.win

    def step(self, x):
        old = self.buf[self.idx]
        self.sum -= old
        self.buf[self.idx] = x
        self.sum += x
        self.idx = (self.idx + 1) % self.win
        if self.count < self.win:
            self.count += 1
        return self.sum / self.count


def error_with_deadband(err, db_rad):
    return 0.0 if abs(err) < db_rad else err


class PIDAxis:
    """PID controller for a single tilt axis. D-term uses measured dtheta."""

    def __init__(self, kp=0.0, ki=0.0, kd=0.0, out_min=-1.0, out_max=1.0, db_rad=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_min = out_min
        self.out_max = out_max
        self.db_rad = db_rad
        self.iaccum = 0.0

    def reset(self):
        self.iaccum = 0.0

    def step(self, theta_meas, dtheta_meas):
        """Compute one PID step.

        Returns (u_sat, debug_dict) where u_sat is the saturated output and
        debug_dict contains diagnostic values used for logging/tuning.
        """
        # desired theta is zero
        err = -theta_meas
        err = error_with_deadband(err, self.db_rad)

        # Integrator
        self.iaccum += err * DT

        # PID (D on measurement)
        uP = self.kp * err
        uI = self.ki * self.iaccum
        uD = -self.kd * dtheta_meas
        u = uP + uI + uD

        # Saturate and apply a simple anti-windup: if saturation happens and
        # integrator would push further into saturation, undo last integration.
        u_sat = func_clip(u, self.out_min, self.out_max)
        if (u != u_sat) and ((u_sat > 0 and err > 0) or (u_sat < 0 and err < 0)):
            self.iaccum -= err * DT
            uI = self.ki * self.iaccum
            u = uP + uI + uD
            u_sat = func_clip(u, self.out_min, self.out_max)

        debug = {
            'err': err,
            'uP': uP,
            'uI': uI,
            'uD': uD,
            'iaccum': self.iaccum,
            'u_unsat': u,
            'u_sat': u_sat,
        }
        return u_sat, debug


def soft_pwm_cap(start_time):
    t = time.time() - start_time
    ramp = np.clip(t / SOFTSTART_SEC, 0.2, 1.0)
    return min(PWM_USER_CAP, ramp * PWM_MAX)


def too_lean(theta_x, theta_y, deg_limit):
    """Return True if the robot is leaned beyond deg_limit (in degrees)."""
    return (abs(theta_x) > np.deg2rad(deg_limit)) or (abs(theta_y) > np.deg2rad(deg_limit))


# ------------------------------ Main ---------------------------------------
def main():
    # --- Data logger setup -------------------------------------------------
    trial_num = int(input("Test Number? "))
    filename = f"ballbot_pid_tuning_{trial_num}.txt"
    dl = dataLogger(filename)

    # --- LCM setup --------------------------------------------------------
    global listening, msg
    lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=0")
    subscription = lc.subscribe("MBOT_BALBOT_FEEDBACK", feedback_handler)
    listening = True
    listener_thread = threading.Thread(target=lcm_listener, args=(lc,), daemon=True)
    listener_thread.start()
    print("Started continuous LCM listener...")

    # Wait briefly for a valid IMU message
    print("Waiting for IMU data...")
    t_wait = time.time()
    while (msg.utime == 0) and (time.time() - t_wait < 3.0):
        lc.handle_timeout(50)
        time.sleep(0.01)

    # --- Controller initialization ----------------------------------------
    theta_x_0 = msg.imu_angles_rpy[0]
    theta_y_0 = msg.imu_angles_rpy[1]
    theta_z_0 = msg.imu_angles_rpy[2]
    print(f"Offsets set: theta_x_0={theta_x_0:.4f}, theta_y_0={theta_y_0:.4f}, theta_z_0={theta_z_0:.4f}")

    db_rad = np.deg2rad(DEADBAND_TH_DEG)
    pid_x = PIDAxis(Kp_x, Ki_x, Kd_x, out_min=-1.0, out_max=1.0, db_rad=db_rad)
    pid_y = PIDAxis(Kp_y, Ki_y, Kd_y, out_min=-1.0, out_max=1.0, db_rad=db_rad)

    ma_thx = MovingAverage(MA_TH_WIN)
    ma_thy = MovingAverage(MA_TH_WIN)
    ma_dthx = MovingAverage(MA_DTH_WIN)
    ma_dthy = MovingAverage(MA_DTH_WIN)
    ma_thx.reset(0.0)
    ma_thy.reset(0.0)
    ma_dthx.reset(0.0)
    ma_dthy.reset(0.0)

    command = mbot_motor_pwm_t()

    header = [
        "t",
        "theta_x", "theta_y", "theta_z",
        "theta_x_f", "theta_y_f", "dtheta_x_f", "dtheta_y_f",
        "enc_pos_1", "enc_pos_2", "enc_pos_3",
        "dpsi_1", "dpsi_2", "dpsi_3",
        "phi_x", "phi_y", "phi_z",
        "dphi_x", "dphi_y", "dphi_z",
        "err_x", "err_y",
        "Tx", "Ty", "Tz", "u1", "u2", "u3",
        "Kp_x", "Ki_x", "Kd_x", "Kp_y", "Ki_y", "Kd_y",
        "pwm_cap", "abort",
    ]
    dl.appendData([" ".join(header)])

    # --- Main control loop ------------------------------------------------
    print("Starting PID balance tuning loop...")
    time.sleep(0.3)
    t_start = time.time()
    prev_thx_f = 0.0
    prev_thy_f = 0.0

    try:
        while True:
            loop_t0 = time.time()

            # read sensors
            theta_x = msg.imu_angles_rpy[0] - theta_x_0
            theta_y = msg.imu_angles_rpy[1] - theta_y_0
            theta_z = msg.imu_angles_rpy[2] - theta_z_0

            # filtering
            thx_f = ma_thx.step(theta_x)
            thy_f = ma_thy.step(theta_y)

            # derivatives (finite diff) + MA
            dthx = (thx_f - prev_thx_f) / DT
            dthy = (thy_f - prev_thy_f) / DT
            prev_thx_f = thx_f
            prev_thy_f = thy_f
            dthx_f = ma_dthx.step(dthx)
            dthy_f = ma_dthy.step(dthy)

            # read encoder positions & delta ticks
            enc_pos_1 = msg.enc_ticks[0]
            enc_pos_2 = msg.enc_ticks[1]
            enc_pos_3 = msg.enc_ticks[2]
            enc_dtick_1 = msg.enc_delta_ticks[0]
            enc_dtick_2 = msg.enc_delta_ticks[1]
            enc_dtick_3 = msg.enc_delta_ticks[2]
            enc_dt = msg.enc_delta_time  # microseconds

            # motor angles and speeds
            psi_1 = calc_enc2rad(enc_pos_1)
            psi_2 = calc_enc2rad(enc_pos_2)
            psi_3 = calc_enc2rad(enc_pos_3)
            if enc_dt > 0:
                dpsi_1 = calc_enc2rad(enc_dtick_1) / (enc_dt * 1e-6)
                dpsi_2 = calc_enc2rad(enc_dtick_2) / (enc_dt * 1e-6)
                dpsi_3 = calc_enc2rad(enc_dtick_3) / (enc_dt * 1e-6)
            else:
                dpsi_1 = dpsi_2 = dpsi_3 = 0.0

            # ball kinematics
            phi_x, phi_y, phi_z = calc_kinematic_conv(psi_1, psi_2, psi_3)
            dphi_x, dphi_y, dphi_z = calc_kinematic_conv(dpsi_1, dpsi_2, dpsi_3)

            # safety
            abort = 1 if too_lean(thx_f, thy_f, TH_ABORT_DEG) else 0

            # PID compute
            if abort:
                Tx = 0.0
                Ty = 0.0
                pid_x.reset()
                pid_y.reset()
            else:
                Tx, dbgx = pid_x.step(thx_f, dthx_f)
                Ty, dbgy = pid_y.step(thy_f, dthy_f)

            Tz = 0.0

            # allocate to wheels and apply soft cap
            u1, u2, u3 = calc_torque_conv(Tx, Ty, Tz)
            pwm_cap = soft_pwm_cap(t_start)
            u1 = func_clip(u1, -pwm_cap, pwm_cap)
            u2 = func_clip(u2, -pwm_cap, pwm_cap)
            u3 = func_clip(u3, -pwm_cap, pwm_cap)

            # publish PWM
            cmd_utime = int(time.time() * 1e6)
            command.utime = cmd_utime
            command.pwm[0] = u1
            command.pwm[1] = u2
            command.pwm[2] = u3
            lc.publish("MBOT_MOTOR_PWM_CMD", command.encode())

            # logging
            t_now = time.time() - t_start
            err_x = -thx_f
            err_y = -thy_f
            row = [
                t_now,
                theta_x, theta_y, theta_z,
                thx_f, thy_f, dthx_f, dthy_f,
                # encoder positions & velocities (motor shaft)
                enc_pos_1, enc_pos_2, enc_pos_3,
                dpsi_1, dpsi_2, dpsi_3,
                # ball kinematics
                phi_x, phi_y, phi_z,
                dphi_x, dphi_y, dphi_z,
                err_x, err_y,
                Tx, Ty, Tz, u1, u2, u3,
                pid_x.kp, pid_x.ki, pid_x.kd, pid_y.kp, pid_y.ki, pid_y.kd,
                pwm_cap, abort,
            ]
            dl.appendData(row)

            # brief console output
            print(
                f"t={t_now:5.2f}s | θ(deg)=({np.rad2deg(thx_f):+5.2f},{np.rad2deg(thy_f):+5.2f}) "
                f"| Tx/Ty=({Tx:+.3f},{Ty:+.3f}) | u=({u1:+.3f},{u2:+.3f},{u3:+.3f}) | cap={pwm_cap:.2f} | abort={abort}"
            )

            # rate control
            loop_dt = time.time() - loop_t0
            sleep_t = DT - loop_dt
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Stopping motors...")

    finally:
        # stop motors
        command = mbot_motor_pwm_t()
        command.utime = int(time.time() * 1e6)
        command.pwm[0] = 0.0
        command.pwm[1] = 0.0
        command.pwm[2] = 0.0
        lc.publish("MBOT_MOTOR_PWM_CMD", command.encode())

        # save data
        print(f"Saving data as {filename}...")
        dl.writeOut()

        # stop listener thread
        global listening
        listening = False
        print("Stopping LCM listener...")
        try:
            listener_thread.join(timeout=1)
        except Exception:
            pass

        print("Shutting down complete.\n")


if __name__ == "__main__":
    main()

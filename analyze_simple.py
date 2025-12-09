#!/usr/bin/env python3
"""Simple log analyzer without pandas"""

log_file = "main_control_1.txt"
print(f"Reading {log_file}...")

with open(log_file, 'r') as f:
    lines = f.readlines()

# Parse header
header = lines[0].strip().split()
print(f"Columns: {header}")

# Get column indices
idx_time = header.index('t_now')
idx_theta_x = header.index('theta_x')
idx_theta_y = header.index('theta_y')
idx_theta_d_x = header.index('theta_d_x')
idx_theta_d_y = header.index('theta_d_y')
idx_Tx = header.index('Tx')
idx_Ty = header.index('Ty')
idx_phi_x = header.index('phi_x')
idx_phi_y = header.index('phi_y')

# Show first 20 data lines
print(f"\n=== FIRST 20 SAMPLES ===")
print(f"{'t_now':>8} {'theta_x':>8} {'theta_y':>8} {'theta_d_x':>8} {'theta_d_y':>8} {'phi_x':>8} {'phi_y':>8} {'Tx':>8} {'Ty':>8}")
for i in range(1, min(21, len(lines))):
    parts = lines[i].strip().split()
    if len(parts) > max(idx_time, idx_theta_x, idx_theta_y, idx_Tx, idx_Ty, idx_phi_x, idx_phi_y):
        print(f"{parts[idx_time]:>8} {parts[idx_theta_x]:>8} {parts[idx_theta_y]:>8} {parts[idx_theta_d_x]:>8} {parts[idx_theta_d_y]:>8} {parts[idx_phi_x]:>8} {parts[idx_phi_y]:>8} {parts[idx_Tx]:>8} {parts[idx_Ty]:>8}")

print(f"\n=== LAST 20 SAMPLES ===")
print(f"{'t_now':>8} {'theta_x':>8} {'theta_y':>8} {'theta_d_x':>8} {'theta_d_y':>8} {'phi_x':>8} {'phi_y':>8} {'Tx':>8} {'Ty':>8}")
for i in range(max(1, len(lines)-20), len(lines)):
    parts = lines[i].strip().split()
    if len(parts) > max(idx_time, idx_theta_x, idx_theta_y, idx_Tx, idx_Ty, idx_phi_x, idx_phi_y):
        print(f"{parts[idx_time]:>8} {parts[idx_theta_x]:>8} {parts[idx_theta_y]:>8} {parts[idx_theta_d_x]:>8} {parts[idx_theta_d_y]:>8} {parts[idx_phi_x]:>8} {parts[idx_phi_y]:>8} {parts[idx_Tx]:>8} {parts[idx_Ty]:>8}")

# Calculate some statistics
theta_x_vals = []
theta_y_vals = []
tx_vals = []
ty_vals = []
theta_d_x_vals = []
theta_d_y_vals = []

for i in range(1, len(lines)):
    parts = lines[i].strip().split()
    if len(parts) > max(idx_theta_x, idx_theta_y, idx_Tx, idx_Ty, idx_theta_d_x, idx_theta_d_y):
        try:
            theta_x_vals.append(float(parts[idx_theta_x]))
            theta_y_vals.append(float(parts[idx_theta_y]))
            theta_d_x_vals.append(float(parts[idx_theta_d_x]))
            theta_d_y_vals.append(float(parts[idx_theta_d_y]))
            tx_vals.append(float(parts[idx_Tx]))
            ty_vals.append(float(parts[idx_Ty]))
        except:
            pass

import numpy as np
print(f"\n=== STATISTICS ===")
print(f"Total samples: {len(theta_x_vals)}")
print(f"theta_x: mean={np.mean(theta_x_vals):.2f}°, std={np.std(theta_x_vals):.2f}°, max={np.max(np.abs(theta_x_vals)):.2f}°")
print(f"theta_y: mean={np.mean(theta_y_vals):.2f}°, std={np.std(theta_y_vals):.2f}°, max={np.max(np.abs(theta_y_vals)):.2f}°")
print(f"theta_d_x (commanded): mean={np.mean(theta_d_x_vals):.2f}°, std={np.std(theta_d_x_vals):.2f}°")
print(f"theta_d_y (commanded): mean={np.mean(theta_d_y_vals):.2f}°, std={np.std(theta_d_y_vals):.2f}°")
print(f"Tx: mean={np.mean(tx_vals):.3f}, std={np.std(tx_vals):.3f}")
print(f"Ty: mean={np.mean(ty_vals):.3f}, std={np.std(ty_vals):.3f}")

# Check for oscillations
theta_x_diff = np.abs(np.diff(theta_x_vals))
theta_y_diff = np.abs(np.diff(theta_y_vals))
tx_diff = np.abs(np.diff(tx_vals))
ty_diff = np.abs(np.diff(ty_vals))

print(f"\n=== JITTER/OSCILLATION ANALYSIS ===")
print(f"Avg angle change per sample: theta_x={np.mean(theta_x_diff):.3f}°, theta_y={np.mean(theta_y_diff):.3f}°")
print(f"Max angle change per sample: theta_x={np.max(theta_x_diff):.3f}°, theta_y={np.max(theta_y_diff):.3f}°")
print(f"Avg control change per sample: Tx={np.mean(tx_diff):.3f}, Ty={np.mean(ty_diff):.3f}")
print(f"Max control change per sample: Tx={np.max(tx_diff):.3f}, Ty={np.max(ty_diff):.3f}")

print(f"\nNOTE: The theta_d values show what the position controller was commanding!")
print(f"If these are oscillating, the position control is causing instability.")

# app.py — Self‑Parking AI Simulator (Forward‑Perpendicular + CSV export)
import os
import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st

st.set_page_config(page_title="Self‑Parking AI — Forward Planner", layout="centered")
st.title("🚗 Self‑Parking AI — Forward Perpendicular (MVP)")

# ========================= Helpers (same as notebook) =========================
def generate_bezier_curve(p0, p1, p2, n=40):
    """Quadratic Bézier p0→p2 with control p1; returns (n,2)."""
    t = np.linspace(0, 1, n).reshape(-1, 1)
    p0, p1, p2 = map(np.array, (p0, p1, p2))
    return (1 - t)**2 * p0 + 2*(1 - t)*t*p1 + t**2 * p2

def make_arc(start_xy, mid_offset, end_offset, n=40):
    sx, sy = start_xy
    p0 = (sx, sy)
    p1 = (sx + mid_offset[0], sy + mid_offset[1])
    p2 = (sx + end_offset[0], sy + end_offset[1])
    return generate_bezier_curve(p0, p1, p2, n)

def straight_segment(start_xy, length, heading_deg, n=15):
    sx, sy = start_xy
    rad = np.radians(heading_deg)
    ex, ey = sx + length*np.cos(rad), sy + length*np.sin(rad)
    xs = np.linspace(sx, ex, n); ys = np.linspace(sy, ey, n)
    return np.column_stack([xs, ys])

def concat_paths(*paths):
    out = []
    for k, p in enumerate(paths):
        p = np.asarray(p)
        if k > 0 and len(out) and len(p):
            out.extend(p[1:])   # skip seam duplicate
        else:
            out.extend(p)
    return np.array(out, dtype=float)

def heading_at(path_xy, i):
    p = np.asarray(path_xy)
    i = int(np.clip(i, 1, len(p)-1))
    dx = p[i,0] - p[i-1,0]
    dy = p[i,1] - p[i-1,1]
    return np.degrees(np.arctan2(dy, dx))

def rear_bumper_center(center_xy, yaw_deg, car_length):
    rad = np.radians(yaw_deg)
    return np.array(center_xy) - 0.5 * car_length * np.array([np.cos(rad), np.sin(rad)])

def rotate_path_about_point(points, center_xy, delta_deg):
    rad = np.radians(delta_deg)
    R = np.array([[np.cos(rad), -np.sin(rad)],
                  [np.sin(rad),  np.cos(rad)]])
    P = np.asarray(points) - center_xy
    return (P @ R.T) + center_xy

def compute_rear_target(spot_origin, spot_width, spot_length, car_length,
                        inset=0.10, enter_from='bottom'):
    x = float(spot_origin[0]) + float(spot_width)/2.0
    if enter_from == 'bottom':
        y = float(spot_origin[1]) + inset
    elif enter_from == 'top':
        y = float(spot_origin[1]) + float(spot_length) - inset
    else:
        raise ValueError("enter_from must be 'bottom' or 'top'")
    return np.array([x, y], dtype=float)

def wrap180(a_deg):
    return ((a_deg + 180.0) % 360.0) - 180.0

def path_headings(P):
    d = np.diff(P, axis=0, prepend=P[:1])
    return np.degrees(np.arctan2(d[:,1], d[:,0]))

# ========================= Planner (forward only) =========================
def generate_path_for_strategy_forward(car_length, car_width,
                                       spot_length, spot_width,
                                       distance_to_spot,
                                       nA=60, nS=18, nB=40):
    def clamp(v, lo, hi): return max(lo, min(hi, v))

    x_swing = clamp(0.3 * spot_width, 0.5, 1.0)
    approach_y = clamp(0.4 * distance_to_spot + 0.3 * spot_length,
                       1.0, distance_to_spot + 0.5 * spot_length)

    pathA = make_arc((0.0, 0.0),
                     (+x_swing, 0.55 * approach_y),
                     (+0.25 * x_swing, approach_y),
                     n=nA)
    pathS = straight_segment(tuple(pathA[-1]),
                             length=0.3,
                             heading_deg=85,
                             n=nS)
    pathB = make_arc(tuple(pathS[-1]),
                     (-0.45 * x_swing, 0.25 * spot_length),
                     (-0.18 * x_swing, 0.38 * spot_length),
                     n=nB)
    return concat_paths(pathA, pathS, pathB)

def plan_and_align_forward(car_length, car_width,
                           spot_length, spot_width,
                           spot_origin,
                           distance_to_spot=2.0,
                           inset=0.10):
    curve_points = generate_path_for_strategy_forward(
        car_length, car_width, spot_length, spot_width, distance_to_spot
    )

    # entry side vs bay midline
    spot_mid_y = float(spot_origin[1]) + float(spot_length)/2.0
    entry_side = 'bottom' if curve_points[0, 1] < spot_mid_y else 'top'

    # land rear bumper on bay inner edge
    rear_target = compute_rear_target(spot_origin, spot_width, spot_length,
                                      car_length, inset=inset, enter_from=entry_side)

    # align yaw to +90°
    end_center = curve_points[-1]
    yaw_end = heading_at(curve_points, len(curve_points) - 1)
    rear_now = rear_bumper_center(end_center, yaw_end, car_length)

    yaw_target = 90.0
    delta_yaw = wrap180(yaw_target - yaw_end)
    curve_points = rotate_path_about_point(curve_points, rear_now, delta_yaw)

    # translate rear bumper to target
    end_center_rot = curve_points[-1]
    rear_now_rot = rear_bumper_center(end_center_rot, yaw_target, car_length)
    delta = rear_target - rear_now_rot
    curve_points = curve_points + delta

    return {
        "curve_points": np.asarray(curve_points, dtype=float),
        "rear_target":  np.asarray(rear_target, dtype=float),
        "yaw_target":   float(yaw_target),
        "entry_side":   entry_side,
        "rear_bumper_after": np.asarray(rear_target, dtype=float),
    }
# ---------------- PID follow of saved CSV ----------------
# (Place this after your existing plotting/summary block, still inside app.py)

import math

# Sidebar controls for PID & vehicle model
with st.sidebar:
    st.header("PID Follow (optional)")
    pid_kp = st.slider("PID kp", 0.0, 5.0, 1.2, 0.1)
    pid_ki = st.slider("PID ki", 0.0, 1.0, 0.0, 0.01)
    pid_kd = st.slider("PID kd", 0.0, 5.0, 0.2, 0.1)
    sim_speed = st.slider("Sim speed v (m/s)", 0.3, 3.0, 1.5, 0.1)
    wheelbase = st.slider("Wheelbase L (m)", 2.0, 3.2, 2.5, 0.05)
    max_steer_deg = st.slider("Max steer (deg)", 10, 45, 35, 1)
    run_pid_btn = st.button("Run PID on saved CSV")

def _wrap_pi(a):
    # wrap angle to [-pi, pi)
    return (a + math.pi) % (2*math.pi) - math.pi

def _load_control_points_csv():
    # tries data/control_points_forward.csv then artifacts/control_points_forward.csv
    base = os.path.dirname(__file__)
    candidates = [
        os.path.join(base, "data", "control_points_forward.csv"),
        os.path.join(base, "artifacts", "control_points_forward.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                arr = np.loadtxt(p, delimiter=",", skiprows=1)  # columns: x,y,heading_deg
                if arr.ndim == 1:
                    arr = arr[None, :]
                return arr
            except Exception as e:
                st.error(f"Failed to read {p}: {e}")
                return None
    return None

class PID:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self.ei = 0.0
        self.prev_e = 0.0
    def __call__(self, e, dt):
        self.ei += e * dt
        de = (e - self.prev_e) / dt if dt > 0 else 0.0
        self.prev_e = e
        return self.kp * e + self.ki * self.ei + self.kd * de

def _nearest_index(path_xy, x, y):
    d = np.hypot(path_xy[:,0] - x, path_xy[:,1] - y)
    return int(np.argmin(d))

def _bicycle_step(x, y, yaw, delta, v, L, dt):
    # x,y in meters; yaw, delta in radians
    x   += v * math.cos(yaw) * dt
    y   += v * math.sin(yaw) * dt
    yaw += (v / L) * math.tan(delta) * dt
    return x, y, _wrap_pi(yaw)

if run_pid_btn:
    data = _load_control_points_csv()
    if data is None:
        st.error("No control_points_forward.csv found in data/ or artifacts/. "
                 "Click 'Plan Path' first (it exports a CSV), then try again.")
    else:
        # Extract a 2D path (x,y) from the CSV
        xy_path = data[:, :2]
        if len(xy_path) < 3:
            st.error("CSV has too few points to follow.")
        else:
            # --- PID follow setup ---
            pid = PID(pid_kp, pid_ki, pid_kd)
            dt = 0.05
            max_steer = math.radians(max_steer_deg)
            total_time = 20.0  # seconds
            steps = int(total_time / dt)

            # Start near the first point, slightly “below” it
            x, y = xy_path[0,0], xy_path[0,1] - 2.0
            yaw = math.radians(90.0)  # facing +Y

            traj = []
            lookahead_pts = 5  # aim a few points ahead for smoother steering

            for _ in range(steps):
                # 1) choose a lookahead target along the path
                i_near = _nearest_index(xy_path, x, y)
                i_tgt = min(i_near + lookahead_pts, len(xy_path) - 1)
                tx, ty = xy_path[i_tgt]

                # 2) desired heading toward target & heading error
                desired_yaw = math.atan2(ty - y, tx - x)
                e_yaw = _wrap_pi(desired_yaw - yaw)

                # 3) PID -> steering command (radians)
                delta = pid(e_yaw, dt)
                delta = max(-max_steer, min(max_steer, delta))

                # 4) bicycle step
                x, y, yaw = _bicycle_step(x, y, yaw, delta, sim_speed, wheelbase, dt)
                traj.append([x, y, yaw, delta])

                # early stop if close to path end
                if np.hypot(x - xy_path[-1,0], y - xy_path[-1,1]) < 0.2:
                    break

            traj = np.asarray(traj)

            # --- Plot result ---
            fig2, ax2 = plt.subplots(figsize=(6, 8))
            ax2.grid(True, alpha=0.35)
            ax2.plot(xy_path[:,0], xy_path[:,1], 'k--', lw=2, label="Reference path")
            ax2.plot(traj[:,0], traj[:,1], 'r-', lw=2, label="PID trajectory")
            ax2.legend(loc='upper left')
            ax2.set_aspect('equal', adjustable='box')
            ax2.set_title("PID Follow of Saved Control Points")
            st.pyplot(fig2)

            # quick stats
            final_err = float(np.hypot(traj[-1,0] - xy_path[-1,0], traj[-1,1] - xy_path[-1,1]))
            st.write(f"Final position error: **{final_err:.2f} m**")
            
# ========================= UI (sidebar) =========================
with st.sidebar:
    st.header("Inputs")
    spot_length = st.number_input("Spot length (m)", 3.5, 8.0, 6.2, 0.1)
    spot_width  = st.number_input("Spot width (m)",  2.2, 4.0, 3.0, 0.1)

    size_preset = st.selectbox("Car size preset", ["Relative: 60% × 55%", "Manual"])
    if size_preset == "Relative: 60% × 55%":
        car_length = spot_length * 0.60
        car_width  = spot_width  * 0.55
        st.caption(f"Car L×W = {car_length:.2f} × {car_width:.2f} m")
    else:
        car_length = st.number_input("Car length (m)", 3.0, 5.6, 4.5, 0.1)
        car_width  = st.number_input("Car width (m)",  1.5, 2.2, 1.8, 0.05)

    distance_to_spot = st.slider("Distance to spot (m)", 0.5, 6.0, 2.0, 0.1)
    inset = st.slider("Rear bumper inset (m)", 0.05, 0.5, 0.10, 0.01)

    spot_x = st.number_input("Spot origin X (m)", -10.0, 30.0, 15.0, 0.1)
    spot_y = st.number_input("Spot origin Y (m)", -10.0, 30.0,  3.0, 0.1)

    run_button = st.button("Plan Path")

spot_origin = np.array([spot_x, spot_y], dtype=float)

# ========================= Run & Render =========================
if run_button:
    plan = plan_and_align_forward(
        car_length=car_length,
        car_width=car_width,
        spot_length=spot_length,
        spot_width=spot_width,
        spot_origin=spot_origin,
        distance_to_spot=distance_to_spot,
        inset=inset
    )

    curve_points = plan["curve_points"]
    rear_target  = plan["rear_target"]

    # --- plot ---
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.grid(True, alpha=0.35)

    bay = patches.Rectangle(
        (spot_origin[0], spot_origin[1]),
        spot_width, spot_length,
        linewidth=1.8, edgecolor='green', facecolor='none',
        linestyle='--', label='Parking Spot', zorder=5
    )
    ax.add_patch(bay)

    ax.scatter(rear_target[0], rear_target[1], s=80, marker='x', label='Rear Target')
    rb = plan["rear_bumper_after"]
    ax.scatter(rb[0], rb[1], s=60, marker='+', label='Rear Bumper (after align)')
    ax.plot(curve_points[:,0], curve_points[:,1], lw=2.2, label='Path')

    xs = curve_points[:,0]; ys = curve_points[:,1]
    sx0, sy0 = float(spot_origin[0]), float(spot_origin[1])
    sx1, sy1 = sx0 + float(spot_width), sy0 + float(spot_length)
    pad = 0.8
    ax.set_xlim(min(xs.min(), sx0, sx1) - pad, max(xs.max(), sx0, sx1) + pad)
    ax.set_ylim(min(ys.min(), sy0, sy1) - pad, max(ys.max(), sy0, sy1) + pad)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='upper left')

    st.pyplot(fig)

    # --- Summary ---
    st.subheader("Plan Summary")
    st.write(f"- Entry side (auto): **{plan['entry_side']}**")
    st.write(f"- Final heading: **{plan['yaw_target']:.1f}°**")
    st.write(f"- Car (L×W): **{car_length:.2f} × {car_width:.2f} m**")
    st.write(f"- Spot (L×W): **{spot_length:.2f} × {spot_width:.2f} m**")
    st.write(f"- Inset: **{inset:.2f} m**, Distance: **{distance_to_spot:.2f} m**")

    # --- Export control points (x, y, heading_deg) ---
    cp = np.column_stack([curve_points, path_headings(curve_points)])

    # Save copy to artifacts/ (optional)
    artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    file_disk = os.path.join(artifacts_dir, "control_points_forward.csv")
    np.savetxt(file_disk, cp, delimiter=",", header="x,y,heading_deg", comments="")
    st.caption(f"Saved a copy to: `{file_disk}`")

    # Offer direct download
    buf = io.StringIO()
    np.savetxt(buf, cp, delimiter=",", header="x,y,heading_deg", comments="")
    st.download_button("⬇️ Download control_points_forward.csv",
                       buf.getvalue(),
                       file_name="control_points_forward.csv",
                       mime="text/csv")
else:
    st.info("Set inputs in the sidebar and click **Plan Path**.")

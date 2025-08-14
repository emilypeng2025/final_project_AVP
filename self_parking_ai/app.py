# app.py — Self‑Parking AI Simulator (Forward Perpendicular MVP)

import streamlit as st
st.set_page_config(page_title="Self‑Parking AI Simulator", layout="centered")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ========================= UI =========================
st.title("🚗 Self‑Parking AI Simulator — Forward Perpendicular (MVP)")

with st.sidebar:
    st.header("Inputs")

    # Parking spot geometry
    spot_length = st.number_input("Spot length (m)", 3.5, 8.0, 6.2, 0.1)
    spot_width  = st.number_input("Spot width (m)",  2.2, 4.0, 3.0, 0.1)

    # Car geometry (preset or manual)
    size_preset = st.selectbox("Car size preset", ["Relative: 60% × 55%", "Manual"])
    if size_preset == "Relative: 60% × 55%":
        car_length = spot_length * 0.60
        car_width  = spot_width  * 0.55
        st.caption(f"Using preset → Car L×W = {car_length:.2f} × {car_width:.2f} m")
    else:
        car_length = st.number_input("Car length (m)", 3.0, 5.6, 4.5, 0.1)
        car_width  = st.number_input("Car width (m)",  1.5, 2.2, 1.8, 0.05)

    # Approach & placement tuning
    distance_to_spot = st.slider("Distance to spot (m)", 0.5, 6.0, 2.0, 0.1)
    inset = st.slider("Rear bumper inset (m)", 0.05, 0.5, 0.10, 0.01)

    # Where the bottom‑left corner of the bay is placed in the world
    spot_x = st.number_input("Spot origin X (m)", -10.0, 30.0, 15.0, 0.1)
    spot_y = st.number_input("Spot origin Y (m)", -10.0, 30.0,  3.0, 0.1)

    run_button = st.button("Plan Path")

spot_origin = np.array([spot_x, spot_y], dtype=float)


# ========================= Geometry Helpers =========================
def generate_bezier_curve(p0, p1, p2, n=40):
    """Quadratic Bézier curve from p0→p2 with control p1; returns (n,2) points."""
    t = np.linspace(0, 1, n).reshape(-1, 1)
    p0, p1, p2 = map(np.array, (p0, p1, p2))
    return (1 - t)**2 * p0 + 2*(1 - t)*t*p1 + t**2 * p2  # shape: (n, 2)

def make_arc(start_xy, mid_offset, end_offset, n=40):
    """Build a quadratic arc using offsets from start."""
    sx, sy = start_xy
    p0 = (sx, sy)
    p1 = (sx + mid_offset[0], sy + mid_offset[1])
    p2 = (sx + end_offset[0], sy + end_offset[1])
    return generate_bezier_curve(p0, p1, p2, n)

def straight_segment(start_xy, length, heading_deg, n=15):
    """Straight segment of given length & heading (deg); returns (n,2) samples."""
    sx, sy = start_xy
    rad = np.radians(heading_deg)
    ex, ey = sx + length*np.cos(rad), sy + length*np.sin(rad)
    xs = np.linspace(sx, ex, n); ys = np.linspace(sy, ey, n)
    return np.column_stack([xs, ys])

def concat_paths(*paths):
    """Concatenate multiple (N,2) path segments; skip duplicate seam points."""
    out = []
    for k, p in enumerate(paths):
        p = np.asarray(p)
        if k > 0 and len(out) and len(p):
            out.extend(p[1:])  # avoid duplicating the seam
        else:
            out.extend(p)
    return np.array(out, dtype=float)

def heading_at(path_xy, i):
    """Approx heading (deg) at index i via finite differences."""
    p = np.asarray(path_xy)
    i = int(np.clip(i, 1, len(p)-1))
    dx = p[i, 0] - p[i-1, 0]
    dy = p[i, 1] - p[i-1, 1]
    return np.degrees(np.arctan2(dy, dx))

def rear_bumper_center(center_xy, yaw_deg, car_length):
    """Rear‑bumper position given car center & yaw."""
    rad = np.radians(yaw_deg)
    return np.array(center_xy) - 0.5 * car_length * np.array([np.cos(rad), np.sin(rad)])

def rotate_path_about_point(points, center_xy, delta_deg):
    """Rotate an array/list of (x,y) about center_xy by delta_deg degrees."""
    rad = np.radians(delta_deg)
    R = np.array([[np.cos(rad), -np.sin(rad)],
                  [np.sin(rad),  np.cos(rad)]])
    P = np.asarray(points) - center_xy
    return (P @ R.T) + center_xy

def compute_rear_target(spot_origin, spot_width, spot_length, car_length,
                        inset=0.10, enter_from='bottom'):
    """Rear‑bumper landing target on the bay’s inner edge (centered in x)."""
    x = float(spot_origin[0]) + float(spot_width) / 2.0
    if enter_from == 'bottom':
        y = float(spot_origin[1]) + inset
    elif enter_from == 'top':
        y = float(spot_origin[1]) + float(spot_length) - inset
    else:
        raise ValueError("enter_from must be 'bottom' or 'top'")
    return np.array([x, y], dtype=float)

def wrap180(a_deg):
    """Wrap angle to [-180, 180) degrees."""
    return ((a_deg + 180.0) % 360.0) - 180.0


# ========================= Forward Path Template =========================
def generate_path_for_strategy_forward(car_length, car_width,
                                       spot_length, spot_width,
                                       distance_to_spot,
                                       nA=60, nS=20, nB=40):
    """
    Create a forward‑perpendicular approach in *local* frame.
    We’ll align it to the actual bay later.
    """
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


# ========================= Plan & Align =========================
def plan_and_align_forward(car_length, car_width,
                           spot_length, spot_width,
                           spot_origin,
                           distance_to_spot=2.0,
                           inset=0.10):
    """
    1) Build a local template path
    2) Detect entry side (above/below bay centerline)
    3) Rotate about current rear bumper so final yaw is +90°
    4) Translate so rear bumper lands at the chosen target on the bay edge
    """
    curve_points = generate_path_for_strategy_forward(
        car_length, car_width, spot_length, spot_width, distance_to_spot
    )

    # Decide entry side using start y vs bay mid y (world y)
    spot_mid_y = float(spot_origin[1]) + float(spot_length)/2.0
    entry_side = 'bottom' if curve_points[0, 1] < spot_mid_y else 'top'

    # Desired rear‑bumper landing point
    rear_target = compute_rear_target(spot_origin, spot_width, spot_length,
                                      car_length, inset=inset, enter_from=entry_side)

    # Align heading: final pose perpendicular inside bay → +90°
    end_center = curve_points[-1]
    yaw_end = heading_at(curve_points, len(curve_points) - 1)
    rear_now = rear_bumper_center(end_center, yaw_end, car_length)

    yaw_target = 90.0
    delta_yaw = wrap180(yaw_target - yaw_end)
    curve_points = rotate_path_about_point(curve_points, rear_now, delta_yaw)

    # Translate so rear bumper matches target
    end_center_rot = curve_points[-1]
    rear_now_rot = rear_bumper_center(end_center_rot, yaw_target, car_length)
    delta = rear_target - rear_now_rot
    curve_points = curve_points + delta

    return {
        "curve_points": curve_points,
        "rear_target": rear_target,
        "yaw_target": float(yaw_target),
        "entry_side": entry_side,
        "rear_bumper_after": rear_target.copy()
    }


# ========================= Run =========================
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

    # Plot
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.grid(True, alpha=0.35)

    # Parking bay rectangle
    spot_rect = patches.Rectangle(
        (spot_origin[0], spot_origin[1]),
        spot_width, spot_length,
        linewidth=1.8, edgecolor='green', facecolor='none',
        linestyle='--', label='Parking Spot', zorder=5
    )
    ax.add_patch(spot_rect)

    # Targets & path
    rear_target = plan["rear_target"]
    ax.scatter(rear_target[0], rear_target[1], s=80, marker='x', label='Rear Target')
    rb = plan["rear_bumper_after"]
    ax.scatter(rb[0], rb[1], s=60, marker='+', label='Rear Bumper (after align)')
    ax.plot(curve_points[:, 0], curve_points[:, 1], lw=2.2, label='Path')

    # Plot limits with padding
    xs = curve_points[:, 0]; ys = curve_points[:, 1]
    sx0, sy0 = float(spot_origin[0]), float(spot_origin[1])
    sx1, sy1 = sx0 + float(spot_width), sy0 + float(spot_length)
    pad = 0.8
    ax.set_xlim(min(xs.min(), sx0, sx1) - pad, max(xs.max(), sx0, sx1) + pad)
    ax.set_ylim(min(ys.min(), sy0, sy1) - pad, max(ys.max(), sy0, sy1) + pad)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='upper left')

    st.pyplot(fig)

    # Summary
    st.subheader("Plan Summary")
    st.write(f"- Entry side (auto): **{plan['entry_side']}**")
    st.write(f"- Final heading: **{plan['yaw_target']:.1f}°**")
    st.write(f"- Car (L×W): **{car_length:.2f} × {car_width:.2f} m**")
    st.write(f"- Spot (L×W): **{spot_length:.2f} × {spot_width:.2f} m**")
    st.write(f"- Inset: **{inset:.2f} m**, Distance: **{distance_to_spot:.2f} m**")
else:
    st.info("Set inputs in the sidebar and click **Plan Path**.")

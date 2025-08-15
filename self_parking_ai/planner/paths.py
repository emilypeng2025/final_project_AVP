# planner/paths.py — forward-perpendicular path + alignment (MVP)

import numpy as np

# --------------- Geometry helpers ---------------

def generate_bezier_curve(p0, p1, p2, n=40):
    """Quadratic Bézier p0→p2 with control p1; returns shape (n, 2)."""
    t = np.linspace(0, 1, n).reshape(-1, 1)
    p0, p1, p2 = map(np.array, (p0, p1, p2))
    return (1 - t)**2 * p0 + 2*(1 - t)*t*p1 + t**2 * p2

def make_arc(start_xy, mid_offset, end_offset, n=40):
    """Build a quadratic arc using offsets from start_xy."""
    sx, sy = start_xy
    p0 = (sx, sy)
    p1 = (sx + mid_offset[0], sy + mid_offset[1])
    p2 = (sx + end_offset[0], sy + end_offset[1])
    return generate_bezier_curve(p0, p1, p2, n)

def straight_segment(start_xy, length, heading_deg, n=15):
    """Straight segment of given length & heading (deg); returns (n,2)."""
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
            out.extend(p[1:])          # avoid duplicating the seam
        else:
            out.extend(p)
    return np.array(out, dtype=float)

def heading_at(path_xy, i):
    """Approx heading (deg) at index i via finite differences."""
    p = np.asarray(path_xy)
    if len(p) < 2:
        return 0.0
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

# ====== Reverse Parallel into Perpendicular Bay ======
# Minimal, smooth S‑curve reverse-in. Uses your existing helpers:
# - straight_segment, heading_at, rear_bumper_center, rotate_path_about_point,
#   compute_rear_target, wrap180
# If you already have sample_bezier defined above, you can delete this one.

def sample_bezier(p0, p1, p2, p3, n=50):
    """Cubic Bézier from p0->p3 (controls p1, p2). Returns (n,2) array."""
    import numpy as np
    t = np.linspace(0.0, 1.0, n)[:, None]
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float); p3 = np.asarray(p3, float)
    B = (1-t)**3*p0 + 3*(1-t)**2*t*p1 + 3*(1-t)*t**2*p2 + t**3*p3
    return B

def _reverse_template_path(car_length, car_width, spot_length, spot_width, distance_to_spot,
                           n_pull=18, n_s1=36, n_s2=36):
    """
    Build a local-frame reverse S-curve:
      1) Pull ahead (slight +y)
      2) Curve left/down
      3) Counter-curve right/down to a target just inside the bay edge
    Path is in *local* coords; final alignment happens later.
    """
    import numpy as np

    def clamp(v, lo, hi): return max(lo, min(hi, v))

    # 1) Pull-ahead to set angle before reversing
    pull = clamp(0.8 * distance_to_spot, 0.5, max(0.6, distance_to_spot))
    path0 = straight_segment((0.0, 0.0), length=pull, heading_deg=85, n=n_pull)

    # 2) Define center target slightly inside the bay inner edge (local y≈0 line)
    inset = 0.10
    rear_target_y = -inset                 # just inside the line (downwards in template)
    target_center = np.array([0.0, rear_target_y + 0.5 * car_length], dtype=float)

    # 3) Lateral sway for S curve based on spot/car width
    side_x = clamp(0.5 * spot_width + 0.3 * car_width, 0.8, 1.8)

    # 4) Mid waypoint (between pull end and target), nudged downward
    p_start = np.asarray(path0[-1], float)
    mid = np.array([
        0.6 * p_start[0] + 0.4 * target_center[0],
        0.6 * p_start[1] + 0.4 * target_center[1] - 1.0
    ], dtype=float)

    # 5) Two cubic Béziers (gentle S)
    #    First segment control points: pull back from start tangent, slightly up
    th = np.deg2rad(85.0)
    t0 = np.array([np.cos(th), np.sin(th)])
    k1, k2 = 2.2, 1.5

    p0 = p_start
    c1 = p0 - k1 * t0
    c2 = np.array([mid[0], mid[1] + 0.6])
    seg1 = sample_bezier(p0, c1, c2, mid, n=n_s1)

    #    Second segment control points: slightly down then into target
    c3 = np.array([mid[0], mid[1] - 0.6])
    c4 = target_center + np.array([0.0, -k2])
    seg2 = sample_bezier(mid, c3, c4, target_center, n=n_s2)

    # 6) Impose left-then-right lateral offsets for a clear S shape
    seg1[:, 0] -= 0.4 * side_x
    seg2[:, 0] += 0.4 * side_x

    # Concatenate: pull ahead + S‑curve (skip seam duplicates)
    path = concat_paths(path0, seg1[1:], seg2[1:])
    return path

def plan_and_align_reverse_parallel(car_length, car_width,
                                    spot_length, spot_width,
                                    spot_origin,
                                    distance_to_spot=2.0,
                                    inset=0.10):
    """
    Reverse‑in maneuver into a perpendicular bay.
    Steps:
      1) Build a local S‑curve template
      2) Auto-detect entry side (above/below bay mid y)
      3) Compute rear-bumper target on bay edge
      4) Rotate path about current rear bumper to yaw_target (+90°) 
      5) Translate so rear bumper lands exactly on the target
    """
    import numpy as np

    # 1) Local template
    curve_points = _reverse_template_path(car_length, car_width, spot_length, spot_width,
                                          distance_to_spot)

    # 2) Entry side (world-space bay mid y)
    spot_mid_y = float(spot_origin[1]) + float(spot_length) / 2.0
    entry_side = 'bottom' if curve_points[0, 1] < spot_mid_y else 'top'

    # 3) Rear-bumper target on inner edge (centered in x)
    rear_target = compute_rear_target(
        spot_origin, spot_width, spot_length, car_length,
        inset=inset, enter_from=entry_side
    )

    # 4) Rotate about *current rear bumper* to align yaw to +90° inside bay
    end_center = curve_points[-1]
    yaw_end    = heading_at(curve_points, len(curve_points) - 1)  # degrees
    rear_now   = rear_bumper_center(end_center, yaw_end, car_length)

    yaw_target = 90.0   # perpendicular to the bay (pointing “up”)
    delta_yaw  = wrap180(yaw_target - yaw_end)
    curve_points = rotate_path_about_point(curve_points, rear_now, delta_yaw)

    # 5) Translate so rear bumper sits exactly at the target
    end_center_rot = curve_points[-1]
    rear_now_rot   = rear_bumper_center(end_center_rot, yaw_target, car_length)
    delta = rear_target - rear_now_rot
    curve_points = curve_points + delta

    return {
        "curve_points": np.asarray(curve_points, dtype=float),
        "rear_target": np.asarray(rear_target, dtype=float),
        "yaw_target": float(yaw_target),
        "entry_side": entry_side,
        "rear_bumper_after": np.asarray(rear_target, dtype=float),
        "ghost_outline": None,
        "fits_width": None,
        "fits_length": None,
    }

# --------------- Forward‑perpendicular template ---------------

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

    # gentle S‑like approach: arc → very short straight → arc
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


# --------------- Plan & align (public API for app.py) ---------------

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
    Returns a dict consumed by app.py.
    """
    # 1) template path
    curve_points = generate_path_for_strategy_forward(
        car_length, car_width, spot_length, spot_width, distance_to_spot
    )

    # 2) entry side (use start y vs bay mid y in world coords)
    spot_mid_y = float(spot_origin[1]) + float(spot_length)/2.0
    entry_side = 'bottom' if curve_points[0, 1] < spot_mid_y else 'top'

    # desired rear‑bumper landing point on bay edge
    rear_target = compute_rear_target(spot_origin, spot_width, spot_length,
                                      car_length, inset=inset, enter_from=entry_side)

    # 3) align heading: final pose perpendicular inside bay → +90°
    end_center = curve_points[-1]
    yaw_end = heading_at(curve_points, len(curve_points) - 1)
    rear_now = rear_bumper_center(end_center, yaw_end, car_length)

    yaw_target = 90.0
    delta_yaw = wrap180(yaw_target - yaw_end)
    curve_points = rotate_path_about_point(curve_points, rear_now, delta_yaw)

    # 4) translate so rear bumper matches target
    end_center_rot = curve_points[-1]
    rear_now_rot = rear_bumper_center(end_center_rot, yaw_target, car_length)
    delta = rear_target - rear_now_rot
    curve_points = curve_points + delta

    return {
        "curve_points": np.asarray(curve_points, dtype=float),
        "rear_target": np.asarray(rear_target, dtype=float),
        "yaw_target": float(yaw_target),
        "entry_side": entry_side,
        "rear_bumper_after": np.asarray(rear_target, dtype=float),
    }

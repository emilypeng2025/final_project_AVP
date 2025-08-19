# avp_free.py — Bezier path (two curves + straight) pure-pursuit, x–y ground
print("[IMU_FREE] controller loaded")

import os, csv, math
from vehicle import Driver
from controller import GPS

# ================== USER / SCENE KNOBS ==================
CSV_FILE = os.path.join(os.path.dirname(__file__), "path.csv")

# Set your bay center & facing here (world coordinates, x–y ground)
GOAL_X, GOAL_Y = -57.0, 45.88     # center of blue bay you placed
GOAL_YAW       = math.pi          # facing -X (same as your car spawn heading)

# If you want the final straight to stop a touch before the geometric center:
FINAL_STOP_OFFSET = 0.0           # meters along -GOAL_YAW; 0.0 = center

# Optional MLP hook: choose how “wide” the curves are (in meters)
def mlp_decide(features: dict) -> float:
    """
    Plug your MLP here. Return a lateral scale (how wide curves swing).
    For now, return a sensible default. Use `features` if you want.
    """
    # e.g., return my_mlp.predict([features_vec])[0]
    return 6.0  # meters lateral swing (try 4–10)
# ========================================================

# ---------------- Driving & controller tuning ------------
SPEED_CRUISE = 9.0       # km/h cruise
SPEED_MIN    = 2.0       # km/h near goal
MAX_STEER    = 0.5       # rad
WHEELBASE    = 2.7       # m

WAYPOINT_RADIUS       = 0.8          # m
LOOK_MIN, LOOK_MAX    = 1.2, 5.0     # m
ALIGN_TOL             = math.radians(6)
STOP_DIST             = 0.40         # m to goal if not inside bay
FORWARD_BIAS          = 0.40         # push the goal forward into the bay (meters)

# Visual bay footprint used for inside_bay test
BAY_LEN, BAY_WID      = 6.2, 2.6     # meters
# --------------------------------------------------------

def clamp(x, a, b): return a if x < a else b if x > b else x
def dist(a, b):     return math.hypot(a[0]-b[0], a[1]-b[1])

def load_xy(csv_path):
    pts = []
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            r = csv.reader(f)
            for i, row in enumerate(r):
                if i == 0 and any(c.isalpha() for c in ",".join(row)):  # header
                    continue
                if len(row) >= 2:
                    pts.append((float(row[0]), float(row[1])))
    return pts

# -------- Bezier utilities (cubic) --------
def bezier3(P0, P1, P2, P3, n=40):
    """Sample a cubic Bezier from P0..P3 into n points (including ends)."""
    out = []
    for i in range(n):
        t = i/(n-1)
        u = 1-t
        x = (u*u*u)*P0[0] + 3*(u*u)*t*P1[0] + 3*u*(t*t)*P2[0] + (t*t*t)*P3[0]
        y = (u*u*u)*P0[1] + 3*(u*u)*t*P1[1] + 3*u*(t*t)*P2[1] + (t*t*t)*P3[1]
        out.append((x, y))
    return out

def heading(p, q):
    return math.atan2(q[1]-p[1], q[0]-p[0])

# -------- Pure-pursuit & tests --------
def pure_pursuit_steer(x, y, yaw, target, look_min=LOOK_MIN, look_max=LOOK_MAX):
    dx = target[0] - x
    dy = target[1] - y
    # transform to vehicle frame
    tx =  math.cos(-yaw)*dx - math.sin(-yaw)*dy
    ty =  math.sin(-yaw)*dx + math.cos(-yaw)*dy
    if tx < 0.1: tx = 0.1
    Ld = clamp(math.hypot(tx, ty), look_min, look_max)
    curvature = 2.0 * ty / (Ld * Ld)
    steer = math.atan(curvature * WHEELBASE)
    return clamp(steer, -MAX_STEER, MAX_STEER)

def inside_bay(px, py, cx, cy, yaw, half_len, half_wid):
    dx, dy = px - cx, py - cy
    u =  math.cos(-yaw)*dx - math.sin(-yaw)*dy  # along bay heading
    v =  math.sin(-yaw)*dx + math.cos(-yaw)*dy  # left/right in bay frame
    return (abs(u) <= half_len) and (abs(v) <= half_wid)

# -------- Build a forward path: curve left, curve right, straight-in --------
def build_procedural_path(start_xy, start_yaw, goal_xy, goal_yaw):
    sx, sy = start_xy
    gx, gy = goal_xy
    features = {
        "dx": gx - sx,
        "dy": gy - sy,
        "start_yaw": start_yaw,
        "goal_yaw": goal_yaw,
        "range": dist(start_xy, goal_xy),
    }
    # how wide to swing the curves (your MLP can decide this)
    LATERAL = mlp_decide(features)     # meters, try 4–10
    AHEAD   = max(10.0, features["range"] * 0.5)  # how far each curve pulls forward

    # Segment A: gentle LEFT from start
    P0 = (sx, sy)
    P1 = (sx + math.cos(start_yaw)*AHEAD*0.5,
          sy + math.sin(start_yaw)*AHEAD*0.5)
    # left is +90° from heading
    P2 = (P1[0] + math.cos(start_yaw + math.pi/2.0)*LATERAL,
          P1[1] + math.sin(start_yaw + math.pi/2.0)*LATERAL)
    P3 = (P2[0] + math.cos(start_yaw)*AHEAD*0.5,
          P2[1] + math.sin(start_yaw)*AHEAD*0.5)

    segA = bezier3(P0, P1, P2, P3, n=35)

    # Segment B: gentle RIGHT to line up toward the goal
    mid_yaw = heading(segA[-3], segA[-1])
    Q0 = segA[-1]
    Q1 = (Q0[0] + math.cos(mid_yaw)*AHEAD*0.4,
          Q0[1] + math.sin(mid_yaw)*AHEAD*0.4)
    # right is -90°
    Q2 = (Q1[0] + math.cos(mid_yaw - math.pi/2.0)*LATERAL,
          Q1[1] + math.sin(mid_yaw - math.pi/2.0)*LATERAL)
    # aim so that the straight segment to goal is short
    STRAIGHT_LEN = 4.0
    tgt_x = gx - math.cos(goal_yaw)*STRAIGHT_LEN
    tgt_y = gy - math.sin(goal_yaw)*STRAIGHT_LEN
    Q3 = (tgt_x, tgt_y)

    segB = bezier3(Q0, Q1, Q2, Q3, n=35)

    # Segment C: short straight to the goal line (possibly offset)
    stop_x = gx - math.cos(goal_yaw)*FINAL_STOP_OFFSET
    stop_y = gy - math.sin(goal_yaw)*FINAL_STOP_OFFSET
    C0 = segB[-1]
    C1 = ( (C0[0] + stop_x)/2.0, (C0[1] + stop_y)/2.0 )
    segC = [C0, C1, (stop_x, stop_y)]

    # Stitch segments, avoid duplicates
    path = segA + segB[1:] + segC[1:]

    # Densify the straight tail for smoother final control
    tail = []
    T = 15
    for i in range(T):
        t = (i+1)/T
        tail.append(( (1-t)*segC[1][0] + t*segC[2][0],
                      (1-t)*segC[1][1] + t*segC[2][1] ))
    path = path[:-1] + tail

    return path

# ======================= Main ===========================
drv = Driver()
dt  = int(drv.getBasicTimeStep())

gps = drv.getDevice("gps") or drv.getGPS("gps")
assert gps is not None, "[IMU_FREE] need a GPS device named 'gps'"
gps.enable(dt)

# Optional cameras if present in world (named dash_cam / top_cam)
for cam_name in ("dash_cam", "top_cam"):
    try:
        cam = drv.getDevice(cam_name)
        cam.enable(dt)
    except Exception:
        pass

drv.setCruisingSpeed(SPEED_CRUISE)

# Let GPS settle
drv.step()
sx, sy, _ = gps.getValues()
start_xy = (sx, sy)

# Estimate initial yaw by peeking one more step
drv.step()
sx2, sy2, _ = gps.getValues()
start_yaw = heading(start_xy, (sx2, sy2))  # small-motion heading

# Build waypoints:
wps = load_xy(CSV_FILE)
if not wps:
    wps = build_procedural_path(start_xy, start_yaw, (GOAL_X, GOAL_Y), GOAL_YAW)

# Make sure we end exactly at the goal (after any offset)
final_goal = (GOAL_X - math.cos(GOAL_YAW)*FINAL_STOP_OFFSET,
              GOAL_Y  - math.sin(GOAL_YAW)*FINAL_STOP_OFFSET)
if dist(wps[-1], final_goal) > 0.05:
    wps.append(final_goal)

print(f"[IMU_FREE] waypoints={len(wps)}  span≈{dist(wps[0], wps[-1]):.1f}m  procedural={'yes' if not os.path.exists(CSV_FILE) else 'no'}")

# Final approach heading is the goal yaw by construction
goal = final_goal
goal_yaw = GOAL_YAW

# Push the logical goal slightly forward into the bay so we stop deeper
goal_bias = (goal[0] + math.cos(goal_yaw)*FORWARD_BIAS,
             goal[1] + math.sin(goal_yaw)*FORWARD_BIAS)

# ----------------- control loop -----------------
idx = 0
prev_pos = start_xy
filt_yaw = start_yaw

def smooth_yaw(pos, yaw, prev):
    dx = pos[0]-prev[0]; dy = pos[1]-prev[1]
    if dx*dx + dy*dy > 0.0004:
        meas = math.atan2(dy, dx)
        d = (meas - yaw + math.pi) % (2*math.pi) - math.pi
        yaw += 0.35*d
    return yaw

while drv.step() != -1:
    x, y, _ = gps.getValues()
    pos = (x, y)
    filt_yaw = smooth_yaw(pos, filt_yaw, prev_pos)
    prev_pos = pos

    # advance waypoint if close
    while idx < len(wps)-1 and dist(pos, wps[idx]) < WAYPOINT_RADIUS:
        idx += 1

    target = wps[idx]
    d_to_target = dist(pos, target)

    # speed schedule (slow near end)
    near_final = idx >= len(wps)-2
    v_cmd = SPEED_CRUISE
    if near_final:
        v_cmd = max(SPEED_MIN, SPEED_CRUISE * clamp(d_to_target/6.0, 0.2, 1.0))
    drv.setCruisingSpeed(v_cmd)

    # steering
    steer = pure_pursuit_steer(x, y, filt_yaw, target)
    drv.setSteeringAngle(steer)

    # stopping condition — aligned & inside bay (or very close)
    aligned = abs((filt_yaw - goal_yaw + math.pi) % (2*math.pi) - math.pi) < ALIGN_TOL
    in_rect  = inside_bay(x, y, goal_bias[0], goal_bias[1], goal_yaw, BAY_LEN/2, BAY_WID/2)
    close    = dist(pos, goal_bias) < STOP_DIST

    if aligned and (in_rect or close):
        drv.setCruisingSpeed(0.0)
        drv.setBrakeIntensity(1.0)
        print("[IMU_FREE] parked — stopped near bay center.")
        break
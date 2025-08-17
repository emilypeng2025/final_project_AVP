# carla_follow_csv.py
import math
import time
import numpy as np
import carla

# ---------- config ----------
CSV_PATH = "self_parking_ai/artifacts/control_points_forward.csv"  # or self_parking_ai/data/...
TOWN = "Town03"

# Anchor: place your parking bay's bottom-left corner at this CARLA Transform
# (Choose a location in Town03; tweak these values after a first run.)
ANCHOR_XY = (80.0, -40.0)      # meters (world)
ANCHOR_YAW_DEG = 0.0           # degrees (0° = +X East in CARLA)
Z_ELEV = 0.1                   # elevate slightly to avoid ground collision

LOOKAHEAD = 3.0                # meters (Pure Pursuit)
WHEELBASE = 2.7                # m (typical compact car)
MAX_STEER_RAD = math.radians(35)
TARGET_SPEED = 1.5             # m/s (slow parking)

# ---------- helpers ----------
def wrap_pi(a):
    return (a + math.pi) % (2*math.pi) - math.pi

def rot2d(x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return x*c - y*s, x*s + y*c

def load_path(csv_path):
    arr = np.loadtxt(csv_path, delimiter=",", skiprows=1)  # x,y,heading_deg
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr[:, :2]  # only x,y

def local_to_world(path_xy, anchor_xy, anchor_yaw_deg):
    yaw = math.radians(anchor_yaw_deg)
    Xw = []
    for x, y in path_xy:
        xr, yr = rot2d(x, y, yaw)
        Xw.append((anchor_xy[0] + xr, anchor_xy[1] + yr))
    return np.array(Xw, dtype=float)

def nearest_index(path_xy, x, y):
    d = np.hypot(path_xy[:,0] - x, path_xy[:,1] - y)
    return int(np.argmin(d))

def pure_pursuit_delta(x, y, yaw, path_xy, lookahead, wheelbase):
    # find lookahead point
    i = nearest_index(path_xy, x, y)
    Lacc = 0.0
    while i < len(path_xy)-1 and Lacc < lookahead:
        seg = np.hypot(path_xy[i+1,0]-path_xy[i,0], path_xy[i+1,1]-path_xy[i,1])
        Lacc += seg
        i += 1
    tx, ty = path_xy[i]
    # transform target into vehicle frame
    dx = tx - x; dy = ty - y
    lx =  math.cos(-yaw)*dx - math.sin(-yaw)*dy
    ly =  math.sin(-yaw)*dx + math.cos(-yaw)*dy
    Ld = max(lookahead, 1e-3)
    kappa = 2.0 * ly / (Ld**2)
    return math.atan(wheelbase * kappa)  # steering angle (rad)

# ---------- main ----------
def main():
    # 1) connect
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.load_world(TOWN)
    world.set_weather(carla.WeatherParameters.ClearNoon)

    # 2) read and map path to world
    path_local = load_path(CSV_PATH)
    path_world = local_to_world(path_local, ANCHOR_XY, ANCHOR_YAW_DEG)

    # 3) spawn vehicle at a point near the start
    blueprints = world.get_blueprint_library()
    bp = blueprints.filter("vehicle.tesla.model3")[0]
    spawn_xy = path_world[0]
    spawn = carla.Transform(
        carla.Location(x=spawn_xy[0], y=spawn_xy[1]-2.0, z=Z_ELEV),
        carla.Rotation(pitch=0.0, yaw=90.0, roll=0.0)  # face +Y to match your planner
    )
    vehicle = world.try_spawn_actor(bp, spawn)
    if vehicle is None:
        raise RuntimeError("Failed to spawn vehicle. Try another anchor pose.")

    # 4) simple control loop
    # keep physics on, we’ll set throttle/steer directly
    dt = 1/20.0
    max_time = 60.0
    elapsed = 0.0

    try:
        while elapsed < max_time:
            t0 = time.time()

            # read current pose
            tf = vehicle.get_transform()
            vel = vehicle.get_velocity()

            x, y = tf.location.x, tf.location.y
            yaw = math.radians(tf.rotation.yaw)
            speed = math.hypot(vel.x, vel.y)

            # lateral control (Pure Pursuit)
            delta = pure_pursuit_delta(x, y, yaw, path_world, LOOKAHEAD, WHEELBASE)
            delta = max(-MAX_STEER_RAD, min(MAX_STEER_RAD, delta))

            # longitudinal: simple P on speed
            e_v = TARGET_SPEED - speed
            throttle = max(0.0, min(0.6, 0.5*e_v))  # clamp
            brake = 0.0 if throttle > 0.05 else min(0.3, -0.5*e_v)

            control = carla.VehicleControl(
                throttle=float(throttle),
                steer=float(delta / MAX_STEER_RAD),  # normalize to [-1,1]
                brake=float(brake)
            )
            vehicle.apply_control(control)

            # stop when close to end
            if np.hypot(x - path_world[-1,0], y - path_world[-1,1]) < 0.3 and speed < 0.2:
                break

            # regulate loop
            time.sleep(max(0.0, dt - (time.time() - t0)))
            elapsed += dt

    finally:
        print("Stopping vehicle and destroying actors…")
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        time.sleep(0.5)
        vehicle.destroy()

if __name__ == "__main__":
    main()

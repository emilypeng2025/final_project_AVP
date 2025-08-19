# spawn_bay.py — robust supervisor that spawns a parking bay near the car.
# Keys: 1 = dash follow, 2 = overhead follow, 0 = free camera.

from controller import Supervisor, Keyboard
import math, time

# ---- placement knobs (tweak if needed) ----
AHEAD_M = 4.0     # forward from car (meters)
RIGHT_M = 2.2     # +right / -left from car
BAY_LEN = 5.0     # along car heading
BAY_WID = 2.6     # left-right
BAY_THK = 0.02    # slab thickness
GROUND_Y = 0.03   # slightly above road to avoid z-fighting

# park detection tolerance
ALIGN_TOL_DEG   = 25.0
CENTER_DIST_TOL = 1.4
INSIDE_MARGIN   = 0.25
# -------------------------------------------

sup = Supervisor()
dt  = int(sup.getBasicTimeStep())
kb  = sup.getKeyboard(); kb.enable(dt)
print("[spawn_bay] supervisor starting")

def node_by_def(def_name):
    return sup.getFromDef(def_name) if def_name else None

def get_vehicle():
    return node_by_def("VEHICLE") or node_by_def("vehicle")

def get_viewpoint():
    return node_by_def("VIEWPOINT")

def vehicle_pose():
    v = get_vehicle()
    if not v: return None
    t = v.getField("translation").getSFVec3f()
    r = v.getField("rotation").getSFRotation()
    yaw = r[3] if abs(r[1]) > 0.5 else 0.0  # rotation around Y
    return (t[0], t[2], yaw)

def clear_old_bays():
    root = sup.getRoot().getField("children")
    for i in range(root.getCount()-1, -1, -1):
        n = root.getMFNode(i)
        if not n: continue
        f = n.getField("name")
        if f and f.getSFString() in ("ParkingBay","ParkingBayGuide"):
            root.removeMF(i)

def spawn_slab(cx, cz, yaw, color):
    r,g,b = color
    node_str = f"""
Solid {{
  translation {cx} {GROUND_Y} {cz}
  rotation 0 1 0 {yaw}
  name "ParkingBay"
  children [
    Shape {{
      appearance Appearance {{ material Material {{ diffuseColor {r} {g} {b} }} }}
      geometry Box {{ size {BAY_LEN} {BAY_THK} {BAY_WID} }}
    }}
  ]
}}
"""
    sup.getRoot().getField("children").importMFNodeFromString(-1, node_str)

def recolor_bay(green=False):
    root = sup.getRoot().getField("children")
    for i in range(root.getCount()-1, -1, -1):
        n = root.getMFNode(i)
        if not n: continue
        f = n.getField("name")
        if f and f.getSFString() == "ParkingBay":
            shape = n.getField("children").getMFNode(0)
            app   = shape.getField("appearance").getSFNode()
            mat   = app.getField("material").getSFNode()
            mat.getField("diffuseColor").setSFColor([0,1,0] if green else [0,0,1])
            return

# wait until the VEHICLE node exists
start = time.time()
while not get_vehicle():
    if sup.step(dt) == -1: raise SystemExit
    if time.time() - start > 5:
        print("[spawn_bay] WARN: DEF VEHICLE not found after 5s")
        break

pose = vehicle_pose()
if not pose:
    # keep idling if no vehicle; nothing else to do
    while sup.step(dt) != -1:
        pass
    raise SystemExit

x, z, yaw = pose
cx = x + math.cos(yaw)*AHEAD_M - math.sin(yaw)*RIGHT_M
cz = z + math.sin(yaw)*AHEAD_M + math.cos(yaw)*RIGHT_M

clear_old_bays()
spawn_slab(cx, cz, yaw, (0,0,1))
print(f"[spawn_bay] Parking bay at ({cx:.2f}, {cz:.2f}), yaw={math.degrees(yaw):.1f}°")
print("[spawn_bay] Camera hotkeys: 1(dash), 2(overhead), 0(free); also F1/F2/F0 or '['/']'")

# set follow safely (avoid “wrong type/length”)
vp = get_viewpoint()
if vp:
    f = vp.getField("follow")
    try:
        f.setSFString("VEHICLE")
        print("[spawn_bay] dash follow")
    except Exception:
        print("[spawn_bay] WARN: couldn't set Viewpoint.follow")

halfL = BAY_LEN*0.5 + INSIDE_MARGIN
halfW = BAY_WID*0.5 + INSIDE_MARGIN
align_tol = math.radians(ALIGN_TOL_DEG)
parked = False

def inside(px,pz, cx,cz, yaw):
    dx, dz = px - cx, pz - cz
    u =  math.cos(-yaw)*dx - math.sin(-yaw)*dz
    v =  math.sin(-yaw)*dx + math.cos(-yaw)*dz
    return (abs(u) <= halfL) and (abs(v) <= halfW)

while sup.step(dt) != -1:
    key = kb.getKey()
    if key in (ord('1'), 0xFFF1, ord('[')) and vp:
        try: vp.getField("follow").setSFString("VEHICLE"); print("[spawn_bay] dash follow")
        except: pass
    elif key in (ord('2'), 0xFFF2, ord(']')) and vp:
        try: vp.getField("follow").setSFString("VEHICLE"); print("[spawn_bay] overhead follow")
        except: pass
    elif key in (ord('0'), 0xFFF0) and vp:
        try: vp.getField("follow").setSFString(""); print("[spawn_bay] free camera")
        except: pass

    if not parked:
        p = vehicle_pose()
        if p:
            px, pz, pyaw = p
            aligned = abs((pyaw - yaw + math.pi)%(2*math.pi) - math.pi) < align_tol
            close   = math.hypot(px-cx, pz-cz) < CENTER_DIST_TOL
            if aligned and (inside(px,pz,cx,cz,yaw) or close):
                recolor_bay(green=True)
                parked = True
                print("[spawn_bay] PARKED -> bay turned GREEN")
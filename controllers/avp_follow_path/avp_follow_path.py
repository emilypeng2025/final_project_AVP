print("[IMU_FREE] controller loaded")  # sanity banner

from vehicle import Driver
from controller import GPS
import csv, os, math

CSV_FILE = os.path.join(os.path.dirname(__file__), "path.csv")
SPEED = 18.0; MAX_STEER = 0.5; WHEELBASE = 2.8
WAYPOINT_RADIUS = 1.5; LOOK_MIN = 3.0; LOOK_MAX = 8.0

def clamp(x,a,b): return max(a, min(b, x))
def dist(a,b): return math.hypot(a[0]-b[0], a[1]-b[1])

def load_pts(fp):
    pts=[]
    if os.path.exists(fp):
        with open(fp) as f:
            for i,row in enumerate(csv.reader(f)):
                if i==0 and any(c.isalpha() for c in ",".join(row)): continue
                if len(row)>=2: pts.append((float(row[0]), float(row[1])))
    return pts or [(0,0),(4,0.3),(7,0.8),(10,1.5)]  # fallback local path

drv = Driver(); dt = int(drv.getBasicTimeStep())
gps = drv.getDevice("gps") or drv.getGPS("gps")
assert gps is not None, "[IMU_FREE] need GPS named 'gps'"
gps.enable(dt); drv.setCruisingSpeed(SPEED)

drv.step()
x0,_,z0 = gps.getValues()
raw = load_pts(CSV_FILE)
xs=[p[0] for p in raw]; ys=[p[1] for p in raw]
span = max(max(xs)-min(xs), max(ys)-min(ys))
wps = [(x0+p[0], z0+p[1]) for p in raw] if span<50 else raw

print(f"[IMU_FREE] waypoints={len(wps)} span={span:.1f}m"); i=0
prev=(x0,z0); yaw=0.0

def upd_yaw(pos, yaw, prev):
    dx=pos[0]-prev[0]; dz=pos[1]-prev[1]
    if dx*dx+dz*dz>0.0004:
        m=math.atan2(dz,dx); d=(m-yaw+math.pi)%(2*math.pi)-math.pi; yaw+=0.3*d
    return yaw

while drv.step()!=-1:
    x,_,z = gps.getValues(); pos=(x,z)
    yaw = upd_yaw(pos, yaw, prev); prev=pos
    while i<len(wps)-1 and dist(pos, wps[i])<WAYPOINT_RADIUS: i+=1
    tgt = wps[i]; d = dist(pos,tgt); L = max(LOOK_MIN, min(LOOK_MAX, d))
    dx=tgt[0]-x; dz=tgt[1]-z
    tx= math.cos(-yaw)*dx - math.sin(-yaw)*dz; tz= math.sin(-yaw)*dx + math.cos(-yaw)*dz
    if tx<0.1: tx=0.1
    curv = 2.0*tz/(L*L); steer = max(-MAX_STEER, min(MAX_STEER, math.atan(curv*WHEELBASE)))
    drv.setSteeringAngle(steer)
    if i>=len(wps)-2 and d<3.0: drv.setCruisingSpeed(max(6.0, SPEED*0.5))
    if i==len(wps)-1 and d<1.0: drv.setCruisingSpeed(0.0); drv.setBrakeIntensity(1.0); print("[IMU_FREE] done"); break

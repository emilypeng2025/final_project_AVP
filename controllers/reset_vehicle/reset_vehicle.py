# reset_vehicle.py — put car back to a known pose once, then exit
from controller import Supervisor

TARGET_TRANSL = [-45.0, 0.05, 2.60]
TARGET_ROT    = [0.0, 1.0, 0.0, 0.0]  # face forward

sup = Supervisor()
dt = int(sup.getBasicTimeStep())

# wait until the car (DEF VEHICLE) exists
for _ in range(200):
    node = sup.getFromDef("VEHICLE")
    if node:
        # set translation & rotation
        node.getField("translation").setSFVec3f(TARGET_TRANSL)
        node.getField("rotation").setSFRotation(TARGET_ROT)
        print("[reset_vehicle] VEHICLE repositioned.")
        break
    sup.step(dt)

# done — stop this controller
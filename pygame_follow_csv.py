import argparse, csv, math, sys
from typing import List, Tuple, Optional
import numpy as np
import pygame

# optional recording (OpenCV)
try:
    import cv2
except Exception:
    cv2 = None

def load_path(csv_file: str) -> np.ndarray:
    pts = []
    with open(csv_file) as f:
        r = csv.reader(f)
        for row in r:
            if not row or row[0].strip().startswith("#"):
                continue
            x = float(row[0]); y = float(row[1])
            yaw = float(row[2]) if len(row) > 2 and row[2] != "" else float("nan")
            pts.append((x, y, yaw))
    if not pts:
        raise ValueError("No points loaded from CSV.")
    return np.array(pts, dtype=float)

def estimate_heading(path: np.ndarray, i: int) -> float:
    if path.shape[1] >= 3 and not math.isnan(path[i,2]):
        return math.radians(path[i,2])
    j = min(i+1, len(path)-1)
    dx = path[j,0] - path[i,0]
    dy = path[j,1] - path[i,1]
    return math.atan2(dy, dx)

def meters_to_screen(path: np.ndarray, scale: float, margin: int = 60):
    xmin, ymin = np.min(path[:,0]), np.min(path[:,1])
    xmax, ymax = np.max(path[:,0]), np.max(path[:,1])
    w = int((xmax - xmin) * scale) + margin*2
    h = int((ymax - ymin) * scale) + margin*2
    w = max(640, w); h = max(480, h)

    def to_px(xm, ym):
        sx = int((xm - xmin) * scale) + margin
        sy = int(h - ((ym - ymin) * scale) - margin)  # invert y for screen coords
        return sx, sy

    return w, h, to_px

def rotated_rect_points(cx, cy, L, W, yaw_rad):
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    hx, hy = L/2, W/2
    corners = [(-hx,-hy),(hx,-hy),(hx,hy),(-hx,hy)]
    return [(c*x - s*y + cx, s*x + c*y + cy) for (x,y) in corners]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="path.csv", help="path points file (x,y[,yaw_deg])")
    ap.add_argument("--scale", type=float, default=60.0, help="pixels per meter")
    ap.add_argument("--fps", type=int, default=30, help="animation FPS")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    ap.add_argument("--carL", type=float, default=4.2, help="car length (m)")
    ap.add_argument("--carW", type=float, default=1.8, help="car width (m)")
    ap.add_argument("--record", default="", help="output mp4 file (requires opencv-python)")
    args = ap.parse_args()

    path = load_path(args.csv)

    pygame.init()
    pygame.display.set_caption("Forward Parking (pygame)")
    w, h, to_px = meters_to_screen(path, args.scale)
    screen = pygame.display.set_mode((w, h))

    path_px = [to_px(x,y) for x,y,_ in path]
    carL_px, carW_px = args.carL * args.scale, args.carW * args.scale

    clock = pygame.time.Clock()
    i = 0
    running = True

    writer = None
    if args.record:
        if cv2 is None:
            print("OpenCV not available; cannot record MP4.", file=sys.stderr)
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.record, fourcc, args.fps, (w, h))

    def draw_frame(idx: int):
        screen.fill((245,245,245))
        # draw parking bay hint near the last point
        ex, ey = to_px(path[-1,0], path[-1,1])
        pygame.draw.rect(screen, (220,230,240), pygame.Rect(ex-40, ey-20, 80, 40), border_radius=6)
        # path
        if len(path_px) >= 2:
            pygame.draw.lines(screen, (120,120,120), False, path_px, 2)
        # car
        x,y,_ = path[idx]
        yaw = estimate_heading(path, idx)
        cx, cy = to_px(x,y)
        pts = rotated_rect_points(cx, cy, carL_px, carW_px, yaw)
        pygame.draw.polygon(screen, (30,144,255), pts)
        # front mark
        fx = (pts[1][0] + pts[2][0]) / 2
        fy = (pts[1][1] + pts[2][1]) / 2
        pygame.draw.circle(screen, (0,0,0), (int(fx), int(fy)), 4)

    last_frame = None
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        draw_frame(i)
        pygame.display.flip()

        if writer is not None:
            surf = pygame.surfarray.array3d(screen)
            frame = surf.swapaxes(0,1)[:, :, ::-1]
            writer.write(frame)
            last_frame = frame

        i += max(1, int(args.speed))
        if i >= len(path):
            if writer is not None and last_frame is not None:
                for _ in range(args.fps):
                    writer.write(last_frame)
            running = False

        clock.tick(args.fps)

    if writer is not None:
        writer.release()
    pygame.quit()
    print("Done.")

if __name__ == "__main__":
    main()

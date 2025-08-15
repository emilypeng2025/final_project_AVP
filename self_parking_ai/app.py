#app.py — Self‑Parking AI Simulator (Forward MVP + MLP suggestion)

import streamlit as st
st.set_page_config(page_title="Self‑Parking AI Simulator", layout="centered")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- MLP strategy recommender ---
try:
    from utils.strategy_predictor import predict_strategy
except Exception as e:
    st.error(
        "Couldn't import utils.strategy_predictor.predict_strategy.\n"
        "Make sure self_parking_ai/utils/strategy_predictor.py exists and is valid.\n\n"
        f"Import error: {e}"
    )
    st.stop()

# --- Path planners (forward required; reverse optional for later) ---
try:
    from planner.paths import plan_and_align_forward
except Exception as e:
    st.error(
        "Couldn't import planner.paths.plan_and_align_forward.\n"
        "Make sure self_parking_ai/planner/paths.py exists and defines plan_and_align_forward().\n\n"
        f"Import error: {e}"
    )
    st.stop()

try:
    from planner.paths import plan_and_align_reverse_parallel
    HAS_REVERSE = True
except Exception:
    HAS_REVERSE = False  # fine for now; we fall back to forward


# ========================= UI =========================
st.title("🚗 Self‑Parking AI Simulator — MVP")

with st.sidebar:
    st.header("Inputs")

    # Parking spot geometry
    spot_length = st.number_input("Spot length (m)", min_value=3.5, max_value=8.0, value=6.2, step=0.1)
    spot_width  = st.number_input("Spot width (m)",  min_value=2.2, max_value=4.0, value=3.0, step=0.1)

    # Car geometry (preset or manual)
    size_preset = st.selectbox("Car size preset", ["Relative: 60% × 55%", "Manual"])
    if size_preset == "Relative: 60% × 55%":
        car_length = spot_length * 0.60
        car_width  = spot_width  * 0.55
        st.caption(f"Preset → Car L×W = {car_length:.2f} × {car_width:.2f} m")
    else:
        car_length = st.number_input("Car length (m)", min_value=3.0, max_value=5.6, value=4.5, step=0.1)
        car_width  = st.number_input("Car width (m)",  min_value=1.5, max_value=2.2, value=1.8, step=0.05)

    # Approach & placement tuning
    distance_to_spot = st.slider("Distance to spot (m)", 0.5, 6.0, 2.0, 0.1)
    inset = st.slider("Rear bumper inset (m)", 0.05, 0.5, 0.10, 0.01)

    # Where the bottom‑left corner of the bay is in the world
    spot_x = st.number_input("Spot origin X (m)", -10.0, 30.0, 15.0, 0.1)
    spot_y = st.number_input("Spot origin Y (m)", -10.0, 30.0,  3.0, 0.1)

    # Predict strategy with your MLP (display only; planning happens on click)
    mlp_strategy, confidence, scores = predict_strategy(
        car_length, car_width, spot_length, spot_width, distance_to_spot, verbose=False
    )
    st.markdown(f"### Suggested Strategy: **{mlp_strategy}**")
    st.caption(f"Confidence: {confidence:.1%}")

    # Let user optionally override
    manual_choice = st.selectbox(
        "Force strategy? (optional)",
        ["Use MLP suggestion", "Forward Perpendicular", "Reverse Parallel", "Cannot Park"]
    )

    run_button = st.button("Plan Path")

# World-space bay origin
spot_origin = np.array([spot_x, spot_y], dtype=float)


# ========================= Run =========================
if run_button:
    # Decide which strategy to run (manual override or MLP)
    strategy = mlp_strategy if manual_choice == "Use MLP suggestion" else manual_choice

    # Choose the planner function
    if strategy == "Forward Perpendicular":
        plan = plan_and_align_forward(
            car_length=car_length,
            car_width=car_width,
            spot_length=spot_length,
            spot_width=spot_width,
            spot_origin=spot_origin,
            distance_to_spot=distance_to_spot,
            inset=inset
        )
    elif strategy == "Reverse Parallel":
        if HAS_REVERSE:
            plan = plan_and_align_reverse_parallel(
                car_length=car_length,
                car_width=car_width,
                spot_length=spot_length,
                spot_width=spot_width,
                spot_origin=spot_origin,
                distance_to_spot=distance_to_spot,
                inset=inset
            )
        else:
            st.warning("Reverse planner not implemented yet. Falling back to Forward.")
            plan = plan_and_align_forward(
                car_length=car_length,
                car_width=car_width,
                spot_length=spot_length,
                spot_width=spot_width,
                spot_origin=spot_origin,
                distance_to_spot=distance_to_spot,
                inset=inset
            )
    else:  # "Cannot Park"
        st.error("This car may not fit into the given spot.")
        st.stop()

    # ---------------- Plot ----------------
    curve_points = np.asarray(plan["curve_points"], dtype=float)
    rear_target  = np.asarray(plan["rear_target"], dtype=float)

    if curve_points is None:
        st.error("No feasible path returned.")
    else:
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
        ax.scatter(rear_target[0], rear_target[1], s=80, marker='x', label='Rear Target')
        rb = np.asarray(plan.get("rear_bumper_after", rear_target), dtype=float)
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
        st.write(f"- Strategy used: **{strategy}** (MLP suggested: **{mlp_strategy}**, conf **{confidence:.1%}**)")
        st.write(f"- Entry side (auto): **{plan['entry_side']}**")
        st.write(f"- Final heading: **{plan['yaw_target']:.1f}°**")
        st.write(f"- Car (L×W): **{car_length:.2f} × {car_width:.2f} m**")
        st.write(f"- Spot (L×W): **{spot_length:.2f} × {spot_width:.2f} m**")
        st.write(f"- Inset: **{inset:.2f} m**, Distance: **{distance_to_spot:.2f} m**")
else:
    st.info("Set inputs in the sidebar and click **Plan Path**.")

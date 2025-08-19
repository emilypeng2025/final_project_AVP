# 🚗 AVP: Self-Parking AI Simulator (Python • Streamlit • Light LLM • Webots) - README

## 📌 Project Overview

 Goal: 
 Show how traditional planning/control + lightweight ML/LLM components can  work together to simulate self-parking — from decision to execution to explanation.

This 2-week final project demonstrates an Autonomous Valet Parking (AVP) system, covering strategy, planning, control, and visualization across three tracks:

	1.	Colab → MLP strategy recommender + multi-segment Bezier planner with animations (forward & reverse parking).

	2.	Streamlit → interactive UI with car/spot inputs, PID tuning, and optional LLM explanations for human-readable reasoning.

	3.	Webots → live 3D simulation of forward parking into a custom bay, with multi-camera views and different car models.

The system simulates how an AI could make parking decisions and execute them safely.

Core capabilities include:
	•	Accepting user inputs (car size, parking spot info) via Streamlit.
	•	Recommending a parking strategy (forward, reverse, or no-fit) using an MLP classifier.
	•	Planning a smooth trajectory with Bezier curves or arc/straight segments.
	•	Controlling the vehicle with PID or replaying the planned path.
	•	Visualizing results with plots, animations, screenshots, and simulation videos.

---

## 🧱 System Architecture

```text
[User Input: Car size + Spot info]
            ⬇
[MLP Classifier: Recommend parking strategy]
            ⬇
[Trajectory Planner: Bezier multi‑segment path]  
            ⬇
[Vehicle Controller: PID follow]
            ⬇
[Sim: Colab animation / Streamlit UI / Webots world]
```

---

## 🧠 Modules Breakdown

### 1. Car Strategy Recommender (MLP Classifier)

**Input Features:**

* Car length, width
* Parking spot length, width
* Angle or distance to spot

**Output Classes:**

* `0` = Reverse Parallel
* `1` = Forward Perpendicular
* `2` = Cannot Park

**Model Structure:**

```python
nn.Sequential(
    nn.Linear(5, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 3)
)
```

### 2. Trajectory Planner

* Use Bezier curves to generate a smooth parking path
*	Multi-segment planning (arc + straight + S-curve) explored in Colab / 多段轨迹（弧线 + 直线 + S 曲线）
* Not implemented in Webots due to time limits → in Webots, the vehicle only goes straight into the ba
* (Future work) Learned planners such as imitation learning could be added for more adaptive behaviors

### 3. Controller

* PID control for steering & speed, tuned via Streamlit sliders
* Parameters kp, ki, kd tuned to reduce oscillations / 参数 kp, ki, kd 用于减少震荡,稳定运动
* (Future work) Reinforcement learning (e.g., DQN, PPO) could replace PID for more robust policies

### 4. Simulation/ 仿真
* Colab → Animated plots, forward + reverse parking paths, exported GIFs / Colab → 动画演示,，前进+倒车泊车路径，导出 GIF/MP4
* Streamlit → Interactive UI with car/spot inputs, PID tuning, and LLM explanation;
Reverse strategy not yet integrated here due to time constraints / Streamlit
* Webots → Live 3D sim with custom bay, multi-camera views, car selection / Webots → 三维仿真，支持自定义车位、多视角摄像机和车型选择
Bezier-based curves were not integrated — car only drives straight in
* (Optional, future) CARLA → replay with multiple cars and sensor simulation 

Due to time limit, some features are missing, priority was to deliver a working pipeline across Colab, Streamlit, and Webots. 
	•	Streamlit → focused on forward perpendicular parking; reverse parallel can be added in the future.
	•	Webots → Bezier curves were prototyped in Colab, but not integrated; only straight-in parking is shown.
	•	Advanced planners/controllers → imitation learning and RL were left for future improvement due to scope.

---

## ⚙️ Installation

### Requirements

	•	Python 3.9–3.11 (tested on 3.10)
	•	PyTorch (for the MLP strategy classifier)
	•	Streamlit (for the interactive UI)
	•	matplotlib, numpy, pandas (for plotting & data handling)
	•	Webots (for 3D simulation)
	•	(Optional) CARLA Simulator (for replay/sensor simulation — not fully integrated)


### What's used for the 3 main tracks:

	•	Colab: no install needed (runs in browser).
	•	Streamlit: needs its own requirements_streamlit.txt.
	•	Webots: needs Webots installed locally (no pip install).

---

1. Clone repository / 克隆项目
git clone https://github.com/emilypeng2025/final_project_AVP.git
cd final_project_AVP

2. Set up environment / 创建虚拟环境

Mac/Linux
python3 -m venv .venv
source .venv/bin/activate

Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

3. Install dependencies / 安装依赖
A. For Streamlit demo
cd self_parking_ai
pip install -r requirements_streamlit.txt

B. For Webots controllers

Install Webots (GUI-based simulator).
Your controllers are inside controllers/, and scenes are in worlds/.

C. (Optional) CARLA simulator (Linux)

If you want to explore CARLA integration:
git clone https://github.com/carla-simulator/carla.git
cd carla
make PythonAPI

## 🚀 How to Run
1. Colab (no local install)
	•	Open notebooks/ in Colab.
	•	Run MLP_strategy.ipynb → trains the MLP strategy classifier.
	•	Run path_planner.ipynb → generates multi-segment Bezier trajectories and exports GIF/MP4.

2. Streamlit UI

From inside self_parking_ai/:
streamlit run app.py

Then open http://localhost:8501 in your browser.
	•	Adjust car/spot size and PID params in the sidebar.
	•	Generate trajectory → download CSV.
	•	(Optional) Click Explain with LLM for human-readable reasoning.

3. Webots Simulation
	•	Open Webots.
	•	Load the world from worlds/city.wbt.
	•	Run controller:
	•	avp_free → forward parking (straight in).
	•	avp_follow_path → CSV-following (planned trajectory).

4. (Optional) CARLA replay

Run carla_follow_csv.py to test CSV-based replay inside CARLA.
(Currently a skeleton — not fully integrated.)

⸻


## Quickstart

A. Run the Streamlit demo
# from inside final_project_AVP/self_parking_ai
python -m venv ../venv
source ../venv/bin/activate   # Windows: ..\venv\Scripts\activate
pip install -r requirements_streamlit.txt
streamlit run app.py


➡️ Then open http://localhost:8501 in your browser.
	•	Adjust car & spot size in sidebar
	•	Tune PID (kp, ki, kd)
	•	Generate trajectory → download CSV
	•	(Optional) Click Explain with LLM

B. Try the Colab notebooks
	•	Open self_parking_ai/notebooks/MLP_strategy.ipynb → train MLP classifier (forward vs reverse vs no-fit).
	•	Open self_parking_ai/notebooks/path_planner.ipynb → generate multi-segment Bezier trajectories, export GIF/MP4.

⸻

C. Run Webots simulation
	•	Install Webots
	•	Open worlds/city.wbt
	•	Choose controller:
	•	avp_free → forward parking (straight-in)
	•	avp_follow_path → CSV-following (from Streamlit planner)

⸻

D. (Optional) CARLA replay (Linux only)
python carla_follow_csv.py
(skeleton script — shows how planned CSV could be replayed in CARLA)

---

## 📊 Visualization

What’s included
	•	Vehicle path — Bezier multi-segment trajectories (forward & reverse)
	•	Steering & speed over time — PID follow (optional)
	•	Strategy confidence (MLP) — classifier probabilities for each strategy
	•	Simulation views — Webots live demo (forward parking into bay)

Where to see them
	•	Colab notebooks (self_parking_ai/notebooks/)
	•	Animated forward & reverse paths
	•	Exported GIF/MP4
	•	MLP confidence visualization
	•	Streamlit app (self_parking_ai/app.py)
	•	Live plot of planned path, car footprint, and waypoints
	•	PID time series (steer/heading error) when enabled
	•	Webots (worlds/*.wbt)
	•	Live during presentation (multi-camera view + straight-in forward parking)

Exported artifacts
	•	CSV control points → self_parking_ai/artifacts/
	•	Presentation media (screenshots, GIFs, MP4s) → self_parking_ai/presentation_day/

Tech stack
	•	Matplotlib → trajectory & plots
	•	Streamlit → interactive visualization
	•	Webots → live 3D simulation demo
	•	(Optional) pygame → quick 2D CSV replay

Notes
	•	Reverse path is visualized in Colab, but not integrated into Webots due to time constraints.
	•	Webots demo shows straight-in forward parking (Bezier prototype stayed in Colab).

---

## 📁 Folder Structure

```
final_project_AVP/
├── README.md                      # ← this file
├── requirements.txt               # (for Webots / conda export, keep as is)
├── path.csv                       # trajectory points exported by planner (x, y, heading,…)
├── pygame_follow_csv.py           # simple 2D replay of saved path
├── carla_follow_csv.py            # (optional) CARLA replay script skeleton
├── controllers/                   # Webots controllers
│   ├── avp_free/                  # forward-parking controller (currently used)
│   ├── avp_follow_path/           # CSV-tracking controller (future integration)
│   ├── reset_vehicle/             # reset vehicle pose in sim
│   └── spawn_bay/                 # generate parking bay objects
├── worlds/                        # Webots world files (*.wbt)
├── self_parking_ai/               # ← main project code lives here
│   ├── app.py                     # Streamlit app (UI + LLM explanation + PID tuning)
│   ├── requirements_streamlit.txt # Streamlit-only dependencies (isolated from root reqs)
│   ├── planner/                   # Bezier curves & geometry helpers
│   ├── controller/                # PID and Pure-Pursuit logic (optional)
│   ├── models/                    # MLP model weights + scaler.pkl
│   ├── data/                      # training / sample data
│   ├── notebooks/                 # Colab notebooks (MLP strategy, multi-segment path, etc.)
│   └── utils/                     # helper functions: I/O, plotting, CSV export
└── snapshots/                     # exported GIFs / MP4s / screenshots (for presentation)
```

---

## 📚 References

* [CARLA Simulator](https://carla.org/)
* [Apollo Auto Open Source](https://github.com/ApolloAuto/apollo)
* [PythonRobotics by Atsushi Sakai](https://github.com/AtsushiSakai/PythonRobotics)

---

## ✅ Next Steps

This project delivered a working pipeline across Colab, Streamlit, and Webots. Some features were left out due to time limits but can be added in future iterations:
	•	Streamlit
	  •	Add reverse parallel strategy (currently only forward parking is executed).
	  •	Extend LLM explanations with more detailed reasoning (multi-step breakdown).
	•	Webots
	  •	Integrate Bezier-based curved trajectories (current demo uses straight-in forward parking only).
	  •	Add sensor simulation (lidar/camera fusion) to better mimic AVP systems.
	  •	Support CSV-following controller for replay of planned paths.
	•	Controller
	  •	Explore Reinforcement Learning (DQN, PPO) to replace PID for robustness.
	  •	Test with disturbance/noise for more realistic handling.
	•	CARLA (Optional)
	  •	Replay planned CSVs with multiple car types.
	  •	Integrate camera/sensor data for perception-driven parking.
	•	General
	  •	Improve visualization: steering/time plots in Streamlit, smoother Colab animations.
	  •	Expand dataset for MLP strategy classifier (different car/spot ratios, angles).

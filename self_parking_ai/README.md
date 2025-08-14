# 🚗 Self-Parking AI Simulator - README

## 📌 Project Overview


Build a self-parking simulation system using Python, Streamlit and matplotlib/pygame/p5/if time permitted, even CARLA (or other simulation environments), which can:

* Accept user inputs (car size, parking space info) -Streamlit(if time permitted import car types with a drop down option with data import)
* Recommend a parking strategy (forward, reverse, or no fit)
* Plan a trajectory using traditional or learned methods
* Control the vehicle to complete the maneuver
* Optionally visualize the process

---

## 🧱 System Architecture

```text
[User Input: Car size + Spot info]
            ⬇
[MLP Classifier: Recommend parking strategy]
            ⬇
[Trajectory Planner: Bezier / RRT*]  
            ⬇
[Vehicle Controller: PID or RL-based]
            ⬇
[CARLA sim executes parking]
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

* Use Bezier curves (or Hybrid A\*) to generate a smooth parking path
* Optionally use learned planning (e.g., imitation learning)

### 3. Controller

* Basic PID control (speed, steering)
* Advanced: Use reinforcement learning (DQN, PPO)

### 4. Simulation

* Use CARLA to:

  * Load parking scenes
  * Control vehicles
  * Simulate sensor data

---

## ⚙️ Installation

### Requirements

* Python 3.8+
* PyTorch
* CARLA Simulator
* matplotlib, numpy, pandas

### Install CARLA (Linux)

```bash
git clone https://github.com/carla-simulator/carla.git
cd carla
make PythonAPI
```

---

## 🚀 How to Run

### 1. Train the MLP model

```bash
python train_mlp_classifier.py
```

### 2. Run the simulator

```bash
python run_simulation.py
```

---

## 📊 Visualization

* Vehicle path
* Steering angles
* Decision confidence from MLP
* Matplotlib/P5/pygame
---

## 📁 Folder Structure

```
self_parking_ai/
├── data/                  # Simulated training data
├── models/                # Saved MLP model weights
├── planner/               # Path planning algorithms (Bezier, A*)
├── controller/            # PID or RL control logic
├── carla_client/          # CARLA interface code
├── configs/               # Car and parking space settings
├── visualization/         # Plotting tools
├── train_mlp_classifier.py
├── run_simulation.py
└── README.md
```

---

## 📚 References

* [CARLA Simulator](https://carla.org/)
* [Apollo Auto Open Source](https://github.com/ApolloAuto/apollo)
* [PythonRobotics by Atsushi Sakai](https://github.com/AtsushiSakai/PythonRobotics)

---

## ✅ Next Steps

* [ ] Build and train the MLP model
* [ ] Integrate trajectory planner
* [ ] Visualize parking paths
* [ ] Test in CARLA simulation
* [ ] Optional: Add UI to enter car & spot dimensions

Feel free to fork, contribute, and adapt for your own self-parking AI project 🚘

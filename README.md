# DroneGuard AI — Anti-Jamming Autonomous Drone System

> **LSTM-powered autonomous drone intelligence for GPS/RC jamming detection and recovery.**  
> Real-time MAVLink integration | ArduPilot/PX4 compatible | Raspberry Pi deployment ready

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![TensorFlow 2.13+](https://img.shields.io/badge/TensorFlow-2.13%2B-orange?logo=tensorflow)](https://tensorflow.org)
[![MAVLink](https://img.shields.io/badge/MAVLink-2.0-red)](https://mavlink.io)
[![ArduPilot](https://img.shields.io/badge/ArduPilot-SITL%20Ready-green)](https://ardupilot.org)
[![License](https://img.shields.io/badge/License-Custom%20(See%20Below)-important)](#license)

---

## What This Project Does

DroneGuard AI trains an LSTM neural network on realistic drone sensor data (IMU, GPS, barometer, RC signal) to detect GPS/RF jamming in real time and autonomously choose the safest response — hover, return home, or emergency land — without any human input or RC link.

**Key capabilities:**

- Detects GPS jamming, RC signal loss, and spoofing signatures from 21 onboard sensor features
- Makes autonomous decisions at 50 Hz with sub-20 ms latency on a Raspberry Pi 4
- Integrates directly with ArduPilot/PX4 via MAVLink (serial, TCP, UDP)
- Includes a full 30-second mission simulator with curved waypoints and gradual jamming ramps
- Exports to TFLite int8 for edge deployment; full Keras model for training
- Streamlit dashboard for live visualization of AI decisions and flight telemetry

---

## Author & Attribution

**Created by Burhan Ali**  
GitHub: [@Burhanali2211](https://github.com/Burhanali2211)  
Email: easyio.tech@gmail.com

> This project was designed and built from scratch as original research work.  
> **Permission is required** before using, adapting, or redistributing any part of this codebase. See [License](#license).

---

## Table of Contents

1. [Architecture](#architecture)
2. [Project Structure](#project-structure)
3. [Quick Start](#quick-start)
4. [Installation](#installation)
5. [Usage — All Entry Points](#usage--all-entry-points)
6. [AI Model Design](#ai-model-design)
7. [Sensor Features (21 inputs)](#sensor-features-21-inputs)
8. [Action Classes](#action-classes)
9. [Safety Systems](#safety-systems)
10. [SITL Testing Guide](#sitl-testing-guide)
11. [Raspberry Pi Deployment](#raspberry-pi-deployment)
12. [Config Reference](#config-reference)
13. [Hardware Compatibility](#hardware-compatibility)
14. [License](#license)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      DroneGuard AI                          │
│                                                             │
│  Sensors (21 features)                                      │
│  ┌──────────────┐    ┌──────────────────────────────────┐  │
│  │ MAVLink      │───▶│ DroneSensorSimulator  (sim mode) │  │
│  │ Bridge       │    │ or MAVLinkBridge      (real HW)  │  │
│  └──────────────┘    └─────────────┬────────────────────┘  │
│                                    │                        │
│  AI Decision Engine                ▼                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  LSTM(32) → Dense(64) → Dense(32) → Softmax(4)     │   │
│  │  + Confidence hysteresis (threshold=0.72, vote=3)   │   │
│  └───────────────────────┬─────────────────────────────┘   │
│                           │  action_id ∈ {0,1,2,3}         │
│  Safety Layer             ▼                                 │
│  ┌──────────────┐    ┌──────────────────────────────────┐  │
│  │  Watchdog    │───▶│  FlightStateMachine              │  │
│  │  (battery,   │    │  PREFLIGHT→TAKEOFF→CRUISE→HOVER  │  │
│  │   GPS, jam,  │    │  →RETURN_HOME→LANDING→LANDED     │  │
│  │   geofence,  │    │  EMERGENCY_LAND (always reachable)│  │
│  │   heartbeat) │    └──────────────┬───────────────────┘  │
│  └──────────────┘                   │                       │
│                                     ▼                       │
│  Actuation                 ┌────────────────┐              │
│  ┌─────────────────────┐   │ ActionExecutor │              │
│  │ MAVLink commands    │◀──│ (MAV_CMD_*)    │              │
│  │ velocity setpoints  │   └────────────────┘              │
│  └─────────────────────┘                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
DroneSimulation Project NIT/
│
├── main.py                  # Train + evaluate (interactive)
├── train.py                 # Headless training script
├── simulate.py              # Full mission simulation
├── run_realtime.py          # 50 Hz real-time flight brain (production)
├── app.py                   # Streamlit live dashboard
├── config.yaml              # All tunable constants (single source of truth)
├── requirements.txt
│
├── src/
│   ├── model/
│   │   ├── drone_ai.py          # DroneAI class: LSTM build, train, predict
│   │   └── data_generator.py    # Synthetic training data with noise profiles
│   │
│   ├── simulator/
│   │   ├── sensor_simulator.py  # Physics-based sensor sim (MPU-6050, GPS, RC)
│   │   └── mission_simulator.py # 30s mission: takeoff→fly→hover→RTH→land
│   │
│   ├── control/
│   │   ├── mavlink_bridge.py    # Real FC interface (serial/tcp/udp + sim fallback)
│   │   ├── state_machine.py     # Legal phase transition graph
│   │   ├── actuation.py         # action_id → MAVLink commands
│   │   └── watchdog.py          # Heartbeat + battery/GPS/jam/geofence override
│   │
│   ├── inference/
│   │   └── tflite_runner.py     # TFLite int8 wrapper for RPi (same interface)
│   │
│   ├── visualization/
│   │   └── plotter.py           # Matplotlib training + confusion matrix plots
│   │
│   └── config_loader.py         # Singleton YAML config loader
│
├── models/                  # Saved model artifacts (git-ignored except .keras/.tflite)
│   ├── drone_ai_model.keras
│   └── drone_ai_model.tflite
│
├── data/                    # Generated training data (git-ignored)
├── logs/                    # Flight logs CSV (git-ignored)
└── outputs/                 # Plot exports (git-ignored)
```

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/Burhanali2211/DroneAiRTH.git
cd "DroneAiRTH"
pip install -r requirements.txt

# 2. Train the AI model
python train.py

# 3. Run the full mission simulation
python simulate.py

# 4. (Optional) Launch the browser dashboard
streamlit run app.py
```

---

## Installation

### Prerequisites

- Python 3.10 or newer
- pip 23+

### Standard (laptop / server)

```bash
pip install -r requirements.txt
```

### Raspberry Pi (edge deployment)

```bash
# Use tflite-runtime instead of full TensorFlow
pip install numpy pandas scikit-learn pymavlink PyYAML
pip install tflite-runtime        # lightweight, no TF dependency
```

Then uncomment the `tflite-runtime` line in `requirements.txt` and comment out `tensorflow`.

---

## Usage — All Entry Points

### `train.py` — Headless training

```bash
python train.py
```

Generates synthetic training data → trains LSTM → saves `.keras` + `.tflite` + scaler to `models/`.  
All hyperparameters are controlled by `config.yaml`.

---

### `main.py` — Interactive train + evaluate

```bash
python main.py
```

Trains the model, then launches an interactive menu:
- View training curves and confusion matrix
- Run single predictions on synthetic sensor data
- Export to TFLite

---

### `simulate.py` — Full mission simulation

```bash
python simulate.py
```

Runs a 30-second mission with:
- Takeoff (0–2s) → fly curved path to target (2–8s) → hover (8–12s)
- GPS/RC jamming starts at 6s, lasts 12s
- AI makes autonomous decisions every 20 ms during jamming
- Returns home and lands (12–30s)
- Prints per-second AI decision log and mission summary
- Saves flight CSV to `logs/`

---

### `run_realtime.py` — 50 Hz production flight brain

```bash
# Simulation mode (no hardware)
python run_realtime.py

# ArduPilot SITL via TCP
python run_realtime.py --mode tcp --host 127.0.0.1 --mav_port 5760

# Real hardware via serial (Raspberry Pi)
python run_realtime.py --mode serial --port /dev/ttyAMA0 --baud 57600

# Use TFLite model (recommended on RPi)
python run_realtime.py --mode serial --port /dev/ttyAMA0 --tflite

# Custom log directory and loop rate
python run_realtime.py --log /mnt/usb/logs --hz 50.0
```

**CLI flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `simulation` | Connection mode: `simulation`, `tcp`, `udp`, `serial` |
| `--host` | `127.0.0.1` | FC host (TCP/UDP) |
| `--port` | `/dev/ttyAMA0` | Serial port |
| `--baud` | `57600` | Serial baud rate |
| `--mav_port` | `5760` | TCP/UDP port |
| `--tflite` | off | Use TFLite runner (RPi) |
| `--log` | `logs/` | Flight log directory |
| `--hz` | `50.0` | Loop frequency |

---

### `app.py` — Streamlit dashboard

```bash
streamlit run app.py
```

Opens a browser UI at `http://localhost:8501` with:
- Live 3D trajectory animation (Plotly)
- AI decision probabilities over time
- Battery, altitude, GPS quality plots
- Mission summary statistics

---

## AI Model Design

### Architecture

```
Input: (batch, window=10, features=21)
  └─ LSTM(32, return_sequences=False)
      └─ Dense(64, activation='relu')
          └─ Dense(32, activation='relu')
              └─ Dense(4, activation='softmax')
```

**Why LSTM?**  
Jamming detection requires temporal context — a single sensor snapshot doesn't distinguish a momentary GPS glitch from sustained jamming. The LSTM window captures the trend over the last 10 timesteps (200 ms at 50 Hz).

### Confidence Hysteresis

Prevents rapid action flipping under noisy conditions:

```python
CONF_THRESHOLD = 0.72     # minimum confidence to accept AI's choice
VOTE_WINDOW    = 3        # consecutive agreeing predictions required
```

EMERGENCY_LAND (action 3) bypasses the vote window and activates at confidence > 0.60 — safety over stability.

### Training Data

Synthetic data generated by `DroneSensorSimulator` with realistic hardware noise:

| Sensor | Noise Model |
|--------|------------|
| MPU-6050 accel | ±0.05 m/s² Gaussian |
| MPU-6050 gyro | ±0.1 deg/s Gaussian |
| BMP280 baro | ±0.08 m Gaussian |
| u-blox M8N GPS | HDOP wander 0.15/s |
| FrSky RSSI | std 3.2 dBm |

Jamming is simulated as a gradual ramp (`min(1.0, elapsed / (jam_dur * 0.3))`) — not a binary flip — matching real-world RF environment degradation.

---

## Sensor Features (21 inputs)

| # | Feature | Source | Description |
|---|---------|--------|-------------|
| 1 | `accel_x` | IMU | X-axis acceleration (m/s²) |
| 2 | `accel_y` | IMU | Y-axis acceleration (m/s²) |
| 3 | `accel_z` | IMU | Z-axis acceleration (m/s²) |
| 4 | `gyro_x` | IMU | Roll rate (deg/s) |
| 5 | `gyro_y` | IMU | Pitch rate (deg/s) |
| 6 | `gyro_z` | IMU | Yaw rate (deg/s) |
| 7 | `compass` | Magnetometer | Heading (0–360°) |
| 8 | `altitude` | Barometer/GPS | Relative altitude (m) |
| 9 | `pos_x` | GPS | East position from home (m) |
| 10 | `pos_y` | GPS | North position from home (m) |
| 11 | `vel_x` | GPS | East velocity (m/s) |
| 12 | `vel_y` | GPS | North velocity (m/s) |
| 13 | `dist_to_target` | Computed | Euclidean distance to mission target (m) |
| 14 | `dist_to_home` | Computed | Euclidean distance to home (m) |
| 15 | `battery` | FC | Battery remaining (%) |
| 16 | `pitch` | IMU | Pitch angle (deg) |
| 17 | `roll` | IMU | Roll angle (deg) |
| 18 | `rc_rssi` | RC receiver | RC signal strength (dBm) |
| 19 | `gps_fix` | GPS | Fix quality (0=none, 1=3D fix) |
| 20 | `gps_hdop` | GPS | Horizontal dilution of precision |
| 21 | `jam_noise` | Derived | Jamming index = 0.5×RSSI_norm + 0.5×HDOP_norm |

---

## Action Classes

| ID | Name | MAVLink Command | Trigger Condition |
|----|------|----------------|------------------|
| 0 | `CONTINUE_TO_TARGET` | SET_POSITION_TARGET velocity | Normal flight |
| 1 | `HOVER_AT_TARGET` | Zero velocity setpoint | Temporary hold |
| 2 | `RETURN_HOME` | MAV_CMD_NAV_RETURN_TO_LAUNCH | Jamming / low battery / GPS lost |
| 3 | `EMERGENCY_LAND` | Controlled descent (0.4 m/s) | Critical battery / heartbeat loss |

---

## Safety Systems

### Watchdog (override priority order)

| Priority | Condition | Action |
|----------|-----------|--------|
| 1 (highest) | Battery < 10% | EMERGENCY_LAND |
| 2 | Battery < 20% | RETURN_HOME |
| 3 | GPS fix == 0 | RETURN_HOME |
| 4 | Jam noise > 0.85 | RETURN_HOME |
| 5 | Distance from home > 15m | RETURN_HOME (geofence) |
| 6 | Heartbeat timeout > 3s | force_land() |

The watchdog runs in a background thread at 1 Hz sending MAVLink heartbeats to the FC. If the GCS link is lost, ArduPilot triggers its own failsafe via `FS_GCS_ENABLE=1`.

### Flight State Machine

```
PREFLIGHT → TAKEOFF → CRUISE → HOVER
                              ↓
                        RETURN_HOME → LANDING → LANDED
                              ↓
                        EMERGENCY_LAND → LANDED
```

Illegal transitions are silently rejected — the AI cannot command `LAND` from `CRUISE` at 30m altitude without first transitioning through allowed states.

### ActionExecutor Safety Guard

- EMERGENCY_LAND above 8m altitude → automatically demoted to RETURN_HOME
- Velocity commands capped at 2.0 m/s
- All commands require GUIDED mode (set automatically before each command)

---

## SITL Testing Guide

Test the full AI pipeline without physical hardware using ArduPilot SITL.

### 1. Install ArduPilot SITL

```bash
# Ubuntu / WSL2
git clone https://github.com/ArduPilot/ardupilot.git
cd ardupilot
git submodule update --init --recursive
Tools/environment_install/install-prereqs-ubuntu.sh -y
. ~/.profile
```

### 2. Launch SITL

```bash
cd ardupilot/ArduCopter
sim_vehicle.py -v ArduCopter --console --map
# Listens on TCP port 5760 by default
```

### 3. Set failsafe parameters

In the SITL console or Mission Planner:
```
param set FS_GCS_ENABLE 1
param set FS_GCS_TIMEOUT 1500
param set GUIDED_SPEED_BGD 2.0
```

### 4. Connect DroneGuard AI

```bash
python run_realtime.py --mode tcp --host 127.0.0.1 --mav_port 5760
```

The system will:
1. Connect to SITL and request 50 Hz data streams
2. Record GPS home position
3. Start the 50 Hz AI loop
4. Arm and takeoff via MAVLink

---

## Raspberry Pi Deployment

### Tested hardware

- Raspberry Pi 4B (4 GB recommended)
- Pixhawk 4 / Cube Orange FC via UART (`/dev/ttyAMA0`)
- Or any ArduPilot-compatible FC

### Setup

```bash
# On RPi (Raspberry Pi OS Bookworm 64-bit)
pip install numpy pandas scikit-learn pymavlink PyYAML tflite-runtime

# Copy project (excluding heavy deps)
rsync -av --exclude='venv' --exclude='data' \
  "DroneSimulation Project NIT/" pi@raspberrypi.local:~/droneguard/

# On RPi — run with TFLite
cd ~/droneguard
python run_realtime.py --mode serial --port /dev/ttyAMA0 --baud 57600 --tflite
```

### Performance

| Hardware | Inference latency | Loop budget (50 Hz) |
|----------|------------------|---------------------|
| Laptop (TF Keras) | ~4 ms | 20 ms ✓ |
| RPi 4B (TFLite int8) | ~8 ms | 20 ms ✓ |
| RPi Zero 2W (TFLite) | ~18 ms | 20 ms ✓ (tight) |

---

## Config Reference

All tunable parameters live in `config.yaml`. No magic numbers in code.

```yaml
model:
  conf_threshold: 0.72      # Minimum AI confidence to accept a decision
  vote_window: 3            # Consecutive votes needed to change action
  sequence_window: 10       # LSTM input window (timesteps)
  battery_critical: 10.0    # % → EMERGENCY_LAND
  battery_low: 20.0         # % → RETURN_HOME
  jam_high: 0.85            # jam_noise threshold → RETURN_HOME

mission:
  target: [3.0, 2.0]        # Mission target [x, y] metres from home
  home: [0.0, 0.0]          # Home origin
  cruise_alt: 1.2           # Default cruise altitude (m)
  max_speed: 2.0            # Velocity cap (m/s)
  max_land_alt: 8.0         # Above this → demote LAND to RTH (m)
  max_range_from_home: 15.0 # Geofence radius (m)
  descent_rate: 0.4         # Emergency landing descent speed (m/s)

simulation:
  duration: 30.0            # Total mission length (s)
  dt: 0.02                  # Physics timestep (s)
  jam_start: 6.0            # Jamming onset time (s)
  jam_dur: 12.0             # Jamming duration (s)
  seed: 7                   # Reproducibility seed

training:
  num_samples: 10000        # Synthetic training samples
  epochs: 60
  batch_size: 64
  window: 10                # Must match model.sequence_window
  seed: 42
```

---

## Hardware Compatibility

| Component | Supported Models |
|-----------|-----------------|
| Flight Controller | Pixhawk 1/2/4/6, Cube Orange/Blue, Matek H743 (ArduPilot/PX4) |
| Companion Computer | Raspberry Pi 3B+, 4B, 5, Jetson Nano |
| GPS | u-blox M8N, M9N, ZED-F9P |
| RC System | FrSky (SBUS/RSSI), TBS Crossfire, ExpressLRS |
| Telemetry | SiK 433/915 MHz, RFD900, WiFi UDP |
| Connection to FC | UART serial, TCP (SITL/companion), UDP (GCS) |

---

## How It Works — Step by Step

1. **Sensor ingestion**: `MAVLinkBridge` reads 7 MAVLink message types at 50 Hz and maps them to a consistent 21-feature dict. In simulation mode, `DroneSensorSimulator` generates physics-based synthetic data with realistic noise.

2. **Feature scaling**: `StandardScaler` (fitted on training data, saved as `models/scaler.json`) normalizes all 21 features to zero mean and unit variance before LSTM inference.

3. **LSTM inference**: A sliding window of the last 10 timesteps is fed to the LSTM. The output is a 4-class probability distribution over actions.

4. **Confidence hysteresis**: If the top prediction doesn't meet the threshold or hasn't been consistent for 3 consecutive frames, the previous action is held. This prevents oscillation at decision boundaries.

5. **Watchdog override**: Regardless of AI output, critical sensor conditions (battery, GPS, jamming, geofence) directly force the appropriate action.

6. **State machine validation**: The FlightStateMachine ensures only legal phase transitions happen. An AI that hallucinates a LAND command at 30m altitude will be rejected.

7. **Actuation**: `ActionExecutor` translates the validated action_id into specific MAVLink commands and sends them to the FC.

8. **Logging**: Every decision, sensor reading, confidence score, and probability is logged to a timestamped CSV in `logs/` for post-flight analysis.

---

## License

Copyright © 2024 Burhan Ali ([@Burhanali2211](https://github.com/Burhanali2211))

**All rights reserved.**

This software and all associated files are the intellectual property of Burhan Ali. You may **not** use, copy, modify, distribute, sublicense, or deploy any part of this codebase — in whole or in part — without explicit written permission from the author.

To request permission, contact: **easyio.tech@gmail.com**

---

## Contributing

This project is not open for public contributions at this time. If you have found a bug or have a feature request, open an issue and the author will review it.

---

## Keywords

`drone AI` · `anti-jamming drone` · `GPS jamming detection` · `autonomous drone` · `LSTM drone control` · `MAVLink Python` · `ArduPilot AI` · `drone GPS spoofing` · `RF jamming detection` · `Raspberry Pi drone` · `autonomous UAV` · `drone machine learning` · `PX4 autonomous` · `drone return to home AI` · `drone failsafe system` · `PyMAVLink` · `TFLite drone` · `drone neural network` · `UAV anti-jam` · `GPS denied navigation`

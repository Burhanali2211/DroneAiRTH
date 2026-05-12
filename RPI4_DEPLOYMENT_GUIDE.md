# Raspberry Pi 4 Deployment Guide
## DroneGuard AI Anti-Jamming System

---

## 1. OS CHOICE & SETUP

### Recommended: Raspberry Pi OS Bookworm 64-bit

**Why this OS?**
- Native Python 3.10+ (no workarounds)
- Lightweight (~2GB used vs Ubuntu ~4GB)
- tflite-runtime support (critical for RPi)
- GPIO direct access for serial wiring
- Official RPi kernel optimized for Pi 4

**Download & Burn:**
```bash
# Windows: Use Raspberry Pi Imager (official tool)
# https://www.raspberrypi.com/software/

# Settings during burn:
# - Hostname: droneguard
# - Username: pi (default)
# - Password: set something strong
# - WiFi: enable + add credentials
# - SSH: ENABLE (critical for remote access)
# - Locale: your region
```

**After first boot (on RPi):**
```bash
# Expand filesystem to full SD card
sudo raspi-config
# → Storage → Expand Filesystem → Reboot

# Update system
sudo apt update
sudo apt upgrade -y

# Enable serial UART (for Pixhawk connection)
sudo raspi-config
# → Interface Options → Serial Port
# → Login shell over serial: NO
# → Serial interface: YES
# → Reboot

# Verify UART enabled
ls -la /dev/ttyAMA0  # Should exist
```

**From your laptop (SSH access):**
```bash
# Find RPi IP (check router or use hostname)
ping droneguard.local

# SSH in
ssh pi@droneguard.local

# Test Python
python3 --version  # Should be 3.11+
```

---

## 2. WIRING DIAGRAM (Pixhawk → RPi GPIO)

### Connection Map

```
PIXHAWK 4 TELEMETRY 2 PORT
┌─────────────────────────────┐
│ Pin 1: +5V    (red)         │
│ Pin 2: TX     (white)  ────┐│
│ Pin 3: RX     (black)  ──┐ ││
│ Pin 4: CTS    (yellow) ──┼─┼─ (optional, leave unconnected)
│ Pin 5: RTS    (green)  ──┼─┼─ (optional, leave unconnected)
│ Pin 6: GND    (brown)  ──┐│ │
└──────────────────────────────┘
                           │ │
                           └─┼─ RPi Pin 10 (RX) - GPIO 15
                             └─ RPi Pin 08 (TX) - GPIO 14
                             
                           GND ─ RPi Pin 06 (GND)
```

### Raspberry Pi GPIO Header (top view)
```
RPi 4 GPIO PINS (40-pin header)
┌─────────────────────────────────────┐
│  1(3.3V)  2(5V)                     │
│  3(SDA)   4(5V)                     │
│  5(SCL)   6(GND)  ◄── CONNECT HERE  │
│  7(GPIO4) 8(TX)   ◄── PIXHAWK TX    │
│  9(GND)  10(RX)   ◄── PIXHAWK RX    │
│ 11-40: ...other GPIO...             │
└─────────────────────────────────────┘
```

### Step-by-step wiring
1. **Power OFF both** Pixhawk and RPi
2. Connect GND (Pin 6) on RPi to GND on Pixhawk (brown wire)
3. Connect Pixhawk TX (white) to RPi RX/Pin 10 (GPIO 15)
4. Connect Pixhawk RX (black) to RPi TX/Pin 8 (GPIO 14)
5. **Do NOT connect +5V** (RPi powers itself; Pixhawk via separate battery/power module)
6. Double-check: use continuity tester if available
7. Power on Pixhawk first, then RPi

### Test connection (on RPi after boot)
```bash
# Check serial port is accessible
ls -la /dev/ttyAMA0

# Monitor serial traffic (debug only)
cat /dev/ttyAMA0  # Ctrl+C to exit

# Or use pyserial test
python3 << 'EOF'
import serial
try:
    ser = serial.Serial('/dev/ttyAMA0', 57600, timeout=1)
    print(f"✓ Serial port open: {ser.name}")
    ser.close()
except Exception as e:
    print(f"✗ Error: {e}")
EOF
```

---

## 3. TRAINING THE MODEL

### Option A: Train on Laptop (Recommended)

**Step 1: Install full TensorFlow environment (on laptop, NOT RPi)**
```bash
cd "DroneSimulation Project NIT"

# Create virtual env
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install full training dependencies
pip install -r requirements.txt

# Verify TensorFlow installed
python -c "import tensorflow; print(tensorflow.__version__)"
```

**Step 2: Generate training data & train**
```bash
python train.py
```

This creates:
- `models/drone_ai_model.keras` (full model, ~1.2 MB)
- `models/drone_ai_model.tflite` (quantized int8, ~320 KB) ◄── USE THIS ON RPi
- `models/scaler.json` (feature normalization, critical)
- `models/training_history.png` (loss curves)

**Typical times:**
- Data gen: 2–3 min
- Training: 3–5 min (50 epochs on laptop GPU)
- Export to TFLite: 30 sec

**Step 3: Copy to RPi**
```bash
# From laptop
scp models/drone_ai_model.tflite pi@droneguard.local:~/droneguard/models/
scp models/scaler.json pi@droneguard.local:~/droneguard/models/
```

### Option B: Train on RPi (Not recommended; very slow)
If you must: install full TensorFlow on RPi (~30 min download, likely to hit memory limits). Better to train on any laptop/cloud and transfer.

---

## 4. INSTALLATION ON RPi

### Step 1: Lightweight dependencies
```bash
# SSH into RPi
ssh pi@droneguard.local

# Install RPi-optimized deps (no TensorFlow, no visualization)
pip install -r requirements_rpi.txt

# Verify
python3 -c "import tflite_runtime; print('✓ TFLite ready')"
```

### Step 2: Copy project to RPi
```bash
# From laptop
scp -r "DroneSimulation Project NIT" pi@droneguard.local:~/droneguard

# Or use rsync (faster for large dirs)
rsync -av --exclude='venv' --exclude='data' --exclude='.git' \
  "DroneSimulation Project NIT/" pi@droneguard.local:~/droneguard/
```

### Step 3: Verify on RPi
```bash
ssh pi@droneguard.local

cd ~/droneguard

# Check project structure
ls -la
# Should see: main.py, run_realtime.py, config.yaml, src/, models/

# Check models exist
ls -la models/
# Should see: drone_ai_model.tflite, scaler.json

# Test imports
python3 << 'EOF'
from src.inference.tflite_runner import TFLiteRunner
from src.config_loader import ConfigLoader
print("✓ All imports OK")
EOF
```

---

## 5. TESTING SEQUENCE

### Test 1: Simulation (no hardware needed)
```bash
# On laptop or RPi
python simulate.py

# Should output:
# - 30-second mission log (takeoff → fly → jam → RTH → land)
# - AI decision log per second
# - CSV saved to logs/mission_*.csv
```

### Test 2: SITL on laptop (ArduPilot simulation)
```bash
# Terminal 1 (laptop): Start SITL
cd ardupilot/ArduCopter
sim_vehicle.py -v ArduCopter --console --map
# Waits for connection; listens on TCP:5760

# Terminal 2 (laptop): Connect DroneGuard
python run_realtime.py --mode tcp --host 127.0.0.1 --mav_port 5760

# Expected: Arming, takeoff, flying, jamming triggered, RTH, landing
# Watch console for "Decision: HOVER_AT_TARGET", "Decision: RETURN_HOME", etc.
```

### Test 3: SITL from RPi (via TCP to laptop SITL)
```bash
# Terminal 1 (laptop): Start SITL (same as above)
cd ardupilot/ArduCopter
sim_vehicle.py -v ArduCopter --console --map

# Terminal 2 (RPi via SSH):
ssh pi@droneguard.local
cd ~/droneguard

# Find laptop IP (e.g., 192.168.1.100)
python run_realtime.py --mode tcp --host 192.168.1.100 --mav_port 5760

# Verifies RPi can reach laptop SITL and inference works at 50 Hz
```

### Test 4: Real Pixhawk via serial (final test)
```bash
# On RPi
ssh pi@droneguard.local
cd ~/droneguard

# Run with TFLite (required on RPi)
python run_realtime.py --mode serial --port /dev/ttyAMA0 --baud 57600 --tflite

# Expected:
# - "Connecting to Pixhawk..." → "Connected!"
# - Heartbeat acknowledgments
# - Sensor readings printed (accel, gyro, GPS)
# - Ready for arming
```

---

## 6. CONFIGURATION CHECKLIST

Before flying, verify `config.yaml`:

```yaml
model:
  conf_threshold: 0.72      # Min confidence to accept AI decision
  vote_window: 3            # Consecutive frames needed
  battery_critical: 10.0    # % → EMERGENCY_LAND
  battery_low: 20.0         # % → RETURN_HOME
  jam_high: 0.85            # Jamming threshold → RTH

mission:
  max_range_from_home: 15.0 # Geofence (m) — SET THIS!
  cruise_alt: 1.2           # Takeoff altitude
  max_speed: 2.0            # Velocity cap (m/s)
  descent_rate: 0.4         # Emergency land speed (m/s)

simulation:
  jam_start: 6.0            # When jamming begins in sim (s)
  jam_dur: 12.0             # How long jamming lasts (s)
```

### Pixhawk parameter requirements
Set these in Mission Planner or Pixhawk console:
```
GUIDED_SPEED_BGD 2.0        # Guided mode background speed
FS_GCS_ENABLE 1             # GCS failsafe enabled
FS_GCS_TIMEOUT 1500         # Timeout 1.5s before failsafe
SERIAL1_PROTOCOL 1          # MAVLink on Serial1
SERIAL1_BAUD 57             # 57600 baud (57 × 100)
```

---

## 7. TROUBLESHOOTING

### Serial port not found
```bash
# Check dmesg for errors
dmesg | grep ttyAMA0

# Enable UART in raspi-config
sudo raspi-config → Interface → Serial → YES

# Reboot
sudo reboot
```

### TFLite runtime not found
```bash
# RPi should have tflite-runtime, NOT tensorflow
pip uninstall tensorflow -y
pip install tflite-runtime

# Verify
python3 -c "import tflite_runtime; print(tflite_runtime.version.VERSION)"
```

### Pixhawk not detected
```bash
# Check baud rate matches config.yaml
# Verify UART wiring (GND, TX, RX)
# Try manual test:
python3 << 'EOF'
import serial
ser = serial.Serial('/dev/ttyAMA0', 57600, timeout=1)
ser.write(b'\xfe\x03\x00\x00\x01')  # MAVLink heartbeat
print(ser.read(100))  # Should see response
EOF
```

### Inference latency too high
```bash
# Check /usr/bin/python3 is 64-bit
file /usr/bin/python3

# If 32-bit, reinstall Bookworm 64-bit
# Verify TFLite int8 (not float32)
ls -lh models/drone_ai_model.tflite
# Should be <400 KB for int8

# Monitor CPU during inference
ssh pi@droneguard.local
htop  # Watch %CPU and load avg
```

---

## 8. MILESTONE CHECKLIST

- [ ] **Week 1:** RPi OS burned, SSH working, serial UART enabled
- [ ] **Week 1:** Laptop training complete, `.tflite` + `scaler.json` on RPi
- [ ] **Week 2:** `simulate.py` runs on laptop
- [ ] **Week 2:** SITL works on laptop (ArduCopter + DroneGuard)
- [ ] **Week 2:** SITL works from RPi via TCP
- [ ] **Week 3:** Pixhawk wired to RPi, serial test passes
- [ ] **Week 3:** `run_realtime.py --mode serial` connects to real Pixhawk
- [ ] **Week 4:** Arm, takeoff, hover, land (manual RC, no jamming)
- [ ] **Week 4:** Arm, takeoff, trigger jamming, watch RTH work
- [ ] **Week 4:** Test EMERGENCY_LAND by pulling battery to 5%

---

## 9. NEXT IMMEDIATE STEPS

1. **Get Bookworm 64-bit image** → Burn to SD (30 min)
2. **Boot RPi, enable UART** via `raspi-config` (10 min)
3. **Train model on laptop** → `python train.py` (5 min)
4. **Copy to RPi** → `scp models/* pi@droneguard.local:~/droneguard/models/` (1 min)
5. **Install tflite-runtime on RPi** (5 min)
6. **Run simulate.py** on laptop to verify model works (2 min)

**Total time to verified-ready state: ~1 hour**

Then test SITL → real hardware in order.

---

## 10. CONTACT & SUPPORT

- Code issues: Debug with `run_realtime.py --verbose` (if added)
- Hardware issues: Check `/dev/ttyAMA0` permissions and baud rates
- Model issues: Verify `scaler.json` exists in `models/`
- Pixhawk comms: Monitor with `mavproxy.py --master=/dev/ttyAMA0 --baudrate=57600`

---

**Ready to start? Pick step 1 above and report back.**

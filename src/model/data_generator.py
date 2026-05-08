"""
data_generator.py — Balanced 4-class training data with real hardware noise profiles.

Noise parameters sourced from datasheets:
  - MPU-6050 IMU:    accel ±0.05 m/s², gyro ±0.1 deg/s, bias drift 0.003 m/s²/s
  - BMP280 baro:     altitude ±0.08 m
  - u-blox M8N GPS:  HDOP wander 0.15/s, fix loss under jamming
  - FrSky RSSI:      noise std 3.2 dBm, range -40 to -120 dBm
  - LiPo battery:    voltage sag under load, non-linear discharge curve
  - RF jamming:      gradual degradation, not binary flip

Generates SEQUENCE windows (window=10, features=21) for LSTM input.
Also generates flat (N, 21) arrays for backward compatibility with old tests.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque

from src.simulator.sensor_simulator import DroneSensorSimulator
from src.config_loader import get_config


def _target() -> np.ndarray:
    try:
        return np.array(get_config()['mission']['target'], dtype=np.float32)
    except Exception:
        return np.array([3.0, 2.0], dtype=np.float32)


TARGET = _target()   # module-level alias; resolved after config loads

# ── Real hardware noise constants (datasheet-sourced) ─────────────────────────
IMU_ACCEL_NOISE  = 0.05     # m/s²       MPU-6050 spec
IMU_GYRO_NOISE   = 0.10     # deg/s      MPU-6050 spec
IMU_BIAS_DRIFT   = 0.003    # m/s²/s     MPU-6050 in-run bias
BARO_NOISE       = 0.08     # m          BMP280 spec
GPS_HDOP_WANDER  = 0.15     # HDOP/s     u-blox M8N open sky
RSSI_NOISE_STD   = 3.2      # dBm        FrSky R-XSR telemetry
COMPASS_NOISE    = 1.5      # deg        HMC5883L spec
BATT_NOISE       = 0.3      # %          ADC + voltage divider noise

# Jamming degrades gradually over 0.5-2s, not instantly
JAM_RAMP_STEPS   = 25       # timesteps over which jamming ramps up


def _apply_real_noise(sensor: dict, rng: np.random.Generator,
                      bias: dict, jam_ramp: float = 1.0) -> dict:
    """
    Apply hardware-realistic noise on top of simulated sensor values.
    bias: persistent per-sensor bias (drifts slowly, not white noise)
    jam_ramp: 0.0-1.0, how far jamming has ramped up (gradual degradation)
    """
    s = sensor.copy()

    # IMU — white noise + bias drift
    s['accel_x'] += rng.normal(0, IMU_ACCEL_NOISE) + bias['accel_x']
    s['accel_y'] += rng.normal(0, IMU_ACCEL_NOISE) + bias['accel_y']
    s['accel_z'] += rng.normal(0, IMU_ACCEL_NOISE * 2)
    s['gyro_x']  += rng.normal(0, IMU_GYRO_NOISE)  + bias['gyro_x']
    s['gyro_y']  += rng.normal(0, IMU_GYRO_NOISE)  + bias['gyro_y']
    s['gyro_z']  += rng.normal(0, IMU_GYRO_NOISE)

    # Barometer altitude noise
    s['altitude'] = max(0.0, s['altitude'] + rng.normal(0, BARO_NOISE))

    # Compass noise
    s['compass'] = (s['compass'] + rng.normal(0, COMPASS_NOISE)) % 360

    # Battery — load sag under high current + ADC noise
    load_sag = 0.5 if abs(s['vel_x']) + abs(s['vel_y']) > 1.0 else 0.0
    s['battery'] = max(0.0, s['battery'] - load_sag + rng.normal(0, BATT_NOISE))

    # RSSI — gradual degradation under jamming, not binary
    if jam_ramp > 0:
        # Partial jamming: RSSI degrades proportionally
        clean_rssi  = rng.normal(-62, RSSI_NOISE_STD)
        jammed_rssi = rng.normal(-115, RSSI_NOISE_STD * 2)
        s['rc_rssi'] = clean_rssi * (1 - jam_ramp) + jammed_rssi * jam_ramp
        # GPS HDOP degrades with jamming
        s['gps_hdop'] = 1.2 + jam_ramp * 97.8 + rng.normal(0, GPS_HDOP_WANDER)
        s['gps_fix']  = float(rng.random() > jam_ramp * 0.9)  # probabilistic loss
        s['jam_noise'] = jam_ramp + rng.normal(0, 0.05)
    else:
        s['rc_rssi']  = rng.normal(-62, RSSI_NOISE_STD)
        s['gps_hdop'] = max(0.8, 1.2 + rng.normal(0, GPS_HDOP_WANDER))
        s['gps_fix']  = 1.0
        s['jam_noise'] = abs(rng.normal(0, 0.02))

    return s


def _random_bias(rng: np.random.Generator) -> dict:
    """Per-flight persistent bias (simulates sensor calibration error)."""
    return {
        'accel_x': rng.normal(0, 0.02),
        'accel_y': rng.normal(0, 0.02),
        'gyro_x':  rng.normal(0, 0.05),
        'gyro_y':  rng.normal(0, 0.05),
    }


def _make_sequence(sim: DroneSensorSimulator, rng: np.random.Generator,
                   bias: dict, window: int = 10,
                   jam_ramp_start: float = 0.0,
                   jam_ramp_end: float = 1.0) -> np.ndarray:
    """
    Build a (window, n_features) sequence by stepping the simulator
    forward `window` ticks, applying real noise at each step.
    jam_ramp_start/end: allow gradual onset of jamming within window.
    """
    frames = []
    for i in range(window):
        ramp = jam_ramp_start + (jam_ramp_end - jam_ramp_start) * (i / window)
        raw  = sim.get_sensor_data()
        noisy = _apply_real_noise(raw, rng, bias, jam_ramp=ramp)
        frames.append(list(noisy.values()))
        sim.time += 0.02   # 50 Hz tick

    return np.array(frames, dtype=np.float32)   # (window, features)


def generate_training_data(num_samples: int = 10000,
                            window: int = 10,
                            seed: int = 42,
                            cache_dir: str = 'data') -> tuple:
    """
    Generate a balanced 4-class dataset with real hardware noise profiles.
    Each sample is a (window, n_features) sequence for LSTM input.

    Classes
    -------
    0  CONTINUE_TO_TARGET  — en-route, good signal, battery 40-100%
    1  HOVER_AT_TARGET     — near target, near-zero velocity
    2  RETURN_HOME         — jammed (gradual onset), velocity toward home
    3  EMERGENCY_LAND      — critical battery < 17%, may or may not be jammed

    Returns
    -------
    X_seq  : (N, window, features)  float32  — LSTM input
    X_flat : (N, features)          float32  — last frame, for flat model compat
    y      : (N,)                   int32
    feature_names : list[str]
    """
    cache = Path(cache_dir)
    cache.mkdir(exist_ok=True)

    print("[DATA] Generating training data with real hardware noise profiles...")

    rng = np.random.default_rng(seed)
    samples_per_class = num_samples // 4
    seq_list, flat_list, y_list = [], [], []

    def make_sim(jammed=False, jam_ramp=1.0):
        s = DroneSensorSimulator(seed=int(rng.integers(0, 1_000_000)))
        s.is_jammed        = jammed
        s.jamming_strength = jam_ramp
        return s

    # ── Class 0: CONTINUE_TO_TARGET ──────────────────────────────────────────
    for _ in range(samples_per_class):
        jammed = rng.random() < 0.2          # mostly clean signal
        ramp   = float(rng.uniform(0, 0.4)) if jammed else 0.0
        sim    = make_sim(jammed, ramp)
        sim.position[:2] = rng.uniform([0.1, 0.1], [2.3, 1.6])
        sim.position[2]  = float(rng.uniform(1.0, 1.5))
        sim.battery      = float(rng.uniform(40.0, 100.0))
        to_target        = TARGET - sim.position[:2]
        unit             = to_target / (np.linalg.norm(to_target) + 1e-6)
        sim.velocity[:2] = unit * float(rng.uniform(0.5, 1.5))
        sim.pitch        = float(rng.uniform(8, 20))
        bias             = _random_bias(rng)
        seq  = _make_sequence(sim, rng, bias, window, 0.0, ramp)
        seq_list.append(seq)
        flat_list.append(seq[-1])
        y_list.append(0)

    # ── Class 1: HOVER_AT_TARGET ──────────────────────────────────────────────
    for _ in range(samples_per_class):
        jammed = rng.random() < 0.4
        ramp   = float(rng.uniform(0, 0.6)) if jammed else 0.0
        sim    = make_sim(jammed, ramp)
        sim.position[:2] = TARGET + rng.uniform(-0.25, 0.25, size=2)
        sim.position[2]  = float(rng.uniform(1.2, 1.5))
        sim.battery      = float(rng.uniform(30.0, 100.0))
        sim.velocity[:2] = rng.uniform(-0.08, 0.08, size=2)
        sim.pitch        = float(rng.uniform(-1.5, 1.5))
        bias             = _random_bias(rng)
        seq  = _make_sequence(sim, rng, bias, window, 0.0, ramp)
        seq_list.append(seq)
        flat_list.append(seq[-1])
        y_list.append(1)

    # ── Class 2: RETURN_HOME ─────────────────────────────────────────────────
    for _ in range(samples_per_class):
        # Jamming onset is gradual — randomly pick ramp start within window
        ramp_start = float(rng.uniform(0.0, 0.5))
        ramp_end   = float(rng.uniform(0.7, 1.0))
        sim = make_sim(jammed=True, jam_ramp=ramp_end)
        sim.position[:2] = rng.uniform([0.4, 0.2], [3.8, 2.8])
        sim.position[2]  = float(rng.uniform(1.0, 1.5))
        sim.battery      = float(rng.uniform(20.0, 60.0))
        to_home          = -sim.position[:2]
        unit             = to_home / (np.linalg.norm(to_home) + 1e-6)
        sim.velocity[:2] = unit * float(rng.uniform(0.5, 1.5))
        sim.pitch        = float(rng.uniform(-20, -6))
        bias             = _random_bias(rng)
        seq  = _make_sequence(sim, rng, bias, window, ramp_start, ramp_end)
        seq_list.append(seq)
        flat_list.append(seq[-1])
        y_list.append(2)

    # ── Class 3: EMERGENCY_LAND ───────────────────────────────────────────────
    for _ in range(samples_per_class):
        jammed = rng.random() < 0.7
        ramp   = float(rng.uniform(0.5, 1.0)) if jammed else 0.0
        sim    = make_sim(jammed, ramp)
        sim.position[:2] = rng.uniform([-0.5, -0.5], [4.0, 3.0])
        sim.position[2]  = float(rng.uniform(0.0, 1.5))
        sim.battery      = float(rng.uniform(0.0, 17.0))
        sim.velocity[:2] = rng.uniform(-0.5, 0.5, size=2)
        bias             = _random_bias(rng)
        seq  = _make_sequence(sim, rng, bias, window, 0.0, ramp)
        seq_list.append(seq)
        flat_list.append(seq[-1])
        y_list.append(3)

    X_seq  = np.array(seq_list,  dtype=np.float32)   # (N, window, features)
    X_flat = np.array(flat_list, dtype=np.float32)   # (N, features)
    y      = np.array(y_list,    dtype=np.int32)

    # Feature names from one sample dict
    _tmp = DroneSensorSimulator(seed=0)
    feature_names = list(_tmp.get_sensor_data().keys())

    # Cache
    np.save(str(cache / 'X_seq.npy'),  X_seq)
    np.save(str(cache / 'X_flat.npy'), X_flat)
    np.save(str(cache / 'X.npy'),      X_flat)   # backward compat
    np.save(str(cache / 'y.npy'),      y)
    (cache / 'feature_names.txt').write_text('\n'.join(feature_names))
    (cache / 'window_size.txt').write_text(str(window))

    print(f"[DATA] {len(X_seq)} samples | window={window} | dist: {np.bincount(y)}")
    print(f"[DATA] Cached to {cache_dir}/")
    return X_seq, X_flat, y, feature_names


def load_cached(cache_dir: str = 'data') -> tuple | None:
    """Return (X_seq, X_flat, y, feature_names) from cache, or None if no cache."""
    cache = Path(cache_dir)
    required = ['X_seq.npy', 'X_flat.npy', 'y.npy', 'feature_names.txt', 'window_size.txt']
    missing  = [f for f in required if not (cache / f).exists()]

    if len(missing) == len(required):
        return None   # no cache at all — regenerate silently

    if missing:
        raise FileNotFoundError(
            f"[DATA] Partial cache in {cache_dir}/ — missing: {missing}. "
            f"Delete the data/ folder and re-run to regenerate cleanly."
        )

    X_seq  = np.load(str(cache / 'X_seq.npy'))
    X_flat = np.load(str(cache / 'X_flat.npy'))
    y      = np.load(str(cache / 'y.npy'))
    fn     = (cache / 'feature_names.txt').read_text().splitlines()
    win    = int((cache / 'window_size.txt').read_text().strip())

    # Validate feature count matches saved metadata
    if X_seq.shape[2] != len(fn):
        raise ValueError(
            f"[DATA] Cache mismatch: X_seq has {X_seq.shape[2]} features "
            f"but feature_names.txt has {len(fn)}. Delete data/ and regenerate."
        )

    print(f"[DATA] Loaded cached sequences from {cache_dir}/ (window={win}, "
          f"samples={len(X_seq)}, features={len(fn)})")
    return X_seq, X_flat, y, fn

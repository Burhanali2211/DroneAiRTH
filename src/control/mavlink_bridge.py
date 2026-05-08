"""
mavlink_bridge.py -Real sensor reader via MAVLink (Pixhawk/ArduPilot/PX4).

Supports three connection modes:
  - Serial (UART on RPi):  MAVLinkBridge('serial', port='/dev/ttyAMA0', baud=57600)
  - TCP (SITL on laptop):  MAVLinkBridge('tcp',    host='127.0.0.1',   port=5760)
  - UDP (telemetry radio): MAVLinkBridge('udp',    host='0.0.0.0',     port=14550)

In SIMULATION mode (default if pymavlink not installed), returns sensor dict
from the existing DroneSensorSimulator -so the entire pipeline runs without
hardware connected.

Real hardware flow:
  bridge = MAVLinkBridge('tcp', host='127.0.0.1', port=5760)  # SITL
  bridge.connect()
  sensors = bridge.get_sensor_dict()    # returns same 21-key dict as simulator
  bridge.close()
"""
import time
import numpy as np
from typing import Optional


# Try importing pymavlink -graceful fallback to simulation mode
try:
    from pymavlink import mavutil
    MAVLINK_AVAILABLE = True
except ImportError:
    MAVLINK_AVAILABLE = False


class MAVLinkBridge:
    """
    Reads all 21 sensor features from a real flight controller via MAVLink,
    returning the same dict format as DroneSensorSimulator.get_sensor_data().

    This ensures the AI model receives identically-structured input whether
    running in simulation or on real hardware.
    """

    # MAVLink message IDs we subscribe to
    REQUIRED_MSGS = [
        'ATTITUDE',            # roll, pitch, yaw, rates
        'GLOBAL_POSITION_INT', # lat, lon, alt, vx, vy, vz
        'GPS_RAW_INT',         # fix_type, hdop
        'RC_CHANNELS',         # rssi channel
        'BATTERY_STATUS',      # remaining %, voltage
        'SCALED_IMU2',         # raw accel + gyro (backup)
        'VFR_HUD',             # airspeed, groundspeed, heading, alt
    ]

    def __init__(self, mode: str = 'simulation',
                 port: str = '/dev/ttyAMA0',
                 host: str = '127.0.0.1',
                 baud: int = 57600,
                 mav_port: int = 5760,
                 home: Optional[np.ndarray] = None,
                 target: Optional[np.ndarray] = None):
        """
        mode: 'serial' | 'tcp' | 'udp' | 'simulation'
        home:   [x, y] metres from launch point (default: origin)
        target: [x, y] mission target in metres from home
        """
        self.mode      = mode
        self.port      = port
        self.host      = host
        self.baud      = baud
        self.mav_port  = mav_port
        self.home      = home   if home   is not None else np.array([0.0, 0.0])
        self.target    = target if target is not None else np.array([3.0, 2.0])

        self._conn        = None
        self._sim         = None       # fallback simulator
        self._cache       = {}         # latest value per message type
        self._connected   = False
        self._home_gps    = None       # (lat, lon) at arm time, set on connect

        if mode == 'simulation' or not MAVLINK_AVAILABLE:
            self._init_simulation()

    # ── Connection ────────────────────────────────────────────────────────────

    def _init_simulation(self):
        from src.simulator.sensor_simulator import DroneSensorSimulator
        self._sim = DroneSensorSimulator(seed=42)
        self._connected = True
        print("[BRIDGE] Simulation mode -using DroneSensorSimulator")

    def connect(self, timeout: float = 10.0):
        if self._sim is not None:
            return   # already in sim mode

        if not MAVLINK_AVAILABLE:
            print("[BRIDGE] pymavlink not installed -falling back to simulation")
            self._init_simulation()
            return

        print(f"[BRIDGE] Connecting via {self.mode}...")
        try:
            if self.mode == 'serial':
                self._conn = mavutil.mavlink_connection(
                    self.port, baud=self.baud)
            elif self.mode == 'tcp':
                self._conn = mavutil.mavlink_connection(
                    f'tcp:{self.host}:{self.mav_port}')
            elif self.mode == 'udp':
                self._conn = mavutil.mavlink_connection(
                    f'udp:{self.host}:{self.mav_port}')
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

            # Wait for heartbeat
            print("[BRIDGE] Waiting for heartbeat...")
            self._conn.wait_heartbeat(timeout=timeout)
            self._connected = True
            print(f"[BRIDGE] Connected! System {self._conn.target_system} "
                  f"Component {self._conn.target_component}")

            # Request data streams at 50 Hz
            self._request_streams(rate=50)
            # Record home position
            self._capture_home()

        except Exception as e:
            print(f"[BRIDGE] Connection failed: {e} -falling back to simulation")
            self._init_simulation()

    def _request_streams(self, rate: int = 50):
        """Ask the FC to send all sensor streams at the given rate."""
        if self._conn is None:
            return
        self._conn.mav.request_data_stream_send(
            self._conn.target_system,
            self._conn.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL,
            rate, 1)

    def _capture_home(self):
        """Record GPS position at connection time as home origin."""
        if self._conn is None:
            return
        msg = self._conn.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=5)
        if msg:
            self._home_gps = (msg.lat / 1e7, msg.lon / 1e7)
            print(f"[BRIDGE] Home GPS: {self._home_gps}")

    def close(self):
        if self._conn:
            self._conn.close()
        self._connected = False

    # ── Sensor reading ─────────────────────────────────────────────────────────

    def get_sensor_dict(self) -> dict:
        """
        Returns the same 21-key dict as DroneSensorSimulator.get_sensor_data().
        In simulation mode: delegates to the simulator.
        In real mode: reads latest MAVLink messages.
        """
        if self._sim is not None:
            return self._sim.get_sensor_data()

        return self._read_mavlink_sensors()

    def _read_mavlink_sensors(self) -> dict:
        """Parse live MAVLink messages into the 21-feature dict."""
        # Drain available messages into cache (non-blocking)
        for _ in range(20):
            msg = self._conn.recv_match(blocking=False)
            if msg is None:
                break
            self._cache[msg.get_type()] = msg

        att  = self._cache.get('ATTITUDE')
        pos  = self._cache.get('GLOBAL_POSITION_INT')
        gps  = self._cache.get('GPS_RAW_INT')
        rc   = self._cache.get('RC_CHANNELS')
        bat  = self._cache.get('BATTERY_STATUS')
        imu  = self._cache.get('SCALED_IMU2')
        hud  = self._cache.get('VFR_HUD')

        # ── IMU ──────────────────────────────────────────────────────────────
        if imu:
            # IMU values from FC are in milli-g and milli-deg/s
            accel_x = imu.xacc  / 1000.0 * 9.81
            accel_y = imu.yacc  / 1000.0 * 9.81
            accel_z = imu.zacc  / 1000.0 * 9.81
            gyro_x  = imu.xgyro / 1000.0
            gyro_y  = imu.ygyro / 1000.0
            gyro_z  = imu.zgyro / 1000.0
        else:
            accel_x = accel_y = gyro_x = gyro_y = gyro_z = 0.0
            accel_z = 9.81

        # ── Attitude ─────────────────────────────────────────────────────────
        if att:
            roll    = float(np.degrees(att.roll))
            pitch   = float(np.degrees(att.pitch))
            compass = float(np.degrees(att.yaw)) % 360
        else:
            roll = pitch = compass = 0.0

        # ── Position (GPS → local metres from home) ───────────────────────────
        if pos and self._home_gps:
            lat_m = (pos.lat / 1e7 - self._home_gps[0]) * 111139
            lon_m = (pos.lon / 1e7 - self._home_gps[1]) * \
                    111139 * np.cos(np.radians(self._home_gps[0]))
            pos_x    = float(lon_m)
            pos_y    = float(lat_m)
            altitude = float(pos.relative_alt / 1000.0)   # mm → m
            vel_x    = float(pos.vx / 100.0)              # cm/s → m/s
            vel_y    = float(pos.vy / 100.0)
        else:
            pos_x = pos_y = vel_x = vel_y = altitude = 0.0

        pos_xy         = np.array([pos_x, pos_y])
        dist_to_home   = float(np.linalg.norm(pos_xy - self.home))
        dist_to_target = float(np.linalg.norm(pos_xy - self.target))

        # ── Battery ───────────────────────────────────────────────────────────
        battery = float(bat.battery_remaining) if bat else 100.0

        # ── GPS quality ───────────────────────────────────────────────────────
        if gps:
            gps_fix  = float(gps.fix_type >= 3)    # 3D fix = good
            gps_hdop = float(gps.eph / 100.0)      # cm → dimensionless
        else:
            gps_fix = 0.0; gps_hdop = 99.0

        # ── RC RSSI ───────────────────────────────────────────────────────────
        # RC_CHANNELS.rssi: 0-254 → map to dBm range -120 to -40
        if rc:
            rssi_raw = rc.rssi
            rc_rssi  = -120 + (rssi_raw / 254.0) * 80
        else:
            rc_rssi = -120.0

        # ── Jamming noise estimate ─────────────────────────────────────────────
        # Derived: if RSSI drops and GPS HDOP rises, infer jamming
        rssi_norm  = max(0.0, min(1.0, (-rc_rssi - 40) / 80))  # 0=clean, 1=jammed
        hdop_norm  = max(0.0, min(1.0, (gps_hdop - 1.0) / 50))
        jam_noise  = float(0.5 * rssi_norm + 0.5 * hdop_norm)

        return {
            'accel_x':        accel_x,
            'accel_y':        accel_y,
            'accel_z':        accel_z,
            'gyro_x':         gyro_x,
            'gyro_y':         gyro_y,
            'gyro_z':         gyro_z,
            'compass':        compass,
            'altitude':       altitude,
            'pos_x':          pos_x,
            'pos_y':          pos_y,
            'vel_x':          vel_x,
            'vel_y':          vel_y,
            'dist_to_target': dist_to_target,
            'dist_to_home':   dist_to_home,
            'battery':        battery,
            'pitch':          pitch,
            'roll':           roll,
            'rc_rssi':        rc_rssi,
            'gps_fix':        gps_fix,
            'gps_hdop':       gps_hdop,
            'jam_noise':      jam_noise,
        }

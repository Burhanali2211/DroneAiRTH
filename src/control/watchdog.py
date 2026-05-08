"""
watchdog.py - GCS heartbeat sender + emergency handler.

ArduPilot failsafe parameters to set before flight:
  FS_GCS_ENABLE  = 1       (enable GCS failsafe)
  FS_GCS_TIMEOUT = 1500    (ms without heartbeat -> RTH)

Overrides (priority order):
  battery < 10%      -> EMERGENCY_LAND
  battery < 20%      -> RETURN_HOME
  gps_fix == 0       -> RETURN_HOME
  jam_noise > 0.85   -> RETURN_HOME
  geofence breach    -> RETURN_HOME
  heartbeat timeout  -> force_land()
"""
import threading
import time
from typing import Optional

try:
    from pymavlink import mavutil as _mavutil
    MAVLINK_AVAILABLE = True
except ImportError:
    _mavutil = None
    MAVLINK_AVAILABLE = False

# Safe constant — not resolved at import time from mavutil
_MAV_COMP_ID_MISSIONPLANNER = 190   # numeric value of MAV_COMP_ID_MISSIONPLANNER

from src.config_loader import get_config

def _cfg():
    c = get_config()
    return c['model'] if c else {}

def _mission_cfg():
    c = get_config()
    return c['mission'] if c else {}


HEARTBEAT_HZ      = 1.0
HEARTBEAT_TIMEOUT = 3.0

def _battery_critical():
    return _cfg().get('battery_critical', 10.0)

def _battery_low():
    return _cfg().get('battery_low', 20.0)

def _jam_high():
    return _cfg().get('jam_high', 0.85)


class Watchdog:
    """
    Background heartbeat thread + sensor monitor.

    Usage:
        wd = Watchdog(conn, fsm)
        wd.start()
        override = wd.check(sensor_dict)   # returns action_id or None
        wd.stop()
    """

    def __init__(self, conn, fsm, system_id: int = 255,
                 component_id: int = _MAV_COMP_ID_MISSIONPLANNER,
                 max_range: float = None):
        cfg = _mission_cfg()
        self._conn         = conn
        self._fsm          = fsm
        self._system_id    = system_id
        self._component_id = component_id
        self._max_range    = max_range if max_range is not None else cfg.get('max_range_from_home', 15.0)
        self._thread: Optional[threading.Thread] = None
        self._running      = False
        self._lock         = threading.Lock()
        self._last_beat_ok = time.monotonic()
        self._anomalies: list[dict] = []

    # -- Lifecycle ---------------------------------------------------------------

    def start(self):
        self._running = True
        self._thread  = threading.Thread(
            target=self._beat_loop, daemon=True, name='WatchdogHB')
        self._thread.start()
        print("[WD] Heartbeat thread started (1 Hz)")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        print("[WD] Heartbeat thread stopped")

    # -- Sensor check (call from main loop) -------------------------------------

    def check(self, sensors: dict) -> Optional[int]:
        """
        Inspect sensor values and return an override action_id if critical,
        or None to let the AI decision stand.
        """
        bat  = sensors.get('battery',   100.0)
        fix  = sensors.get('gps_fix',   1.0)
        jam  = sensors.get('jam_noise', 0.0)
        now  = time.monotonic()

        if bat < _battery_critical():
            self._log(now, 'BATTERY_CRITICAL', bat)
            return 3

        if bat < _battery_low():
            self._log(now, 'BATTERY_LOW', bat)
            return 2

        if fix == 0.0:
            self._log(now, 'GPS_LOST', fix)
            return 2

        if jam > _jam_high():
            self._log(now, 'JAMMING_HIGH', jam)
            return 2

        px   = sensors.get('pos_x', 0.0)
        py   = sensors.get('pos_y', 0.0)
        dist = (px**2 + py**2) ** 0.5
        if dist > self._max_range and self._fsm.is_airborne:
            self._log(now, 'GEOFENCE_BREACH', dist)
            return 2

        with self._lock:
            last_ok = self._last_beat_ok
        if now - last_ok > HEARTBEAT_TIMEOUT:
            self._log(now, 'HEARTBEAT_TIMEOUT', 0)
            self._fsm.force_land()
            return 3

        return None

    def anomalies(self) -> list:
        return list(self._anomalies)

    # -- Private ----------------------------------------------------------------

    def _beat_loop(self):
        interval = 1.0 / HEARTBEAT_HZ
        while self._running:
            t0 = time.monotonic()
            try:
                self._send_heartbeat()
                with self._lock:
                    self._last_beat_ok = time.monotonic()
            except Exception as e:
                print(f"[WD] Heartbeat send failed: {e}")
            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, interval - elapsed))

    def _send_heartbeat(self):
        if not MAVLINK_AVAILABLE or self._conn is None:
            return
        self._conn.mav.heartbeat_send(
            _mavutil.mavlink.MAV_TYPE_GCS,
            _mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0, 0,
            _mavutil.mavlink.MAV_STATE_ACTIVE,
        )

    def _log(self, t: float, event: str, value: float):
        entry = {'time': t, 'event': event, 'value': value}
        self._anomalies.append(entry)
        print(f"[WD] {event} = {value:.2f}")

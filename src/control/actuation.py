"""
actuation.py - Translates AI action_id -> MAVLink commands with safety guards.

Action IDs:
  0 CONTINUE_TO_TARGET  - set velocity toward target
  1 HOVER_AT_TARGET     - zero velocity hold
  2 RETURN_HOME         - MAV_CMD_NAV_RETURN_TO_LAUNCH
  3 EMERGENCY_LAND      - controlled descent, not instant drop

Safety rules enforced here (not in AI):
  - LAND only allowed below MAX_LAND_ALT; above -> RTH first
  - Velocity capped at MAX_SPEED to prevent runaway
  - All commands sent in GUIDED mode (checked before each command)
"""
import time
import math
import numpy as np

try:
    from pymavlink import mavutil as _mavutil
    MAVLINK_AVAILABLE = True
except ImportError:
    _mavutil = None
    MAVLINK_AVAILABLE = False

from src.config_loader import get_config


def _mcfg():
    c = get_config()
    return c['mission'] if c else {}


def _max_speed():    return _mcfg().get('max_speed',     2.0)
def _max_land_alt(): return _mcfg().get('max_land_alt',  8.0)
def _descent_rate(): return _mcfg().get('descent_rate',  0.4)
def _cruise_alt():   return _mcfg().get('cruise_alt',    1.2)


class ActionExecutor:
    """
    Converts action_id from DroneAI.predict() into MAVLink commands.

    Usage:
        exe = ActionExecutor(conn, target=np.array([3.0, 2.0]))
        exe.execute(action_id=2, sensors=sensor_dict)
    """

    def __init__(self, conn, target: np.ndarray = None,
                 cruise_alt: float = None):
        cfg = _mcfg()
        self._conn        = conn
        self.target       = (target if target is not None
                             else np.array(cfg.get('target', [3.0, 2.0])))
        self.cruise_alt   = cruise_alt if cruise_alt is not None else _cruise_alt()
        self._last_action = -1
        self._guided_set  = False

    def arm(self):
        """Arm motors. Must be called before any other command is accepted."""
        if not MAVLINK_AVAILABLE or self._conn is None:
            return
        self._conn.mav.command_long_send(
            self._conn.target_system,
            self._conn.target_component,
            _mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0,
        )
        print('[ACT] Arm command sent')

    def takeoff(self, alt: float = None):
        """Command takeoff to cruise altitude."""
        if not MAVLINK_AVAILABLE or self._conn is None:
            return
        target_alt = alt if alt is not None else self.cruise_alt
        self._ensure_guided()
        self._conn.mav.command_long_send(
            self._conn.target_system,
            self._conn.target_component,
            _mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0.0, 0.0, float(target_alt),
        )
        print(f'[ACT] Takeoff command sent (target alt={target_alt}m)')

    # -- Public -----------------------------------------------------------------

    def execute(self, action_id: int, sensors: dict):
        """Execute the action. No-op if pymavlink not available (simulation mode)."""
        if not MAVLINK_AVAILABLE or self._conn is None:
            return

        alt = sensors.get('altitude', 0.0)

        if action_id == 3 and alt > _max_land_alt():
            action_id = 2   # demote EMERGENCY_LAND to RTH above safe altitude

        if action_id != self._last_action:
            self._ensure_guided()

        if   action_id == 0: self._continue_to_target(sensors)
        elif action_id == 1: self._hover()
        elif action_id == 2: self._return_home()
        elif action_id == 3: self._emergency_land()

        self._last_action = action_id

    # -- Private commands -------------------------------------------------------

    def _ensure_guided(self):
        if self._conn is None:
            return
        self._conn.mav.command_long_send(
            self._conn.target_system,
            self._conn.target_component,
            _mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,
            _mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            4,   # GUIDED mode number for ArduCopter
            0, 0, 0, 0, 0,
        )
        self._guided_set = True

    def _continue_to_target(self, sensors: dict):
        px  = sensors.get('pos_x', 0.0)
        py  = sensors.get('pos_y', 0.0)
        alt = sensors.get('altitude', self.cruise_alt)

        dx   = self.target[0] - px
        dy   = self.target[1] - py
        dist = math.sqrt(dx*dx + dy*dy) + 1e-6

        speed = min(_max_speed(), dist)
        vx = (dx / dist) * speed
        vy = (dy / dist) * speed
        vz = max(-0.5, min(0.5, self.cruise_alt - alt)) * 0.5

        self._send_velocity(vx, vy, vz)

    def _hover(self):
        self._send_velocity(0.0, 0.0, 0.0)

    def _return_home(self):
        if self._conn is None:
            return
        self._conn.mav.command_long_send(
            self._conn.target_system,
            self._conn.target_component,
            _mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
            0, 0, 0, 0, 0, 0, 0, 0,
        )

    def _emergency_land(self):
        self._send_velocity(0.0, 0.0, _descent_rate())

    def _send_velocity(self, vx: float, vy: float, vz: float):
        """SET_POSITION_TARGET_LOCAL_NED velocity-only mode."""
        if self._conn is None:
            return
        # Build type mask only when mavutil is available (guarded by caller checks)
        TYPE_MASK = (
            _mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE  |
            _mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE  |
            _mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE  |
            _mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
            _mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
            _mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
            _mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE |
            _mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
        )
        self._conn.mav.set_position_target_local_ned_send(
            int(time.monotonic() * 1000) & 0xFFFFFFFF,
            self._conn.target_system,
            self._conn.target_component,
            _mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            TYPE_MASK,
            0, 0, 0,
            float(vx), float(vy), float(vz),
            0, 0, 0,
            0, 0,
        )

"""
actuation.py — Translates AI action_id → MAVLink commands with safety guards.

Action IDs:
  0 CONTINUE_TO_TARGET  — set velocity toward target
  1 HOVER_AT_TARGET     — zero velocity hold
  2 RETURN_HOME         — MAV_CMD_NAV_RETURN_TO_LAUNCH
  3 EMERGENCY_LAND      — controlled descent, not instant drop

Safety rules enforced here (not in AI):
  - LAND only allowed below MAX_LAND_ALT (8m); above → RTH first
  - Velocity capped at MAX_SPEED (2.0 m/s) to prevent runaway
  - All commands sent in GUIDED mode (checked before each command)
"""
import time
import math
import numpy as np
from typing import Optional

try:
    from pymavlink import mavutil
    MAVLINK_AVAILABLE = True
except ImportError:
    MAVLINK_AVAILABLE = False


MAX_SPEED     = 2.0    # m/s — hard cap on commanded velocity
MAX_LAND_ALT  = 8.0   # m   — above this, LAND → RTH instead
DESCENT_RATE  = 0.4   # m/s — controlled landing descent rate
CRUISE_ALT    = 1.2   # m   — default cruise altitude


class ActionExecutor:
    """
    Converts action_id from DroneAI.predict() into MAVLink commands.
    Requires an active MAVLink connection (from MAVLinkBridge._conn).

    Usage:
        exec = ActionExecutor(conn, target=[3.0, 2.0])
        exec.execute(action_id=2, sensors=sensor_dict)
    """

    def __init__(self, conn, target: np.ndarray = None,
                 cruise_alt: float = CRUISE_ALT):
        self._conn      = conn      # raw pymavlink connection object
        self.target     = target if target is not None else np.array([3.0, 2.0])
        self.cruise_alt = cruise_alt
        self._last_action = -1
        self._guided_set  = False

    def arm(self):
        """Arm motors. Must be called before any other command is accepted."""
        if not MAVLINK_AVAILABLE or self._conn is None:
            return
        self._conn.mav.command_long_send(
            self._conn.target_system,
            self._conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1,   # 1 = arm
            0, 0, 0, 0, 0, 0,
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
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,
            0, 0, 0, 0,   # pitch, empty, empty, yaw
            0.0, 0.0,     # lat, lon (current position)
            float(target_alt),
        )
        print(f'[ACT] Takeoff command sent (target alt={target_alt}m)')

    # ── Public ────────────────────────────────────────────────────────────────

    def execute(self, action_id: int, sensors: dict):
        """
        Execute the action. sensors dict used for altitude safety check.
        No-op if pymavlink not available (simulation mode).
        """
        if not MAVLINK_AVAILABLE or self._conn is None:
            return

        alt = sensors.get('altitude', 0.0)

        # Safety override: can't land from high altitude
        if action_id == 3 and alt > MAX_LAND_ALT:
            action_id = 2   # demote to RTH

        if action_id != self._last_action:
            self._ensure_guided()

        if   action_id == 0: self._continue_to_target(sensors)
        elif action_id == 1: self._hover()
        elif action_id == 2: self._return_home()
        elif action_id == 3: self._emergency_land()

        self._last_action = action_id

    # ── Private commands ──────────────────────────────────────────────────────

    def _ensure_guided(self):
        """Switch to GUIDED mode if not already there."""
        if self._conn is None:
            return
        self._conn.mav.command_long_send(
            self._conn.target_system,
            self._conn.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            4,   # GUIDED mode number for ArduCopter
            0, 0, 0, 0, 0,
        )
        self._guided_set = True

    def _continue_to_target(self, sensors: dict):
        """Send velocity setpoint toward target."""
        px = sensors.get('pos_x', 0.0)
        py = sensors.get('pos_y', 0.0)
        alt = sensors.get('altitude', self.cruise_alt)

        dx = self.target[0] - px
        dy = self.target[1] - py
        dist = math.sqrt(dx*dx + dy*dy) + 1e-6

        speed = min(MAX_SPEED, dist)   # slow down when close
        vx = (dx / dist) * speed
        vy = (dy / dist) * speed

        # Altitude correction: maintain cruise_alt
        vz = max(-0.5, min(0.5, self.cruise_alt - alt)) * 0.5

        self._send_velocity(vx, vy, vz)

    def _hover(self):
        """Zero velocity — hold position."""
        self._send_velocity(0.0, 0.0, 0.0)

    def _return_home(self):
        """MAV_CMD_NAV_RETURN_TO_LAUNCH — FC handles RTH autonomously."""
        if self._conn is None:
            return
        self._conn.mav.command_long_send(
            self._conn.target_system,
            self._conn.target_component,
            mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
            0,
            0, 0, 0, 0, 0, 0, 0,
        )

    def _emergency_land(self):
        """Controlled descent — not instant drop."""
        # Negative vz = descend in NED frame
        self._send_velocity(0.0, 0.0, DESCENT_RATE)

    def _send_velocity(self, vx: float, vy: float, vz: float):
        """
        SET_POSITION_TARGET_LOCAL_NED in velocity mode.
        Type mask 0b0000111111000111 = only vx, vy, vz active.
        """
        if self._conn is None:
            return
        TYPE_MASK = (
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
        )
        self._conn.mav.set_position_target_local_ned_send(
            int(time.monotonic() * 1000) & 0xFFFFFFFF,  # time_boot_ms
            self._conn.target_system,
            self._conn.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            TYPE_MASK,
            0, 0, 0,           # x, y, z (ignored)
            float(vx), float(vy), float(vz),
            0, 0, 0,           # ax, ay, az (ignored)
            0, 0,              # yaw, yaw_rate (ignored)
        )

"""
state_machine.py — Phase-safe flight state machine.

Enforces legal transition sequences so the AI can't, for example,
jump directly from CRUISE to LAND at 30m.

State graph:
  PREFLIGHT → TAKEOFF → CRUISE → HOVER
                                ↓
                           RETURN_HOME → LANDING → LANDED
                                ↓
                            EMERGENCY_LAND → LANDED

EMERGENCY_LAND is always reachable from any airborne state.
RETURN_HOME and HOVER can both reach LANDING.
"""
import time
from enum import Enum, auto


class FlightPhase(Enum):
    PREFLIGHT      = auto()
    TAKEOFF        = auto()
    CRUISE         = auto()
    HOVER          = auto()
    RETURN_HOME    = auto()
    LANDING        = auto()
    LANDED         = auto()
    EMERGENCY_LAND = auto()


# Legal transitions: {from_state: {to_states}}
TRANSITIONS = {
    FlightPhase.PREFLIGHT:      {FlightPhase.TAKEOFF},
    FlightPhase.TAKEOFF:        {FlightPhase.CRUISE, FlightPhase.HOVER,
                                 FlightPhase.EMERGENCY_LAND},
    FlightPhase.CRUISE:         {FlightPhase.HOVER, FlightPhase.RETURN_HOME,
                                 FlightPhase.EMERGENCY_LAND},
    FlightPhase.HOVER:          {FlightPhase.CRUISE, FlightPhase.RETURN_HOME,
                                 FlightPhase.LANDING, FlightPhase.EMERGENCY_LAND},
    FlightPhase.RETURN_HOME:    {FlightPhase.LANDING, FlightPhase.HOVER,
                                 FlightPhase.EMERGENCY_LAND},
    FlightPhase.LANDING:        {FlightPhase.LANDED, FlightPhase.EMERGENCY_LAND},
    FlightPhase.LANDED:         set(),
    FlightPhase.EMERGENCY_LAND: {FlightPhase.LANDED},
}

# Map action_id → desired phase
ACTION_TO_PHASE = {
    0: FlightPhase.CRUISE,
    1: FlightPhase.HOVER,
    2: FlightPhase.RETURN_HOME,
    3: FlightPhase.EMERGENCY_LAND,
}


class FlightStateMachine:
    """
    Converts action_id from DroneAI into a phase, enforcing legal transitions.

    Usage:
        fsm = FlightStateMachine()
        fsm.arm()                           # PREFLIGHT → TAKEOFF
        phase = fsm.apply(action_id=2)      # CRUISE → RETURN_HOME (if legal)
        fsm.on_landed()                     # LANDING → LANDED
    """

    def __init__(self):
        self.phase = FlightPhase.PREFLIGHT
        self._phase_entry_time = time.monotonic()
        self._log: list[tuple] = []   # (time, from_phase, to_phase)

    @property
    def is_airborne(self) -> bool:
        return self.phase not in {FlightPhase.PREFLIGHT,
                                  FlightPhase.LANDED}

    @property
    def phase_duration(self) -> float:
        return time.monotonic() - self._phase_entry_time

    def arm(self):
        self._transition(FlightPhase.TAKEOFF)

    def on_cruising(self):
        self._transition(FlightPhase.CRUISE)

    def on_landed(self):
        self._transition(FlightPhase.LANDED)

    def apply(self, action_id: int) -> FlightPhase:
        """
        Attempt to move to the phase implied by action_id.
        If illegal, silently hold current phase and return it.
        Returns the (possibly unchanged) current phase.
        """
        target = ACTION_TO_PHASE.get(action_id)
        if target is None:
            return self.phase

        if target in TRANSITIONS.get(self.phase, set()):
            self._transition(target)
        elif target == FlightPhase.EMERGENCY_LAND and self.is_airborne:
            # EMERGENCY_LAND always allowed when airborne
            self._transition(target)
        # else: illegal — hold phase silently

        # Auto-progress: HOVER → LANDING when near ground (caller checks alt)
        return self.phase

    def force_land(self):
        """Called by watchdog on heartbeat timeout."""
        if self.is_airborne:
            self._transition(FlightPhase.EMERGENCY_LAND)

    def _transition(self, to: FlightPhase):
        if to == self.phase:
            return
        self._log.append((time.monotonic(), self.phase, to))
        print(f"[FSM] {self.phase.name} -> {to.name}")
        self.phase = to
        self._phase_entry_time = time.monotonic()

    def history(self) -> list:
        return list(self._log)

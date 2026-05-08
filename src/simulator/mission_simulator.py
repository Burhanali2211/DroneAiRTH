import numpy as np
import pandas as pd
from pathlib import Path

from .sensor_simulator import DroneSensorSimulator


class MissionSimulator:
    """
    Full 30-second anti-jamming mission with curved 2D waypoint path.

    Outbound: home → wp1 → wp2 → target  (curved arc)
    Return:   target → wp3 → wp4 → home  (different arc)

    AI makes decisions every timestep; during jamming those are logged
    and printed at 1 Hz.
    """

    # Curved waypoints so the 2D trajectory is not a straight diagonal
    WAYPOINTS_OUT  = [np.array([1.0, 0.3]),
                      np.array([2.0, 1.5]),
                      np.array([3.0, 2.0])]   # = target

    WAYPOINTS_BACK = [np.array([2.5, 0.8]),
                      np.array([1.2, 0.2]),
                      np.array([0.0, 0.0])]   # = home

    PHASES = {
        'TAKEOFF':       (0.0,   2.0),
        'FLY_TO_TARGET': (2.0,   8.0),
        'HOVER_TARGET':  (8.0,  12.0),
        'RETURN_HOME':   (12.0, 22.0),
        'LAND':          (22.0, 30.0),
    }

    def __init__(self, ai):
        self.ai      = ai          # DroneAI instance
        self.sim     = DroneSensorSimulator(seed=7)
        self.log: list[dict] = []
        self.success = False

    # ------------------------------------------------------------------

    def run(self, duration: float = 30.0, dt: float = 0.02,
            jam_start: float = 6.0, jam_dur: float = 12.0):

        from src.model.drone_ai import DroneAI  # avoid circular at module level

        self.sim.reset()
        self.ai.reset_state()
        self.log     = []
        self.success = False

        wp_out_idx  = 0
        wp_back_idx = 0
        jam_end     = jam_start + jam_dur
        prev_jammed = False
        last_print  = -1.0

        print("\n" + "=" * 65)
        print("MISSION SIMULATION")
        print("=" * 65)

        t = 0.0
        while t <= duration:
            self.sim.time = t

            # Determine phase
            phase = 'LAND'
            for ph, (t0, t1) in self.PHASES.items():
                if t0 <= t < t1:
                    phase = ph
                    break

            # ---- Physics ----
            if phase == 'TAKEOFF':
                self.sim.step_takeoff(dt)

            elif phase == 'FLY_TO_TARGET':
                wp = self.WAYPOINTS_OUT[wp_out_idx]
                if np.linalg.norm(self.sim.position[:2] - wp) < 0.15:
                    wp_out_idx = min(wp_out_idx + 1, len(self.WAYPOINTS_OUT) - 1)
                self.sim.step_fly_to(dt, self.WAYPOINTS_OUT[wp_out_idx])

            elif phase == 'HOVER_TARGET':
                self.sim.step_hover(dt)

            elif phase == 'RETURN_HOME':
                wp = self.WAYPOINTS_BACK[wp_back_idx]
                if np.linalg.norm(self.sim.position[:2] - wp) < 0.15:
                    wp_back_idx = min(wp_back_idx + 1, len(self.WAYPOINTS_BACK) - 1)
                self.sim.step_fly_to(dt, self.WAYPOINTS_BACK[wp_back_idx])

            else:
                self.sim.step_land(dt)

            # ---- Jamming state ----
            jammed_now = (jam_start <= t < jam_end)
            self.sim.is_jammed = jammed_now
            if jammed_now:
                ramp = min(1.0, (t - jam_start) / max(0.5, jam_dur * 0.3))
                self.sim.jamming_strength = ramp
            else:
                self.sim.jamming_strength = 0.0

            if jammed_now and not prev_jammed:
                print(f"\n[JAM] JAMMING STARTED at t={t:.1f}s")
                print("[JAM] RC + GPS LOST - AI autonomous mode ON\n")
            elif not jammed_now and prev_jammed:
                print(f"\n[JAM] Signal recovered at t={t:.1f}s\n")
            prev_jammed = jammed_now

            # ---- Sensors + AI decision ----
            sensor     = self.sim.get_sensor_data()
            action_id, confidence, all_probs = self.ai.predict(list(sensor.values()))

            entry = {
                't':            t,
                'phase':        phase,
                'jammed':       jammed_now,
                'action_id':    action_id,
                'action_name':  DroneAI.ACTION_NAMES[action_id],
                'confidence':   confidence,
                'prob_continue': float(all_probs[0]),
                'prob_hover':    float(all_probs[1]),
                'prob_rth':      float(all_probs[2]),
                'prob_land':     float(all_probs[3]),
                **sensor,
            }
            self.log.append(entry)

            if jammed_now and t - last_print >= 1.0:
                last_print = t
                print(f"[AI] t={t:5.1f}s | {DroneAI.ACTION_NAMES[action_id]:22} | "
                      f"conf={confidence:.1%} | "
                      f"dist_home={sensor['dist_to_home']:.2f}m | "
                      f"bat={sensor['battery']:.1f}%")

            # Success: back home after jamming ends
            if t > jam_end + 2.0:
                if sensor['dist_to_home'] < 0.6 and sensor['altitude'] < 0.3:
                    self.success = True

            t = round(t + dt, 4)

        self._print_summary(jam_dur)

        # Auto-save flight log
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / f"sim_{int(t*100):06d}.csv"
        pd.DataFrame(self.log).to_csv(str(log_path), index=False)
        print(f"[SIM] Flight log saved: {log_path}")

    # ------------------------------------------------------------------

    def _print_summary(self, jam_dur: float):
        from src.model.drone_ai import DroneAI

        df    = pd.DataFrame(self.log)
        jam_df = df[df['jammed']]
        final  = df.iloc[-1]

        print("\n" + "=" * 65)
        print("MISSION RESULTS")
        print("=" * 65)
        print(f"  Flight time:      {df['t'].max():.1f}s")
        print(f"  Jamming duration: {jam_dur:.1f}s")
        print(f"  AI decisions:     {len(jam_df)}")
        print(f"  Final position:   ({final['pos_x']:.2f}m, {final['pos_y']:.2f}m)")
        print(f"  Dist from home:   {final['dist_to_home']:.2f}m")
        print(f"  Battery left:     {final['battery']:.1f}%")
        print(f"  Status:           {'SUCCESS' if self.success else 'INCOMPLETE'}")

        if len(jam_df) > 0:
            counts = jam_df['action_name'].value_counts()
            print("\n  AI decisions during jamming:")
            for action, cnt in counts.items():
                print(f"    {action}: {cnt} ({cnt / len(jam_df):.1%})")
        print("=" * 65 + "\n")

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.log)

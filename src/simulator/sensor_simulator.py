import numpy as np


class DroneSensorSimulator:
    """
    2D drone physics + sensor model (X-Y plane, altitude Z).
    Produces 21-feature sensor dicts including jamming signal indicators.
    """

    ACTIONS = ['CONTINUE_TO_TARGET', 'HOVER_AT_TARGET', 'RETURN_HOME', 'EMERGENCY_LAND']

    TARGET = np.array([3.0, 2.0])   # mission target in X-Y metres

    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.reset()

    def reset(self):
        self.time            = 0.0
        self.position        = np.array([0.0, 0.0, 0.0])  # x, y, z
        self.velocity        = np.array([0.0, 0.0, 0.0])
        self.heading         = 0.0   # degrees
        self.roll            = 0.0
        self.pitch           = 0.0
        self.battery         = 100.0
        self.target          = self.TARGET.copy()
        self.is_jammed       = False
        self.jamming_strength = 0.0

    # ------------------------------------------------------------------
    # Sensor output
    # ------------------------------------------------------------------

    def _noise(self, std):
        return np.random.normal(0, std)

    def get_sensor_data(self) -> dict:
        """Return all 21 sensor features as a dict."""

        # IMU
        accel_x = self.pitch / 30 * 9.81 + self._noise(0.05)
        accel_y = self.roll  / 30 * 9.81 + self._noise(0.05)
        accel_z = 9.81 + self._noise(0.1)

        gyro_x = self.roll  * 0.5 + self._noise(0.1)
        gyro_y = self.pitch * 0.5 + self._noise(0.1)
        gyro_z = self._noise(0.1)          # heading rate (was always 0 in v1)

        compass = (self.heading + self._noise(1.0)) % 360

        altitude       = max(0.0, self.position[2] + self._noise(0.01))
        pos_x          = float(self.position[0])
        pos_y          = float(self.position[1])
        vel_x          = float(self.velocity[0])
        vel_y          = float(self.velocity[1])
        dist_to_target = float(np.linalg.norm(self.position[:2] - self.target))
        dist_to_home   = float(np.linalg.norm(self.position[:2]))
        battery        = self.battery

        # Jamming signal indicators
        if self.is_jammed:
            rc_rssi  = -120 + self._noise(5) * self.jamming_strength
            gps_fix  = 0.0
            gps_hdop = 99.0
            jam_noise = self.jamming_strength + self._noise(0.05)
        else:
            rc_rssi  = -60 + self._noise(3)
            gps_fix  = 1.0
            gps_hdop = 1.2 + self._noise(0.1)
            jam_noise = self._noise(0.02)

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
            'pitch':          self.pitch,
            'roll':           self.roll,
            'rc_rssi':        rc_rssi,
            'gps_fix':        gps_fix,
            'gps_hdop':       gps_hdop,
            'jam_noise':      jam_noise,
        }

    # ------------------------------------------------------------------
    # Physics steps
    # ------------------------------------------------------------------

    def _drain_battery(self, dt: float, rate: float):
        """rate = % per second"""
        self.battery = max(0.0, self.battery - rate * dt)

    def step_takeoff(self, dt: float):
        self.position[2] = min(self.position[2] + 0.25 * dt, 1.5)
        self.velocity[2] = 0.25
        self.pitch = 0.0
        self.roll  = 0.0
        self._drain_battery(dt, rate=0.5)

    def step_fly_to(self, dt: float, waypoint: np.ndarray):
        """Fly toward an arbitrary 2D waypoint."""
        direction = waypoint - self.position[:2]
        dist = np.linalg.norm(direction)
        if dist > 0.05:
            unit  = direction / dist
            speed = min(1.5, dist * 2)
            self.velocity[:2] = unit * speed
            self.position[:2] += self.velocity[:2] * dt
            self.heading = float(np.degrees(np.arctan2(unit[1], unit[0]))) % 360
            self.pitch   = 15 * (speed / 1.5)
        else:
            self.velocity[:2] = 0
            self.pitch = 0.0
        self.position[2] = 1.5
        self._drain_battery(dt, rate=0.8)

    def step_hover(self, dt: float):
        self.velocity[:2] = 0
        self.pitch = np.sin(self.time * 2) * 1.0
        self.roll  = np.cos(self.time * 2) * 0.8
        self.position[2] = 1.5
        self._drain_battery(dt, rate=0.5)

    def step_land(self, dt: float):
        self.position[2] = max(0.0, self.position[2] - 0.3 * dt)
        self.velocity     = np.zeros(3)
        self.pitch = 0.0
        self.roll  = 0.0
        self._drain_battery(dt, rate=0.4)

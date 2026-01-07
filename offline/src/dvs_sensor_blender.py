import bpy
import math
import mathutils
from mathutils import Vector, Quaternion
import numpy as np
from dvs_sensor import DvsSensor

# Global variable
bins = []
for dec in range(-3, 2, 1):
    bins.append(np.arange(10 ** dec, 10 ** (dec + 1), 10 ** dec))
bins = np.array(bins)
FREQ = bins.reshape(bins.shape[0] * bins.shape[1])


class Blender_DvsSensor(DvsSensor):
    """
    Structure to handle the Camera with Blender parameters such as position, optics, etc.

    Extension: optional look-at + drift + microsaccades control.

    - If look-at drift/microsaccades are enabled, orientation is controlled in QUATERNION mode:
        base quaternion points camera (-Z) to a target object
        drift is a continuous random-walk (Gaussian increments each frame)
        microsaccades are discrete events (Poisson trigger) spread over a short duration
        total pan/tilt = drift + microsaccade offsets
    - If not enabled, the original Euler integration (angle += angular_speed * dt) is used.

    Patch added:
    - Optional gating: drift increments can be frozen while a microsaccade is active.
      This better matches the common modeling assumption that drift is an inter-microsaccade process.
    """

    pixel_pitch = 0.015  # mm
    focal = 8.0          # mm
    def_x = 640          # pixel
    def_y = 640          # pixel

    position = np.array([0.0, 0.0, 0.0], float)              # Blender units
    angle = np.array([0.0, 0.0, 0.0], float)                 # Euler angles (radians)
    speed = np.array([0.0, 0.0, 0.0], float)                 # BU/s
    angular_speed = np.array([0.0, 0.0, 0.0], float)         # rad/s

    def __init__(self, name):
        """Create a new Blender camera object."""
        super().__init__(name)  # if DvsSensor defines init; safe even if empty

        self.name = name
        cam_data = bpy.data.cameras.new(name)
        self.cam = bpy.data.objects.new(name, cam_data)

        # --- Look-at + drift + microsaccades state (disabled by default) ---
        self._use_quat = False
        self._target_obj = None

        # Drift parameters (Gaussian increments per frame, in degrees)
        self._drift_pan_sigma_deg = 0.0
        self._drift_tilt_sigma_deg = 0.0
        self._drift_pan = 0.0   # accumulated drift (radians)
        self._drift_tilt = 0.0  # accumulated drift (radians)

        # Microsaccade parameters
        self._ms_rate_hz = 2.0            # events per second
        self._ms_amp_deg = (0.05, 0.5)    # amplitude range per event (degrees)
        self._ms_dur_ms = (10, 30)        # duration range (ms)

        # Microsaccade event state
        self._ms_active = False
        self._ms_frames_left = 0
        self._ms_pan_step = 0.0
        self._ms_tilt_step = 0.0
        self._ms_pan = 0.0    # accumulated microsaccade offset (radians)
        self._ms_tilt = 0.0   # accumulated microsaccade offset (radians)

        # NEW: drift gating during microsaccades
        self._freeze_drift_during_ms = True

        self._rng = np.random.default_rng()

    # -------------------------
    # Camera intrinsics / optics
    # -------------------------
    def set_sensor(self, nx, ny, pp):
        """
        Initialize the properties of the sensor:
        definition (nx, ny) and pixel pitch (mm).
        """
        self.pixel_pitch = pp
        self.def_x = nx
        self.def_y = ny
        self.cam.data.sensor_height = self.pixel_pitch * self.def_y
        self.cam.data.sensor_width = self.pixel_pitch * self.def_x
        self.shape = (nx, ny)

    def set_sensor_optics(self, focal):
        """Set the optics according to the focal length (mm)."""
        self.focal = focal
        self.cam.data.angle_x = 2 * math.atan(self.cam.data.sensor_width / (2 * self.focal))
        self.cam.data.angle_y = 2 * math.atan(self.cam.data.sensor_height / (2 * self.focal))

    # -------------------------
    # Pose / motion
    # -------------------------
    def update_cam(self):
        """Update the camera pose in Blender from stored position/Euler angle."""
        self.cam.location = Vector((self.position[0], self.position[1], self.position[2]))
        self.cam.rotation_mode = 'XYZ'
        self.cam.rotation_euler = mathutils.Euler((self.angle[0], self.angle[1], self.angle[2]))

    def set_position(self, position):
        """Set camera position (Blender units)."""
        self.position = np.array(position, dtype=float)
        self.cam.location = Vector((position[0], position[1], position[2]))

    def set_angle(self, angle):
        """Set camera Euler angle (radians)."""
        self.angle = np.array(angle, dtype=float)
        self.cam.rotation_mode = 'XYZ'
        self.cam.rotation_euler = mathutils.Euler((angle[0], angle[1], angle[2]))

    def set_speeds(self, speed, angular_speed):
        """Set translation speed (BU/s) and Euler angular speed (rad/s)."""
        self.speed = np.array(speed, dtype=float)
        self.angular_speed = np.array(angular_speed, dtype=float)

    # -------------------------
    # Look-at + drift + microsaccades API
    # -------------------------
    def enable_look_at_drift_and_microsaccades(
        self,
        target_obj,
        drift_sigma_deg=(0.02, 0.02),
        microsaccade_rate_hz=2.0,
        microsaccade_amp_deg=(0.2, 1.0),
        microsaccade_dur_ms=(10, 30),
        seed=None,
        drift_freeze_during_ms=True,   # NEW
    ):
        """
        Enable look-at + drift + microsaccades mode.

        drift_sigma_deg: (pan_sigma, tilt_sigma) Gaussian increment std per frame (degrees).
                         This creates continuous drift as a random walk.
        microsaccade_rate_hz: Poisson rate (events/second).
        microsaccade_amp_deg: (min,max) amplitude per event (degrees).
        microsaccade_dur_ms: (min,max) duration per event (ms), spread over frames.
        drift_freeze_during_ms: if True, drift increments are frozen while a microsaccade is active.
        """
        self._target_obj = target_obj
        self._use_quat = True
        self.cam.rotation_mode = 'QUATERNION'

        self._drift_pan_sigma_deg = float(drift_sigma_deg[0])
        self._drift_tilt_sigma_deg = float(drift_sigma_deg[1])

        self._ms_rate_hz = float(microsaccade_rate_hz)
        self._ms_amp_deg = tuple(microsaccade_amp_deg)
        self._ms_dur_ms = tuple(microsaccade_dur_ms)

        self._freeze_drift_during_ms = bool(drift_freeze_during_ms)

        # Reset state
        self._drift_pan = 0.0
        self._drift_tilt = 0.0

        self._ms_active = False
        self._ms_frames_left = 0
        self._ms_pan_step = 0.0
        self._ms_tilt_step = 0.0
        self._ms_pan = 0.0
        self._ms_tilt = 0.0

        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def disable_look_at_drift_and_microsaccades(self):
        """Disable look-at drift/microsaccades and return to Euler integration."""
        self._use_quat = False
        self._target_obj = None
        self._ms_active = False
        self._ms_frames_left = 0

    def _base_quat_look_at(self, target_obj):
        """Quaternion base that points camera (-Z) towards target_obj."""
        direction = target_obj.location - self.cam.location
        if direction.length == 0:
            return self.cam.rotation_quaternion.copy()
        return direction.to_track_quat('-Z', 'Y')

    def _apply_pan_tilt_on_base(self, base_quat, pan_offset, tilt_offset):
        """Apply pan/tilt around local axes defined by base_quat."""
        axis_right = base_quat @ Vector((1, 0, 0))
        axis_up = base_quat @ Vector((0, 1, 0))

        q_pan = Quaternion(axis_up, pan_offset)
        q_tilt = Quaternion(axis_right, tilt_offset)

        self.cam.rotation_mode = 'QUATERNION'
        self.cam.rotation_quaternion = (q_pan @ q_tilt @ base_quat)

    # -------------------------
    # Drift (continuous)
    # -------------------------
    def _update_drift(self):
        """Random-walk drift: Gaussian increments each frame (sigmas in degrees)."""
        pan_sigma = math.radians(self._drift_pan_sigma_deg)
        tilt_sigma = math.radians(self._drift_tilt_sigma_deg)

        if pan_sigma > 0.0:
            self._drift_pan += float(self._rng.normal(0.0, pan_sigma))
        if tilt_sigma > 0.0:
            self._drift_tilt += float(self._rng.normal(0.0, tilt_sigma))

    # -------------------------
    # Microsaccades (discrete events)
    # -------------------------
    def _maybe_start_microsaccade(self, dt_s):
        """Poisson trigger of microsaccades; sets per-frame pan/tilt steps."""
        if self._ms_active:
            return
        if self._ms_rate_hz <= 0.0:
            return

        # Approximate Poisson for small dt: P(event) ~ rate * dt
        p = self._ms_rate_hz * float(dt_s)
        if self._rng.random() >= p:
            return

        dur_ms = int(self._rng.integers(int(self._ms_dur_ms[0]), int(self._ms_dur_ms[1]) + 1))
        frames = max(1, int(round((dur_ms / 1000.0) / float(dt_s))))

        amp_deg = float(self._rng.uniform(self._ms_amp_deg[0], self._ms_amp_deg[1]))
        amp_rad = math.radians(amp_deg)

        # Random direction in (pan, tilt) plane
        phi = float(self._rng.uniform(0.0, 2.0 * math.pi))
        pan_total = amp_rad * math.cos(phi)
        tilt_total = amp_rad * math.sin(phi)

        self._ms_active = True
        self._ms_frames_left = frames
        self._ms_pan_step = pan_total / frames
        self._ms_tilt_step = tilt_total / frames

    def _update_microsaccade(self):
        """
        Accumulate microsaccade offset over its duration.
        After the event ends, we keep the final offset (landing point).
        """
        if self._ms_active and self._ms_frames_left > 0:
            self._ms_pan += self._ms_pan_step
            self._ms_tilt += self._ms_tilt_step

            self._ms_frames_left -= 1
            if self._ms_frames_left <= 0:
                self._ms_active = False

    # -------------------------
    # Time update
    # -------------------------
    def update_time(self, dt):
        """
        Update the Blender world by updating camera pose.

        dt: delay since last update in seconds.

        - Translation: always integrated from self.speed.
        - Orientation:
            - If look-at drift/microsaccades enabled:
                base quaternion + (drift + microsaccade) pan/tilt
                drift can be frozen during microsaccades (optional gating)
            - Else:
                Euler integration angle += angular_speed * dt
        """
        # Translation
        self.position = self.position + self.speed * float(dt)
        self.cam.location = Vector((self.position[0], self.position[1], self.position[2]))

        # Orientation
        if self._use_quat and (self._target_obj is not None):
            self.cam.rotation_mode = 'QUATERNION'
            base_quat = self._base_quat_look_at(self._target_obj)

            # 1) Microsaccade trigger/update first
            self._maybe_start_microsaccade(dt)
            self._update_microsaccade()

            # 2) Drift update (optionally frozen during ms)
            if not (self._freeze_drift_during_ms and self._ms_active):
                self._update_drift()

            # 3) Total offsets
            pan_total = self._drift_pan + self._ms_pan
            tilt_total = self._drift_tilt + self._ms_tilt

            self._apply_pan_tilt_on_base(base_quat, pan_total, tilt_total)

        else:
            self.angle = self.angle + self.angular_speed * float(dt)
            self.update_cam()

    def print_position(self):
        """Print position and Euler angle (only meaningful in Euler mode)."""
        s1 = " x : %f, y : %f, z : %f \n" % (self.position[0], self.position[1], self.position[2])
        print(s1)
        s2 = " a1 : %f, a2 : %f, a3 : %f \n" % (self.angle[0], self.angle[1], self.angle[2])
        print(s2)

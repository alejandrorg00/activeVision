from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from mathutils import Quaternion


@dataclass
class CameraMotionState:
    drift_pan: float = 0.0
    drift_tilt: float = 0.0

    ms_pan: float = 0.0
    ms_tilt: float = 0.0
    ms_pan_step: float = 0.0
    ms_tilt_step: float = 0.0
    ms_frames_left: int = 0
    ms_active: bool = False


class MicrosaccadeCameraController:
    """
    Offline camera controller.

    Adds three components:
    - slow drift
    - discrete microsaccades
    - frame-wise jitter

    The perturbation is applied around a fixed base camera orientation.
    """

    def __init__(
        self,
        camera,
        base_rotation_quaternion,
        drift_sigma_deg=(0.01, 0.01),
        microsaccade_rate_hz: float = 5.0,
        microsaccade_amp_deg=(0.05, 0.30),
        microsaccade_dur_ms=(10, 30),
        jitter_sigma_deg=(0.005, 0.005),
        seed: int | None = None,
    ):
        self.camera = camera
        self.base_rotation = base_rotation_quaternion.copy()
        self.state = CameraMotionState()

        self.drift_sigma_pan = math.radians(float(drift_sigma_deg[0]))
        self.drift_sigma_tilt = math.radians(float(drift_sigma_deg[1]))

        self.ms_rate_hz = float(microsaccade_rate_hz)
        self.ms_amp_deg = tuple(microsaccade_amp_deg)
        self.ms_dur_ms = tuple(microsaccade_dur_ms)

        self.jitter_sigma_pan = math.radians(float(jitter_sigma_deg[0]))
        self.jitter_sigma_tilt = math.radians(float(jitter_sigma_deg[1]))

        self.rng = np.random.default_rng(seed)

    def update(self, dt: float) -> dict:
        s = self.state

        # Slow accumulated drift
        s.drift_pan += float(self.rng.normal(0.0, self.drift_sigma_pan))
        s.drift_tilt += float(self.rng.normal(0.0, self.drift_sigma_tilt))

        # Start microsaccade
        if not s.ms_active:
            if self.ms_rate_hz > 0 and self.rng.random() < self.ms_rate_hz * dt:
                dur_ms = float(self.rng.uniform(self.ms_dur_ms[0], self.ms_dur_ms[1]))
                n_frames = max(1, int(round((dur_ms / 1000.0) / dt)))

                amp = math.radians(float(self.rng.uniform(self.ms_amp_deg[0], self.ms_amp_deg[1])))
                phi = float(self.rng.uniform(0.0, 2.0 * math.pi))

                s.ms_pan_step = amp * math.cos(phi) / n_frames
                s.ms_tilt_step = amp * math.sin(phi) / n_frames
                s.ms_frames_left = n_frames
                s.ms_active = True

        # Continue microsaccade
        if s.ms_active:
            s.ms_pan += s.ms_pan_step
            s.ms_tilt += s.ms_tilt_step
            s.ms_frames_left -= 1

            if s.ms_frames_left <= 0:
                s.ms_active = False
                s.ms_pan_step = 0.0
                s.ms_tilt_step = 0.0

        # White jitter
        jitter_pan = float(self.rng.normal(0.0, self.jitter_sigma_pan))
        jitter_tilt = float(self.rng.normal(0.0, self.jitter_sigma_tilt))

        pan = s.drift_pan + s.ms_pan + jitter_pan
        tilt = s.drift_tilt + s.ms_tilt + jitter_tilt

        q_pan = Quaternion((0.0, 1.0, 0.0), pan)
        q_tilt = Quaternion((1.0, 0.0, 0.0), tilt)

        self.camera.rotation_mode = "QUATERNION"
        self.camera.rotation_quaternion = self.base_rotation @ q_pan @ q_tilt

        return {
            "pan_rad": pan,
            "tilt_rad": tilt,
            "pan_deg": math.degrees(pan),
            "tilt_deg": math.degrees(tilt),
            "drift_pan_deg": math.degrees(s.drift_pan),
            "drift_tilt_deg": math.degrees(s.drift_tilt),
            "ms_pan_deg": math.degrees(s.ms_pan),
            "ms_tilt_deg": math.degrees(s.ms_tilt),
            "jitter_pan_deg": math.degrees(jitter_pan),
            "jitter_tilt_deg": math.degrees(jitter_tilt),
            "microsaccade_active": bool(s.ms_active),
        }
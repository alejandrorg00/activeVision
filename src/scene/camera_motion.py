# Alejandro Rodriguez-Garcia
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from mathutils import Quaternion


@dataclass
class CameraMotionState:
    """
    Camera drift state.

    Values are stored in radians.
    """
    drift_pan: float = 0.0
    drift_tilt: float = 0.0


class DriftCameraController:
    """
    Offline drift-only camera controller.

    Components:
        - slow accumulated drift only

    No microsaccades.
    No frame-wise jitter.

    The perturbation is applied around a fixed base camera orientation.

    Motion model
    ------------
    drift_pan_t = drift_pan_{t-1} + Normal(0, drift_sigma_pan)
    drift_tilt_t = drift_tilt_{t-1} + Normal(0, drift_sigma_tilt)
    """

    def __init__(
        self,
        camera,
        base_rotation_quaternion,
        drift_sigma_deg=(0.01, 0.01),
        seed: int | None = None,
    ):
        self.camera = camera
        self.base_rotation = base_rotation_quaternion.copy()
        self.state = CameraMotionState()

        self.drift_sigma_pan = math.radians(float(drift_sigma_deg[0]))
        self.drift_sigma_tilt = math.radians(float(drift_sigma_deg[1]))

        self.rng = np.random.default_rng(seed)

    def update(self, dt: float | None = None) -> dict:
        """
        Advance drift state by one frame and apply it to the camera.

        dt is accepted for API compatibility but is not used because this is a
        frame-wise random walk with per-frame sigma.
        """
        s = self.state

        delta_drift_pan = float(self.rng.normal(0.0, self.drift_sigma_pan))
        delta_drift_tilt = float(self.rng.normal(0.0, self.drift_sigma_tilt))

        s.drift_pan += delta_drift_pan
        s.drift_tilt += delta_drift_tilt

        q_pan = Quaternion((0.0, 1.0, 0.0), s.drift_pan)
        q_tilt = Quaternion((1.0, 0.0, 0.0), s.drift_tilt)

        self.camera.rotation_mode = "QUATERNION"
        self.camera.rotation_quaternion = self.base_rotation @ q_pan @ q_tilt

        return {
            "pan_rad": float(s.drift_pan),
            "tilt_rad": float(s.drift_tilt),
            "pan_deg": float(math.degrees(s.drift_pan)),
            "tilt_deg": float(math.degrees(s.drift_tilt)),
            "drift_pan_deg": float(math.degrees(s.drift_pan)),
            "drift_tilt_deg": float(math.degrees(s.drift_tilt)),
            "delta_drift_pan_deg": float(math.degrees(delta_drift_pan)),
            "delta_drift_tilt_deg": float(math.degrees(delta_drift_tilt)),
        }
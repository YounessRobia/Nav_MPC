"""Coordinate transformations and rotation utilities."""

import numpy as np
from typing import Tuple


def wrap_angle(angle: float) -> float:
    """
    Wrap angle to [-pi, pi] range.

    Args:
        angle: Angle in radians

    Returns:
        Wrapped angle in [-pi, pi]
    """
    return ((angle + np.pi) % (2 * np.pi)) - np.pi


def wrap_angles(angles: np.ndarray) -> np.ndarray:
    """
    Wrap array of angles to [-pi, pi] range.

    Args:
        angles: Array of angles in radians

    Returns:
        Wrapped angles in [-pi, pi]
    """
    return ((angles + np.pi) % (2 * np.pi)) - np.pi


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert Euler angles (ZYX convention) to rotation matrix.

    The rotation is applied in order: yaw (Z) -> pitch (Y) -> roll (X)
    This transforms vectors from body frame to inertial frame.

    Args:
        roll: Roll angle (phi) in radians
        pitch: Pitch angle (theta) in radians
        yaw: Yaw angle (psi) in radians

    Returns:
        3x3 rotation matrix R_IB (body to inertial)
    """
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    R = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    return R


def rotation_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Extract Euler angles from rotation matrix (ZYX convention).

    Args:
        R: 3x3 rotation matrix

    Returns:
        Tuple of (roll, pitch, yaw) in radians
    """
    # Handle gimbal lock
    if np.abs(R[2, 0]) >= 1.0 - 1e-10:
        # Gimbal lock: pitch = +/- 90 degrees
        yaw = 0.0
        if R[2, 0] < 0:
            pitch = np.pi / 2
            roll = np.arctan2(R[0, 1], R[0, 2])
        else:
            pitch = -np.pi / 2
            roll = np.arctan2(-R[0, 1], -R[0, 2])
    else:
        pitch = -np.arcsin(R[2, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])

    return roll, pitch, yaw


def body_to_inertial(
    vector_body: np.ndarray, roll: float, pitch: float, yaw: float
) -> np.ndarray:
    """
    Transform vector from body frame to inertial frame.

    Args:
        vector_body: 3D vector in body frame
        roll, pitch, yaw: Euler angles in radians

    Returns:
        3D vector in inertial frame
    """
    R = euler_to_rotation_matrix(roll, pitch, yaw)
    return R @ vector_body


def inertial_to_body(
    vector_inertial: np.ndarray, roll: float, pitch: float, yaw: float
) -> np.ndarray:
    """
    Transform vector from inertial frame to body frame.

    Args:
        vector_inertial: 3D vector in inertial frame
        roll, pitch, yaw: Euler angles in radians

    Returns:
        3D vector in body frame
    """
    R = euler_to_rotation_matrix(roll, pitch, yaw)
    return R.T @ vector_inertial


def ned_to_enu(vector_ned: np.ndarray) -> np.ndarray:
    """
    Convert vector from NED (North-East-Down) to ENU (East-North-Up).

    Args:
        vector_ned: 3D vector in NED coordinates

    Returns:
        3D vector in ENU coordinates
    """
    return np.array([vector_ned[1], vector_ned[0], -vector_ned[2]])


def enu_to_ned(vector_enu: np.ndarray) -> np.ndarray:
    """
    Convert vector from ENU (East-North-Up) to NED (North-East-Down).

    Args:
        vector_enu: 3D vector in ENU coordinates

    Returns:
        3D vector in NED coordinates
    """
    return np.array([vector_enu[1], vector_enu[0], -vector_enu[2]])


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """
    Create skew-symmetric matrix from 3D vector.

    Args:
        v: 3D vector

    Returns:
        3x3 skew-symmetric matrix [v]_x such that [v]_x @ w = v x w
    """
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute angle between two vectors.

    Args:
        v1, v2: Input vectors

    Returns:
        Angle in radians [0, pi]
    """
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if v1_norm < 1e-10 or v2_norm < 1e-10:
        return 0.0
    cos_angle = np.clip(np.dot(v1, v2) / (v1_norm * v2_norm), -1.0, 1.0)
    return np.arccos(cos_angle)

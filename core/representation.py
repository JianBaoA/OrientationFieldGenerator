"""
Module: core/representation.py
Responsibility:
  - Provide orientation representation conversions (Euler, rotation matrices, quaternions).
  - Offer math helpers for orientation transforms.

Public API:
  - rotation_matrix_z(theta_deg)
  - rotation_matrix_to_quaternion(R)
"""

import numpy as np
import logging
from typing import Tuple
from dataclasses import dataclass
from math import cos, sin, radians, degrees, acos, atan2
from functools import wraps
import time

logger = logging.getLogger(__name__)

__all__ = [
    "rotation_matrix_z",
    "rotation_matrix_to_quaternion",
    "OrientationRepresentationError",
    "QuaternionNormalizationError",
    "RotationMatrixOrthogonalityError",
    "InvalidEulerAngleError",
    "EulerAngle",
    "Quaternion",
]


def rotation_matrix_z(theta_deg: np.ndarray) -> np.ndarray:
    """绕 Z 轴旋转的矩阵批量生成，theta_deg 可为标量或数组"""
    theta_rad = np.deg2rad(theta_deg)
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    matrix = np.zeros((*theta_rad.shape, 3, 3), dtype=np.float32)
    matrix[..., 0, 0] = cos_t
    matrix[..., 0, 1] = -sin_t
    matrix[..., 1, 0] = sin_t
    matrix[..., 1, 1] = cos_t
    matrix[..., 2, 2] = 1.0
    return matrix


def rotation_matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """将旋转矩阵批量转换为四元数 (w,x,y,z)"""
    original_shape = matrix.shape[:-2]
    m = matrix.reshape(-1, 3, 3)
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    q = np.zeros((m.shape[0], 4), dtype=np.float32)
    positive = trace > 0
    if np.any(positive):
        s = np.sqrt(trace[positive] + 1.0) * 2.0
        q[positive, 0] = 0.25 * s
        q[positive, 1] = (m[positive, 2, 1] - m[positive, 1, 2]) / s
        q[positive, 2] = (m[positive, 0, 2] - m[positive, 2, 0]) / s
        q[positive, 3] = (m[positive, 1, 0] - m[positive, 0, 1]) / s

    negative = ~positive
    if np.any(negative):
        m_neg = m[negative]
        neg_idx = np.where(negative)[0]
        cond1 = m_neg[:, 0, 0] > m_neg[:, 1, 1]
        cond2 = m_neg[:, 0, 0] > m_neg[:, 2, 2]
        idx = cond1 & cond2
        if np.any(idx):
            s = np.sqrt(1.0 + m_neg[idx, 0, 0] - m_neg[idx, 1, 1] - m_neg[idx, 2, 2]) * 2.0
            q[neg_idx[idx], 0] = (m_neg[idx, 2, 1] - m_neg[idx, 1, 2]) / s
            q[neg_idx[idx], 1] = 0.25 * s
            q[neg_idx[idx], 2] = (m_neg[idx, 0, 1] + m_neg[idx, 1, 0]) / s
            q[neg_idx[idx], 3] = (m_neg[idx, 0, 2] + m_neg[idx, 2, 0]) / s

        rest = ~idx
        if np.any(rest):
            cond3 = m_neg[rest, 1, 1] > m_neg[rest, 2, 2]
            idx2 = rest.copy()
            idx2[rest] = cond3
            if np.any(idx2):
                s = np.sqrt(1.0 + m_neg[idx2, 1, 1] - m_neg[idx2, 0, 0] - m_neg[idx2, 2, 2]) * 2.0
                q[neg_idx[idx2], 0] = (m_neg[idx2, 0, 2] - m_neg[idx2, 2, 0]) / s
                q[neg_idx[idx2], 1] = (m_neg[idx2, 0, 1] + m_neg[idx2, 1, 0]) / s
                q[neg_idx[idx2], 2] = 0.25 * s
                q[neg_idx[idx2], 3] = (m_neg[idx2, 1, 2] + m_neg[idx2, 2, 1]) / s

            idx3 = rest.copy()
            idx3[rest] = ~cond3
            if np.any(idx3):
                s = np.sqrt(1.0 + m_neg[idx3, 2, 2] - m_neg[idx3, 0, 0] - m_neg[idx3, 1, 1]) * 2.0
                q[neg_idx[idx3], 0] = (m_neg[idx3, 1, 0] - m_neg[idx3, 0, 1]) / s
                q[neg_idx[idx3], 1] = (m_neg[idx3, 0, 2] + m_neg[idx3, 2, 0]) / s
                q[neg_idx[idx3], 2] = (m_neg[idx3, 1, 2] + m_neg[idx3, 2, 1]) / s
                q[neg_idx[idx3], 3] = 0.25 * s
    return q.reshape(*original_shape, 4)


class OrientationRepresentationError(Exception):
    """取向表征转换过程中的基础异常"""
    pass


class QuaternionNormalizationError(OrientationRepresentationError):
    """四元数归一化失败"""
    pass


class RotationMatrixOrthogonalityError(OrientationRepresentationError):
    """旋转矩阵正交性校验失败"""
    pass


class InvalidEulerAngleError(OrientationRepresentationError):
    """欧拉角数值校验失败"""
    pass


def _log_conversion_attempt(representation_type: str):
    """装饰器：记录取向表征换的尝试与耗时"""
    def decorator(func):
        @wraps(func)
        def function_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            logger.debug(f"开始{representation_type}转换，输入参数: {args[1:] if args else kwargs}")
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start_time
                logger.debug(f"{representation_type}转换成功，耗时: {elapsed:.6f}s")
                return result
            except OrientationRepresentationError:
                raise
            except Exception as e:
                logger.error(f"{representation_type}转换过程中发生未预期错误: {e}", exc_info=True)
                raise OrientationRepresentationError(f"表征转换失败: {e}") from e
        return function_wrapper
    return decorator


def _validate_rotation_matrix_orthogonality(matrix: np.ndarray, tolerance: float = 1e-9) -> None:
    """校验旋转矩阵的正交性"""
    if not np.allclose(matrix @ matrix.T, np.eye(3), atol=tolerance):
        raise RotationMatrixOrthogonalityError("矩阵不满足正交条件")
    if not np.isclose(np.linalg.det(matrix), 1.0, atol=tolerance):
        raise RotationMatrixOrthogonalityError("矩阵行列式不为1")


@dataclass(frozen=True)
class EulerAngle:
    """Bunge欧拉角表征，采用(φ1, Φ, φ2)约定"""
    phi1: float
    phi: float
    phi2: float

    def __post_init__(self) -> None:
        """实例化后立即进行数值校验"""
        object.__setattr__(self, 'phi1', float(self.phi1))
        object.__setattr__(self, 'phi', float(self.phi))
        object.__setattr__(self, 'phi2', float(self.phi2))
        if not all(0.0 <= x <= 360.0 for x in (self.phi1, self.phi2)):
            raise InvalidEulerAngleError("φ1或φ2超出0-360度范围")
        if not 0.0 <= self.phi <= 180.0:
            raise InvalidEulerAngleError("Φ超出0-180度范围")

    @_log_conversion_attempt("欧拉角转四元数")
    def to_quaternion(self) -> 'Quaternion':
        """将欧拉角转换为单位四元数"""
        phi1_half = radians(self.phi1) / 2.0
        phi_half = radians(self.phi) / 2.0
        phi2_half = radians(self.phi2) / 2.0

        c1, s1 = cos(phi1_half), sin(phi1_half)
        c, s = cos(phi_half), sin(phi_half)
        c2, s2 = cos(phi2_half), sin(phi2_half)

        q0 = c1 * c * c2 + s1 * c * s2
        q1 = s1 * c * c2 - c1 * c * s2
        q2 = c1 * s * c2 + s1 * s * s2
        q3 = c1 * s * s2 - s1 * s * c2

        return Quaternion(q0, q1, q2, q3)

    @_log_conversion_attempt("欧拉角转旋转矩阵")
    def to_rotation_matrix(self) -> np.ndarray:
        """将欧拉角转换为3x3旋转矩阵"""
        phi1_rad, phi_rad, phi2_rad = radians(self.phi1), radians(self.phi), radians(self.phi2)

        c1, s1 = cos(phi1_rad), sin(phi1_rad)
        c, s = cos(phi_rad), sin(phi_rad)
        c2, s2 = cos(phi2_rad), sin(phi2_rad)

        matrix = np.array([
            [c1*c2 - s1*c*s2, -c1*s2 - s1*c*c2, s1*s],
            [s1*c2 + c1*c*s2, -s1*s2 + c1*c*c2, -c1*s],
            [s*s2, s*c2, c]
        ], dtype=np.float64)

        _validate_rotation_matrix_orthogonality(matrix)
        return matrix

    def to_radians(self) -> Tuple[float, float, float]:
        """返回弧度制欧拉角"""
        return radians(self.phi1), radians(self.phi), radians(self.phi2)


@dataclass(frozen=True)
class Quaternion:
    """单位四元数表征旋转，采用(q0, q1, q2, q3)约定，q0为标量部分"""
    q0: float
    q1: float
    q2: float
    q3: float

    def __post_init__(self) -> None:
        """校验四元数是否为单位四元数"""
        norm_sq = self.q0**2 + self.q1**2 + self.q2**2 + self.q3**2
        if not np.isclose(norm_sq, 1.0, atol=1e-9):
            raise QuaternionNormalizationError(f"四元数未归一化，模方为{norm_sq}")

    @classmethod
    def from_axis_angle(cls, axis: np.ndarray, angle_degrees: float) -> 'Quaternion':
        """通过旋转轴和旋转角构建单位四元数"""
        if axis.shape != (3,):
            raise ValueError("旋转轴必须为三维数组")
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-12:
            raise QuaternionNormalizationError("旋转轴模长过小，无法归一化")
        axis_normalized = axis / axis_norm
        angle_half_rad = radians(angle_degrees) / 2.0
        q0 = cos(angle_half_rad)
        q_vec = axis_normalized * sin(angle_half_rad)
        return cls(float(q0), float(q_vec[0]), float(q_vec[1]), float(q_vec[2]))

    @_log_conversion_attempt("四元数转欧拉角")
    def to_euler_angle(self) -> EulerAngle:
        """将四元数转换为Bunge欧拉角"""
        q0, q1, q2, q3 = self.q0, self.q1, self.q2, self.q3

        phi = acos(2.0 * (q0**2 + q3**2) - 1.0)
        if abs(phi) < 1e-12 or abs(phi - np.pi) < 1e-12:
            phi1 = atan2(2.0 * (q0*q1 - q2*q3), 1.0 - 2.0 * (q1**2 + q2**2))
            phi2 = 0.0
        else:
            phi1 = atan2((q0*q1 + q2*q3), (q0*q2 - q1*q3))
            phi2 = atan2((q0*q3 + q1*q2), (q0*q2 - q1*q3))

        phi1 = (phi1 + 2*np.pi) % (2*np.pi)
        phi2 = (phi2 + 2*np.pi) % (2*np.pi)

        return EulerAngle(degrees(phi1), degrees(phi), degrees(phi2))

    @_log_conversion_attempt("四元数转旋转矩阵")
    def to_rotation_matrix(self) -> np.ndarray:
        """将四元数转换为3x3旋转矩阵"""
        q0, q1, q2, q3 = self.q0, self.q1, self.q2, self.q3

        matrix = np.array([
            [1-2*(q2**2+q3**2), 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
            [2*(q1*q2+q0*q3), 1-2*(q1**2+q3**2), 2*(q2*q3-q0*q1)],
            [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 1-2*(q1**2+q2**2)]
        ], dtype=np.float64)

        _validate_rotation_matrix_orthogonality(matrix)
        return matrix

    def conjugate(self) -> 'Quaternion':
        """返回共轭四元数"""
        return Quaternion(self.q0, -self.q1, -self.q2, -self.q3)

    def multiply(self, other: 'Quaternion') -> 'Quaternion':
        """四元数乘法，表示旋转的复合"""
        q0 = self.q0*other.q0 - self.q1*other.q1 - self.q2*other.q2 - self.q3*other.q3
        q1 = self.q0*other.q1 + self.q1*other.q0 + self.q2*other.q3 - self.q3*other.q2
        q2 = self.q0*other.q2 - self.q1*other.q3 + self.q2*other.q0 + self.q3*other.q1
        q3 = self.q0*other.q3 + self.q1*other.q2 - self.q2*other.q1 + self.q3*other.q0
        return Quaternion(q0, q1, q2, q3)

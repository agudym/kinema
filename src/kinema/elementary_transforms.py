# Inspired by roboticstoolbox https://github.com/petercorke/robotics-toolbox-python

# Robot Kinematics Playground (Python Library)
# Author: anton.gudym@gmail.com (Anton Gudym)

import numpy as np
import jax
import jax.numpy as jnp
from typing import Optional

jax.config.update("jax_enable_x64", True)

class SE3:
    def __init__(self, mat: np.ndarray | jnp.ndarray | None = None):
        assert mat is None or mat.shape == (4,4)
        if mat is None:
            mat = jnp.eye(4, dtype=jnp.float64)
        else:
            mat = jnp.asarray(mat, dtype=jnp.float64)
        self.mat = mat

    def calc_delta(self, other: 'SE3'):
        d_trans = other.mat[:3, -1] - self.mat[:3, -1]
        R = other.mat[:3, :3] @ self.mat[:3, :3].T
        d_angle = log_SO3(R)
        return jnp.concatenate((d_trans, d_angle))

    def inv(self):
        R_T = self.mat[:3, :3].T
        inv_mat = jnp.eye(4, dtype=jnp.float64)
        inv_mat = inv_mat.at[:3, :3].set(R_T)
        inv_mat = inv_mat.at[:3, 3].set(-R_T @ self.mat[:3, 3])
        return SE3(inv_mat)
    
    def numpy(self) :
        return np.array(self.mat)

    def __mul__(self, other):
        if isinstance(other, SE3):
            return SE3(self.mat @ other.mat)
        elif isinstance(other, ET):
            return ETS([self, other])
        elif isinstance(other, ETS):
            return ETS([self] + other.ets)
        return NotImplemented

    def __matmul__(self, other):
        return self.__mul__(other)


class ET :
    def __init__(self, id: int, qindex:int = -1):
        # Type of the transformation
        self.id = id
        # Index of the parameter in a list (of unique parameters)
        self.qindex = qindex

    @property
    def axis(self):
        return "xyz"[(abs(self.id) - 1) % 3]
    
    @property
    def is_rotation(self):
        return abs(self.id) >= 4

    def inv(self):
        return ET(-self.id, self.qindex)

    def eval(self, v):
        return ET._eval_jax(self.id, v, jnp.eye(4, dtype=jnp.float64))

    @staticmethod
    def tx(q: float | None = None):
        return ET(1) if q is None else SE3(ET(1).eval(q))
    @staticmethod
    def ty(q: float | None = None):
        return ET(2) if q is None else SE3(ET(2).eval(q))
    @staticmethod
    def tz(q: float | None = None):
        return ET(3) if q is None else SE3(ET(3).eval(q))
    @staticmethod
    def Rx(q: float | None = None):
        return ET(4) if q is None else SE3(ET(4).eval(q))
    @staticmethod
    def Ry(q: float | None = None):
        return ET(5) if q is None else SE3(ET(5).eval(q))
    @staticmethod
    def Rz(q: float | None = None):
        return ET(6) if q is None else SE3(ET(6).eval(q))

    def __mul__(self, other):
        if isinstance(other, (ET, SE3)):
            return ETS([self, other])
        elif isinstance(other, ETS):
            return ETS([self] + other.ets)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (ET, SE3)):
            return ETS([other, self])
        return NotImplemented

    @staticmethod
    @jax.jit
    def _eval_jax(id, v, mat):
        v = v * jnp.sign(id)
        abs_id = jnp.abs(id)
        c, s = jnp.cos(v), jnp.sin(v)
        
        m_tx = jnp.array([[1., 0., 0., v], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype=jnp.float64)
        m_ty = jnp.array([[1., 0., 0., 0.], [0., 1., 0., v], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype=jnp.float64)
        m_tz = jnp.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., v], [0., 0., 0., 1.]], dtype=jnp.float64)
        m_rx = jnp.array([[1., 0., 0., 0.], [0., c, -s, 0.], [0., s, c, 0.], [0., 0., 0., 1.]], dtype=jnp.float64)
        m_ry = jnp.array([[c, 0., s, 0.], [0., 1., 0., 0.], [-s, 0., c, 0.], [0., 0., 0., 1.]], dtype=jnp.float64)
        m_rz = jnp.array([[c, -s, 0., 0.], [s, c, 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype=jnp.float64)
        
        return jnp.where(abs_id == 1, m_tx,
                 jnp.where(abs_id == 2, m_ty,
                   jnp.where(abs_id == 3, m_tz,
                     jnp.where(abs_id == 4, m_rx,
                       jnp.where(abs_id == 5, m_ry, 
                         jnp.where(abs_id == 6, m_rz, mat))))))

class ETS :
    def __init__(self, ets_list=None):
        if ets_list is None:
            self.ets = []
        elif isinstance(ets_list, (ET, SE3)):
            self.ets = [ets_list]
        elif isinstance(ets_list, ETS):
            self.ets = ets_list.ets.copy()
        else:
            self.ets = list(ets_list)

        # When ETS is evaluated we cache the metadata
        self._mats, self._ids, self._ets2qindex = None, None, None
        
        self._idxlist = np.arange(len(self.ets))

    @property
    def n(self):
        return sum(1 for et in self.ets if isinstance(et, ET))

    def joints(self):
        return (et for et in self.ets if isinstance(et, ET))

    def inv(self):
        res = []
        for et in reversed(self.ets):
            res.append(et.inv())
        return ETS(res)

    def __iter__(self):
        return iter(self.ets)

    def __mul__(self, other):
        if isinstance(other, (ET, SE3)):
            return ETS(self.ets + [other])
        elif isinstance(other, ETS):
            return ETS(self.ets + other.ets)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (ET, SE3)):
            return ETS([other] + self.ets)
        return NotImplemented

    def eval(self, q):
        if not self._has_cache():
           self._cache()
        
        # ets can heave identical ETs with 1 parameter or static where first q is used as a stub
        if len(q) > 0:
            q_unpack = q[self._ets2qindex]
        else:
            q_unpack = jnp.zeros(len(self.ets))

        return SE3(
            ETS._eval(
                q_unpack, self._ids, self._mats
            )
        )

    @staticmethod
    @jax.jit
    def _eval(q_var, ids_arr, mats_arr):
        mapped_mats = jax.vmap(ET._eval_jax)(ids_arr, q_var, mats_arr)
        def scan_fn(carry, x):
            return carry @ x, None
        return jax.lax.scan(scan_fn, jnp.eye(4, dtype=jnp.float64), mapped_mats)[0]
    
    @staticmethod
    @jax.jit
    @jax.jacrev
    def _jacob(q_var, ids_arr, mats_arr):
        mat = ETS._eval(q_var, ids_arr, mats_arr)
        t = mat[:3, 3]       # translation
        R = mat[:3, :3]      # rotation
        r = log_SO3(R)       # axis-angle vector
        return jnp.concatenate([t, r])

    def jacob0(self, q):
        if not self._has_cache():
            self._cache()
        
        if len(q) > 0:
            q_unpack = q[self._ets2qindex]
        else:
            q_unpack = jnp.zeros(len(self.ets))

        jac_unpacked = np.array(
            ETS._jacob(
                q_unpack, self._ids, self._mats
            )
        )
        
        jac = np.zeros((6, len(q)))
        if len(q) > 0:
            mask = self._ets2qindex >= 0
            jac[:, self._ets2qindex[mask]] = jac_unpacked[:, self._idxlist[mask]]

        return jac
    
    def _has_cache(self) :
        return self._mats is not None and self._ids is not None and self._ets2qindex is not None
    
    def _cache(self) :
        ids = []
        mats = []
        self._ets2qindex = []
        q_idx = 0
        for et in self.ets:
            if isinstance(et, SE3):
                ids.append(0)
                mats.append(et.mat)
                self._ets2qindex.append(-1)
            else:
                ids.append(et.id)
                mats.append(jnp.eye(4, dtype=jnp.float64))
                
                if et.qindex >= 0:
                    self._ets2qindex.append(et.qindex)
                else:
                    self._ets2qindex.append(q_idx)
                q_idx += 1
        self._ets2qindex = np.array(self._ets2qindex, dtype=np.int32)
        self._mats = jnp.array(mats, dtype=jnp.float64)
        self._ids = jnp.array(ids, dtype=jnp.int32)

def log_SO3(R, epsilon: float = 1e-12):
    """
    Rotation matrix 3x3 (SO3) to axis-angle vector (batched version)
    
    Args:
        R: Rotation matrices of shape (..., 3, 3) or (3, 3)
        epsilon: Numerical tolerance
    
    Returns:
        Axis-angle vectors of shape (..., 3)
    """
    xnp = jax.numpy
    I = xnp.eye(3, dtype=jnp.float64)
    
    # Ensure R has at least 3 dimensions for consistent indexing
    original_shape = R.shape
    if R.ndim == 2:
        R = R[np.newaxis, ...]  # Add batch dimension
    
    batch_shape = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3)
    batch_size = R_flat.shape[0]
    
    # Compute trace for each rotation matrix
    trace_R = xnp.trace(R_flat, axis1=1, axis2=2)            # (batch,)
    near_zero_case = xnp.zeros((batch_size, 3), dtype=jnp.float64)

    cos_theta = (trace_R - 1) / 2
    safe_cos = xnp.clip(cos_theta, -1 + epsilon, 1 - epsilon)
    theta = xnp.arccos(safe_cos)
    
    # Find the index of maximum diagonal element for each matrix
    diag = R_flat[:, (0, 1, 2), (0, 1, 2)]  # Shape: (batch_size, 3)
    i = xnp.argmax(diag, axis=1)  # Shape: (batch_size,)
    
    # Create batch indices
    batch_indices = xnp.arange(batch_size)
    
    # Compute w for the near-pi case
    # For each matrix, get the i-th column of I and R
    I_cols = I[:, i]  # Shape: (3, batch_size)
    R_cols = R_flat[batch_indices, :, i]  # Shape: (batch_size, 3)
    
    denominator = 2 * (1 + diag[batch_indices, i])
    # Handle numerical stability for denominator
    safe_denom = xnp.where(denominator < epsilon, epsilon, denominator)
    w = (R_cols + I_cols.T) / xnp.sqrt(safe_denom)[:, None]
    
    # Compute the multiplier
    sin_theta = xnp.sin(theta)
    # Avoid division by zero
    safe_sin_theta = xnp.where(theta < epsilon, 1.0, 2 * sin_theta)
    mul = theta / safe_sin_theta
    
    # Compute the general case (non-degenerate)
    general_case = xnp.stack([
        R_flat[:, 2, 1] - R_flat[:, 1, 2],
        R_flat[:, 0, 2] - R_flat[:, 2, 0], 
        R_flat[:, 1, 0] - R_flat[:, 0, 1]
    ], axis=1) * mul[:, None]
    
    # Handle the near-pi rotation case  
    near_pi_case = w * xnp.pi
    
    # Apply conditions
    result = xnp.where(
        theta[...,None] < epsilon, 
        near_zero_case,
        xnp.where(
            trace_R[...,None] + 1 < epsilon,
            near_pi_case,
            general_case
        )
    )
    
    # Reshape back to original batch shape
    result = result.reshape(*batch_shape, 3)
    
    # If input was a single matrix, remove the batch dimension
    if len(original_shape) == 2:
        result = result[0]
    
    return result


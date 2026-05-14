# Inspired by roboticstoolbox https://github.com/petercorke/robotics-toolbox-python
# https://github.com/jhavl/dkt/blob/main/Part%201/2%20The%20Manipulator%20Jacobian.ipynb

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
        return ET._eval_jax(v, self.id, jnp.eye(4, dtype=jnp.float64), jnp.eye(4, dtype=jnp.float64))

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
    def _eval_jax(v, id, mat_left, mat_right):
        v = v * jnp.sign(id)
        abs_id = jnp.abs(id)
        c, s = jnp.cos(v), jnp.sin(v)
        
        m_tx = jnp.array([[1., 0., 0., v], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype=jnp.float64)
        m_ty = jnp.array([[1., 0., 0., 0.], [0., 1., 0., v], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype=jnp.float64)
        m_tz = jnp.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., v], [0., 0., 0., 1.]], dtype=jnp.float64)
        m_rx = jnp.array([[1., 0., 0., 0.], [0., c, -s, 0.], [0., s, c, 0.], [0., 0., 0., 1.]], dtype=jnp.float64)
        m_ry = jnp.array([[c, 0., s, 0.], [0., 1., 0., 0.], [-s, 0., c, 0.], [0., 0., 0., 1.]], dtype=jnp.float64)
        m_rz = jnp.array([[c, -s, 0., 0.], [s, c, 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype=jnp.float64)
        
        m_et = jnp.where(abs_id == 1, m_tx,
                 jnp.where(abs_id == 2, m_ty,
                   jnp.where(abs_id == 3, m_tz,
                     jnp.where(abs_id == 4, m_rx,
                       jnp.where(abs_id == 5, m_ry, 
                         jnp.where(abs_id == 6, m_rz, jnp.eye(4, dtype=jnp.float64)))))))
                         
        return mat_left @ m_et @ mat_right

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
        self._mats_left, self._mats_right, self._ids, self._ets2qindex = None, None, None, None

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

        return SE3(
            ETS._eval(
                self._unpack(q), self._ids, self._mats_left, self._mats_right
            )
        )
    
    def delta_se3(self, q, pose_dst: SE3) :
        """ Evaluate source pose SE3(q), compute delta to destination SE3 pose """
        if not self._has_cache():
           self._cache()
        
        return ETS._delta_se3(
            pose_dst.mat, self._unpack(q), self._ids, self._mats_left, self._mats_right
        )

    def jacob(self, q):
        if not self._has_cache():
            self._cache()
        
        jac_unpacked = ETS._jacob(
            self._unpack(q), self._ids, self._mats_left, self._mats_right
        )
        
        jac = jnp.zeros((6, len(q)), dtype=jnp.float64)
        if len(q) > 0:
            jac = jac.at[:, self._ets2qindex].add(jac_unpacked)

        return jac

    @staticmethod
    @jax.jit
    def _eval(*args):
        transform_chain = jax.vmap(ET._eval_jax)(*args)
        transform_accum = jnp.eye(4, dtype=jnp.float64)
        for i in range(len(transform_chain)):
            transform_accum = transform_accum @ transform_chain[i]
            
        return transform_accum
    
    @staticmethod
    @jax.jit
    def _jacob(*args):
        # When we compute d / dt R(q(t)), the angular velocity tensor [omega(q)]_x appears:
        #
        # dR(q)/dt = [omega(q)]_x @ R(q)
        # (from d / dt R @ R.T = 0)
        #
        # Applying the chain rule: [omega]_x = (dR/dq * dq/dt) @ R^T.
        # We extract the Spatial Jacobian (J_s) via un-skewing the tensor (dR/dqi @ R^T) for each joint i.
        # J_s maps joint velocities to spatial rotational velocities (omega = J_s(q) * dq/dt).

        mat = ETS._eval(*args)
        # dmat_dq shape: (3, 4, len(q))
        dmat_dq = jax.jacrev(lambda *a: ETS._eval(*a)[:3])(*args)
        skew_mat = jnp.einsum('ijo,jk->iko', dmat_dq[:3, :3, :], mat[:3, :3].T)

        jac_spatial = jnp.stack([
            skew_mat[2, 1, :],
            skew_mat[0, 2, :],
            skew_mat[1, 0, :]
        ], axis=0)

        return jnp.concatenate([dmat_dq[:3, 3, :], jac_spatial], axis=0)

    @staticmethod
    @jax.jit
    def _delta_se3(mat_dst: jnp.ndarray, *args) :
        # Compute the Cartesian error delta between source and destination SE3 poses:
        # e(q) = log_SO3(R_src(q) @ R_dst^T) be the axis-angle error delta.
        #
        # For Inverse Kinematics, by Lie group theory (using taylor series derivatives),
        # the exact derivative of this error vector is:
        #   d(e)/dq = J_l^{-1}(e) @ J_s(q)
        # where J_l^{-1} is the Inverse Left Jacobian of SO(3) and J_s is the Spatial Jacobian.
        #
        # For IK Solvers, when optimizing problem min e(q) -> q, using approximation
        # d(e)/dq ~= J_s(q),
        # works well as the rotation error approaches zero (e -> 0), J_l^{-1}(e) -> Identity matrix,
        # such that optimization goes without differentiating log_SO3.

        mat_src = ETS._eval(*args)
        d_trans = mat_src[:3, -1] - mat_dst[:3, -1]
        d_R = mat_src[:3, :3] @ mat_dst[:3, :3].T
        d_rotvec = log_SO3(d_R)
        return jnp.concatenate((d_trans, d_rotvec))
    
    def _has_cache(self) :
        return self._mats_left is not None and self._ids is not None and self._ets2qindex is not None
    
    def _cache(self) :
        """ Compacting sequence of transformations, preparing metadata for fast jit-operations """
        ids = []
        mats_left = []
        mats_right = []
        self._ets2qindex = []
        q_idx = 0
        
        cur_left = np.eye(4, dtype=np.float64)
        
        nodes = []
        for et in self.ets:
            if isinstance(et, SE3): # Static transformation
                if len(nodes) == 0:
                    cur_left = cur_left @ et.mat
                else:
                    nodes[-1][3] = nodes[-1][3] @ et.mat
            else: # Parametrized transformation
                q_val = et.qindex if et.qindex >= 0 else q_idx
                q_idx += 1
                nodes.append([et.id, q_val, cur_left, np.eye(4, dtype=np.float64)])
                cur_left = np.eye(4, dtype=np.float64)

        if len(nodes) == 0:
            nodes.append([0, 0, cur_left, np.eye(4, dtype=np.float64)])
            
        for n in nodes:
            ids.append(n[0])
            self._ets2qindex.append(n[1])
            mats_left.append(n[2])
            mats_right.append(n[3])
            
        self._ets2qindex = np.array(self._ets2qindex, dtype=np.int32)
        self._mats_left = jnp.array(mats_left, dtype=jnp.float64)
        self._mats_right = jnp.array(mats_right, dtype=jnp.float64)
        self._ids = jnp.array(ids, dtype=jnp.int32)
    
    def _unpack(self, q) :
        """
        Input q is a unique (compact) vector of parameters to ET, while
        list[ET] can heave non unique (repeating) et, we should get 1-1 vector
        of ET's parameters
        """
        if len(q) > 0:
            q_unpack = q[self._ets2qindex]
        else:
            q_unpack = jnp.zeros(len(self._ets2qindex))
        return q_unpack

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


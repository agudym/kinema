# Robot Kinematics Playground (Python Library)
# Author: anton.gudym@gmail.com (Anton Gudym)

from typing import Callable, Tuple, Optional
from dataclasses import dataclass

from enum import Enum
import time
import numpy as np
import scipy

try:
    import sksparse.cholmod
    sparse_cholesky_available = True
except ImportError as e:
    # https://scikit-sparse.readthedocs.io/en/latest/index.html
    # Scikit-sparse not found, fallback to slower linear solver
    # Try 'pip install scikit-sparse', SuiteSparse is required, try for Ubuntu 'sudo apt install libsuitesparse-dev'."
    sparse_cholesky_available = False

class RobustLossType(Enum):
    """
    Cost(x) = Sum_{i=1,n} r_i(x)^2 / 2 ->
    Cost(x) = Sum_{i=1,n} rho((r_i(x)/scale)^2) * scale^2 / 2
    """
    Linear=1 # rho(r^2) = r^2, nothing changes
    Huber=2  # rho(r^2) = (r^2) if (r^2 =< 1) else (2 * |r| - 1)
    Cauchy=3 # rho(r^2) = ln(1 + r^2)

@dataclass
class RobustLoss:
    """ Robustifier functionality """
    type:RobustLossType = RobustLossType.Linear
    # rho(r^2) -> s^2 * rho((r/s)^2)
    scale:float = 1.

    def evaluate(self, residuals_sq:np.ndarray, cost_only:bool):
        """ Evaluate values, derivatives and second derivatives: rho[0],rho[1],rho[2] """
        # TODO avoid re-allocations
        rho = np.empty((3, len(residuals_sq)))

        ss = self.scale**2
        r_sq = residuals_sq / ss

        if self.type == RobustLossType.Linear:
            # Redundant for Linear but left for simplicity
            rho[0] = r_sq

            if not cost_only:
                rho[1, :] = 1
                rho[2, :] = 0
        elif self.type == RobustLossType.Huber:
            mask = r_sq <= 1
            rho[0, mask] = r_sq[mask]
            rho[0, ~mask] = 2 * r_sq[~mask]**0.5 - 1

            if not cost_only:
                rho[1, mask] = 1
                rho[1, ~mask] = r_sq[~mask]**-0.5
                rho[2, mask] = 0
                rho[2, ~mask] = -0.5 * r_sq[~mask]**-1.5
        elif self.type == RobustLossType.Cauchy:
            rho[0] = np.log1p(r_sq)

            if not cost_only:
                t = 1 + r_sq
                rho[1] = 1 / t
                rho[2] = -1 / t**2
        else:
            raise NotImplementedError("Unknown robust loss!")

        rho[0] *= ss
        if not cost_only:
            rho[2] /= ss
            # Common approach is to account 
            # for second! derivative of robust loss rho''(r_sq) (while Gauss-Newton assumes r''(x)~=0)
            # Brilliant description with details:
            # http://ceres-solver.org/nnls_modeling.html#_CPPv4N5ceres12LossFunctionE

            # With rho'(r^2) > 0 (common property of the robust loss),
            # One could note that 2 popular LM+RobustLoss implementations share the following similarity
            # Both do 0-clipping of ether `rho' + 2 * rho'' * residuals_sq`(SCIPY)
            # or explicitly clipping rho'' to be in [0, inf] for "outlier region",
            # moreover it's quite important to do that clip (CERES, see the link above):
            rho[2, rho[2] < 0] = 0

        return rho

class MinimizerLevenbergMarquardt :
    """
    Common implementation of the algorithm, key features are taken from
    https://github.com/ceres-solver/ceres-solver/blob/master/include/ceres/tiny_solver.h
    Additional support of Robust loss added.

    Main Gauss-Newton equations :
    `Cost(x) = Sum_{i=1,n} r_i(x)^2 / 2`, where
    `r_i(x)` is provided residuals,
    `Cost_approx(dx) ~= Cost(x) + Grad_C(x).T * dx + dx.T @ Hessian @ dx / 2`, where
    `Grad_C(x) = J_r.T @ (r1,...,r_n).T`,
    `H ~= J_r.T @ J_r`,
               d_r1/d_x1,  d_r1/d_x2, ...,  d_r1/d_x_n,
               d_r2/d_x1,  d_r2/d_x2, ...,  d_r2/d_x_n,
    J_r(x) =   d_r3/d_x1,  d_r3/d_x2, ...,  d_r3/d_x_n,
              ....
              d_r_m/d_x1, d_r_m/d_x2, ..., d_r_m/d_x_n,
    is provided residuals Jacobian.
    Optimal value comes from linear equation:
        `d_Cost_approx(dx) / d_dx = J_r.T @ (r1,...,r_n).T + J_r.T @ J_r @ dx = 0`

    + multiple Levenberg-Marquard's heuristics...
    """
    def __init__(self,
                 iterations_max: int = 50,
                 robust_loss:RobustLoss|None = None,
                 tol_cost_agrad:float = 1e-10,
                 tol_param_rstep:float = 1e-8,
                 tol_cost_astep:float = 1e-6,
                 diag_min_val:float = 1e-6):
        # Termination criteria
        self.iterations_max: int = iterations_max
        # see `RobustLoss`
        self.robust_loss = robust_loss if robust_loss is not None else RobustLoss()
        # max(Grad_C(x)) < tol_grad
        self.tol_cost_agrad:float = tol_cost_agrad
        # ||x_new - x_old|| <= tol_x * (||x_new|| + tol_x)
        self.tol_param_rstep:float = tol_param_rstep
        # |Cost(x_new) - Cost(x_old)| < tol_cost_step
        self.tol_cost_astep:float = tol_cost_astep
        # tol_cost_val > ||r(x)||^2 / 2
        self.tol_cost_val:float = np.finfo(np.float64).eps**2
        # Avoid zeros on JTJ-diagonal with clipping
        self.diag_min_val = diag_min_val

        self.initial_trust_region_radius:float = 1e4
        self.timer_update = [0]
        self.timer_solve = [0]

    def run(self, x:np.ndarray, func_residuals:Callable, jacobian:Callable, *args,
             weights:Optional[np.ndarray]=None):
        """
        Parameters:
        -----------
        x : np.ndarray
            Initial parameter vector.
        func_residuals : Callable
            Function returning residuals f(x).
        jacobian : Callable
            Function returning Jacobian df/dx(x).
        *args
            Additional arguments passed to func_residuals and jacobian.
        weights : Optional[np.ndarray]
            Optional per-residual weights. If provided, the optimization uses
            weighted residuals f_w(x) = w * f(x) and weighted Jacobian df_w/dx = w * df/dx.
            Shape must match the number of residuals.

        Returns:
        --------
        `Tuple[np.ndarray, Tuple[str, Termination, State]]` - optimized x and description,
        termination type and optimizer state
        """
        self.timer_update = []
        self.timer_solve = []
        state = MinimizerLevenbergMarquardt.State(func_residuals, jacobian, self.robust_loss, weights)
        
        u = 1. / self.initial_trust_region_radius
        v = 2.
        def is_state_update_required():
            return v == 2
        for iteration in range(1, self.iterations_max+1):
            if is_state_update_required() :
                upd_start_time = time.time()
                state.update(x, *args)
                self.timer_update.append(time.time() - upd_start_time)
                if state.gradient_max_norm < self.tol_cost_agrad :
                    return x, self._create_result_desc(Termination.GradNorm, iteration, state)
                
                if state.cost < self.tol_cost_val :
                    return x, self._create_result_desc(Termination.Cost, iteration, state)

            diagonal_jtj = state._jtj.diagonal()
            # Clipping constant from 
            # https://github.com/ceres-solver/ceres-solver/blob/master/include/ceres/tiny_solver.h#L253C51-L253C63
            diagonal_jtj = np.clip(diagonal_jtj * (1 + u), self.diag_min_val, 1e32)
            
            try:
                solve_start_time = time.time()
                jtj_r = state._jtj.copy()
                if state._issparse_jac:
                    jtj_r.setdiag(diagonal_jtj)
                    if not sparse_cholesky_available:
                        lm_step = scipy.sparse.linalg.splu(jtj_r.tocsc()).solve(-state._gradients)
                    else:
                        lm_step = sksparse.cholmod.cholesky(jtj_r.tocsc())(-state._gradients)
                else:
                    np.fill_diagonal(jtj_r, diagonal_jtj)
                    lm_step = scipy.linalg.cho_solve(
                        scipy.linalg.cho_factor(jtj_r), -state._gradients)
                self.timer_solve.append(time.time() - solve_start_time)

                dx = state._scale_jac * lm_step
                dx_norm = float(np.linalg.norm(dx))
                if dx_norm < self.tol_param_rstep * (np.linalg.norm(x) + self.tol_param_rstep) :
                    return x, self._create_result_desc(Termination.DeltaParamNorm, iteration, state, dx_norm=dx_norm) 
                x_new = x + dx

                upd_start_time = time.time()
                cost_change = state.cost - state.calc_residuals_and_cost(x_new, *args, cost_only=True)
                cost_change_model = lm_step @ (-state._gradients - state._jtj @ lm_step / 2)
                self.timer_update[-1] += time.time() - upd_start_time

                rho = cost_change / cost_change_model
            except Exception as e:
                rho = 0
                cost_change = np.nan
                # print(e)

            if rho > 0: # successful iteration
                x = x_new
                if np.abs(cost_change) < self.tol_cost_astep :
                    state.update(x, *args)
                    return x, self._create_result_desc(Termination.DeltaCost, iteration, state, cost_change=cost_change) 
                u *= np.max((1. / 3., 1. - (2 * rho - 1)**3))
                v = 2.
            else: # unsuccessful iteration
                if np.abs(cost_change) < self.tol_cost_astep :
                    return x, self._create_result_desc(Termination.DeltaCost, iteration, state, cost_change=cost_change)
                # Reduce the size of the trust region.
                u *= v
                v *= 2.

        return x, self._create_result_desc(Termination.IterationsLimit, iteration, state)

    def _create_result_desc(self,
                     term_type: "Termination",
                     iteration:int,
                     state: "State",
                     dx_norm:Optional[float]=None,
                     cost_change:Optional[float]=None):
        msg  = f"Lev-Marq InvKin Iterations: {iteration} All - {state._update_count} Jacobian-Update.\n"
        if iteration > 1:
            msg += f" Time(avg) [sec]: {np.mean(self.timer_solve):.3f}(solve), {np.mean(self.timer_update):.3f}(update).\n"
        
        if term_type == Termination.GradNorm:
            msg += f"Termination by Gradient max-norm {state.gradient_max_norm} is too small.\n"
        elif term_type == Termination.Cost:
            msg += f"Termination by Cost: {state.cost} is too small.\n"
        elif term_type == Termination.DeltaParamNorm:
            assert dx_norm is not None
            msg += f"Termination by Parameter-step: {dx_norm} is too small.\n"
        elif term_type == Termination.DeltaCost:
            assert cost_change is not None
            msg += f"Termination by Cost-step: {cost_change} is too small.\n"
        elif term_type == Termination.IterationsLimit:
            msg += "Termination by Iterations limit reached."
        else:
            raise NotImplementedError("Unsupported termination type!")
        
        return msg, term_type, state
    
    class State :
        def __init__(self,
                     func_residuals:Callable,
                     func_jacobian:Callable,
                     robust_loss:RobustLoss,
                     weights:Optional[np.ndarray]=None):
            self._robust_loss = robust_loss
            self._update_count = 0
            # f(x)
            self._func_residuals = func_residuals
            # df/dx(x)
            self._func_jacobian = func_jacobian
            # Optional per-residual weights: f_w(x) = w * f(x), df_w/dx = w * df/dx
            self._weights = weights
            
            # Sum f_i(x_k)^2 / 2
            # sum of Squared, Robustified (if Robust Kernel is used) residuals
            self.cost = None
            # original non-robustified residuals f_i(x_k)
            self.residuals = None

            self._gradients = None
            # J(x_k) = df/dx(x_k)
            self._jac = None
            self._jtj = None
            # J_s(x_k) = J @ diag(scale_jac)
            self._scale_jac = None
            # Is jacobian sparse ?
            self._issparse_jac = False

        @staticmethod
        def issprase(arr):
            return scipy.sparse.issparse(arr)

        @property
        def iterations(self) :
            """ Total amount of cost/jacobian updates """
            return self._update_count

        def update(self, x:np.ndarray, *args):
            self._update_count += 1

            self.cost, self.residuals, residuals_w, jac_rho_weights = self.calc_residuals_and_cost(x, *args)

            self._jac = self._func_jacobian(x, *args)
            self._issparse_jac = self.issprase(self._jac)

            # Combine per-residual weights and robust loss weights: df_w/dx = w * df/dx
            jac_weights = self._weights if jac_rho_weights is None else (
                jac_rho_weights if self._weights is None else self._weights * jac_rho_weights)
            if jac_weights is not None:
                if self._issparse_jac:
                    self._jac = self._jac.multiply(jac_weights[:, np.newaxis])
                else:
                    self._jac = jac_weights[:, np.newaxis] * self._jac  # row-wise scaling

            if self._scale_jac is None:
                if self._issparse_jac:
                    scale = scipy.sparse.linalg.norm(self._jac, axis=0)
                elif isinstance(self._jac, np.ndarray) :
                    scale = np.linalg.norm(self._jac, axis=0)
                else:
                    raise ValueError("Unsupported jacobian type. Should be either numpy array or scipy-sparse COO-array.")
                self._scale_jac = 1 / (1 + scale)

            self._jac *= self._scale_jac # col-wise scaling
            self._jtj = self._jac.T @ self._jac
            self._gradients = self._jac.T @ residuals_w
            self.gradient_max_norm = np.max(np.abs(self._gradients))
        
        def calc_residuals_and_cost(self, x:np.ndarray, *args, cost_only:bool=False):
            # residuals_w for optimization and residuals for visualization
            residuals_w = residuals = self._func_residuals(x, *args)

            # Apply per-residual weights: f_w(x) = w * f(x)
            if self._weights is not None:
                residuals_w = self._weights * residuals
            residuals_sq = residuals_w**2

            rho = self._robust_loss.evaluate(residuals_sq, cost_only)

            jac_rho_weights = None
            if self._robust_loss.type != RobustLossType.Linear and not cost_only:
                # RE-weighting residuals and Jacobian
                jac_rho_weights = (rho[1] + 2 * rho[2] * residuals_sq)**0.5
                residuals_w = residuals_w * rho[1] / jac_rho_weights

            cost = np.sum(rho[0]) / 2

            if cost_only:
                return cost
            else:
                return cost, residuals, residuals_w, jac_rho_weights

class Termination(Enum):
    # max(Grad_C(x)) < tol_grad
    GradNorm = 1
    # tol_cost_val > ||r(x)||^2 / 2
    Cost = 2
    # ||x_new - x_old|| <= tol_x * (||x_new|| + tol_x)
    DeltaParamNorm = 3
    # |Cost(x_new) - Cost(x_old)| < tol_cost_step
    DeltaCost = 4
    IterationsLimit = 5

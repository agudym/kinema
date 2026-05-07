import numpy as np
import scipy.sparse
from numpy.random import MT19937, RandomState, SeedSequence

import jax
import jax.numpy as jnp

from kinema.optimization import MinimizerLevenbergMarquardt, RobustLoss, RobustLossType

jax.config.update("jax_enable_x64", True)

def test_optimization_lm():
    N = 250
    rs = RandomState(MT19937(SeedSequence(0)))
    x_gt = rs.rand(N) * 2 - 1
    
    def function_jax(x, ftype):
        if ftype == 0:
            residuals = (x - x_gt) / N**0.5
        else:
            residuals = (((x - x_gt)**2 + (x[0] - x_gt[0])**2) / N)**0.5
        if ftype >= 2:
            residual_outlier = (x[0] - 100500) / N**0.5
            residuals = jnp.r_[residuals, residual_outlier]
        return residuals
    
    jacobian_jax = jax.jit(jax.jacobian(function_jax), static_argnames="ftype")

    function = lambda x, *args: np.array(function_jax(x, *args))
    
    for jacobian_struct in ("DENSE", "SPARSE"):
        if jacobian_struct == "DENSE":
            jacobian = lambda x, *args: np.array(jacobian_jax(x, *args))
        else:
            jacobian = lambda x, *args: scipy.sparse.coo_array(jacobian_jax(x, *args))
        
        for ftype, losstype in zip((0, 1, 2, 3), (RobustLossType.Linear, RobustLossType.Linear, RobustLossType.Cauchy, RobustLossType.Linear)):
            weights = None
            if ftype == 3:
                weights = np.ones(N + 1)
                weights[-1] = 0

            minimizer = MinimizerLevenbergMarquardt(robust_loss=RobustLoss(losstype, 0.05))
            x_est, (desc, termcrit, state) = minimizer.run( np.zeros(x_gt.shape), function, jacobian, ftype, weights=weights)

            x_err = np.abs(x_est - x_gt)
            x_err_max_id = np.argmax(x_err)

            assert x_err[x_err_max_id] < 1e-5

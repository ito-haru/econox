# src/econox/strategies/optimizer.py
"""
Optimization and Fixed-Point strategies using Optimistix.
Wraps numerical solvers to provide a consistent interface for Econox components.
"""

from typing import Callable, Any
import equinox as eqx
import optimistix as optx
from jaxtyping import PyTree, Scalar, Array

# =============================================================================
# 1. Minimizer (For Estimator)
# =============================================================================

class Minimizer(eqx.Module):
    """
    Wrapper for optimistix.minimize.
    Implements the econox.protocols.Optimizer interface.
    
    You can customize the method and tolerances at initialization.
    
    Examples:
        >>> # Default (BFGS, tol=1e-6)
        >>> opt = Minimizer()
        
        >>> # Custom tolerance
        >>> opt = Minimizer(rtol=1e-3, atol=1e-3)
        
        >>> # Custom method (e.g., Nelder-Mead)
        >>> opt = Minimizer(method=optx.NelderMead())
    """
    rtol: float = 1e-6
    atol: float = 1e-6
    method: optx.AbstractMinimiser = optx.BFGS(rtol=rtol, atol=atol)
    max_steps: int = 1000
    throw: bool = False

    def minimize(
        self, 
        loss_fn: Callable[[PyTree], Scalar], 
        init_params: PyTree
    ) -> PyTree:
        """
        Minimizes the loss function using the specified method and tolerances.
        """
        # Inject the wrapper's tolerances into the solver instance
        # This ensures consistent behavior even if the user swapped the method
        solver = eqx.tree_at(
            lambda s: (s.rtol, s.atol),
            self.method,
            (self.rtol, self.atol)
        )

        def wrapped_loss_fn(params, args) -> Array:
            return loss_fn(params)
        
        sol = optx.minimise(
            fn=wrapped_loss_fn,
            solver=solver,
            y0=init_params,
            max_steps=self.max_steps,
            throw=self.throw
        )
        
        return sol.value


# =============================================================================
# 2. Fixed Point Solver (For Inner/Outer Solvers)
# =============================================================================

class FixedPoint(eqx.Module):
    """
    Wrapper for optimistix.fixed_point.
    Used by Solvers to find Value Functions (Bellman) or Equilibrium (GE).
    
    Attributes:
        method: The fixed-point algorithm (default: FixedPointIteration).
        rtol: Relative tolerance.
        atol: Absolute tolerance.
    """
    
    rtol: float = 1e-8
    atol: float = 1e-8
    method: optx.AbstractFixedPointSolver = optx.FixedPointIteration(rtol=rtol, atol=atol)
    max_steps: int = 2000
    throw: bool = False

    def find_fixed_point(
        self, 
        step_fn: Callable[[PyTree, Any], PyTree], 
        init_val: PyTree,
        args: Any = None
    ) -> PyTree:
        """
        Solves for y such that y = step_fn(y, args).
        """
        # Inject tolerances
        solver = eqx.tree_at(
            lambda s: (s.rtol, s.atol),
            self.method,
            (self.rtol, self.atol)
        )

        sol = optx.fixed_point(
            fn=step_fn,
            solver=solver,
            y0=init_val,
            args=args,
            max_steps=self.max_steps,
            throw=self.throw
        )
        return sol.value
# src/econox/strategies/optimizer.py
"""
Optimization and Fixed-Point strategies using Optimistix.
Wraps numerical solvers to provide a consistent interface for Econox components.
"""

from typing import Callable, Any
import equinox as eqx
import optimistix as optx
from jaxtyping import Float, PyTree, Scalar, Array, Bool, Int

# =============================================================================
# 1. Optimization Strategies
# =============================================================================

class OptimizationResult(eqx.Module):
    """
    A generic container for optimization results.
    Decouples the Estimator from the specific backend (optimistix/jaxopt).

    Attributes:
        params: The optimized parameters (PyTree).
        loss: The final loss value (Scalar).
        success: Whether the optimization was successful (Bool).
        steps: Number of optimization steps taken (Int).
    """
    params: PyTree
    loss: Scalar
    success: Bool[Array, ""]
    steps: Int[Array, ""]

class Minimizer(eqx.Module):
    """
    Wrapper for optimistix.minimise.
    Implements the econox.protocols.Optimizer interface.
    
    You can customize the method and tolerances at initialization.
    
    Examples:
        >>> # Default (BFGS, tol=1e-6)
        >>> opt = Minimizer()
        
        >>> # Custom method (e.g., Nelder-Mead) and tolerances
        >>> opt = Minimizer(method=optx.NelderMead(atol=1e-5, rtol=1e-5))
    """
    method: optx.AbstractMinimiser = optx.BFGS(rtol=1e-6, atol=1e-6)
    max_steps: int = eqx.field(static=True, default=1000)
    throw: bool = eqx.field(static=True, default=False)

    def minimize(
        self, 
        loss_fn: Callable[[PyTree], Scalar], 
        init_params: PyTree
    ) -> OptimizationResult:
        """
        Minimizes the loss function using the specified method and tolerances.

        Parameters
        ----------
        loss_fn : Callable[[PyTree], Scalar]
            The loss function to minimize. Takes parameters and returns a scalar loss.
        init_params : PyTree
            Initial parameter values for optimization.
    
        Returns
        -------
        OptimizationResult
            Contains the optimized parameters, final loss, success status, and iteration count.
        """
        # Inject the wrapper's tolerances into the solver instance
        # This ensures consistent behavior even if the user swapped the method
        def wrapped_loss_fn(params, args) -> tuple[Scalar, Scalar]:
            loss = loss_fn(params)
            return loss, loss
        
        sol:optx.Solution = optx.minimise(
            fn=wrapped_loss_fn,
            solver=self.method,
            y0=init_params,
            max_steps=self.max_steps,
            throw=self.throw,
            has_aux=True
        )
        params: PyTree = sol.value
        success: Bool = sol.result == optx.RESULTS.successful
        final_loss: Float[Array, ""] = sol.aux
        steps: Int = sol.stats["num_steps"]

        result: OptimizationResult = OptimizationResult(
            params=params,
            loss=final_loss,
            success=success,
            steps=steps
        )
        return result


# =============================================================================
# 2. Fixed Point Strategies
# =============================================================================

class FixedPointResult(eqx.Module):
    """
    Container for fixed-point computation results.
    Used by internal solvers (Bellman, Equilibrium) to report convergence status.

    Attributes:
        value: The computed fixed-point value (PyTree).
        success: Whether the fixed-point iteration was successful (Bool).
        steps: Number of iterations taken (Int).
    """
    value: PyTree
    success: Bool[Array, ""]
    steps: Int[Array, ""]

class FixedPoint(eqx.Module):
    """
    Wrapper for optimistix.fixed_point.
    
    Examples:
        >>> # Default (FixedPointIteration)
        >>> fp = FixedPoint()
        
        >>> # Custom (Anderson Acceleration)
        >>> fp = FixedPoint(method=optx.AndersonAcceleration(rtol=1e-5, atol=1e-5))
    """
    method: optx.AbstractFixedPointSolver = optx.FixedPointIteration(rtol=1e-8, atol=1e-8)
    max_steps: int = eqx.field(static=True, default=2000)
    throw: bool = eqx.field(static=True, default=False)

    def find_fixed_point(
        self, 
        step_fn: Callable[[PyTree, Any], PyTree], 
        init_val: PyTree,
        args: Any = None
    ) -> FixedPointResult:
        """
        Solves for y such that y = step_fn(y, args).
        Returns a FixedPointResult containing the solution and status.

        Parameters
        ----------
        step_fn : Callable[[PyTree, Any], PyTree]
            The fixed-point function. Takes current value and args, returns next value.
        init_val : PyTree
            Initial guess for the fixed-point iteration.
        args : Any, optional
            Additional arguments passed to the fixed-point function.
        """
        
        sol = optx.fixed_point(
            fn=step_fn,
            solver=self.method,
            y0=init_val,
            args=args,
            max_steps=self.max_steps,
            throw=self.throw
        )

        return FixedPointResult(
            value=sol.value,
            success=(sol.result == optx.RESULTS.successful),
            steps=sol.stats["num_steps"]
        )

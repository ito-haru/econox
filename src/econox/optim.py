# src/econox/optim.py
"""
Optimization and Fixed-Point strategies using Optimistix.
Wraps numerical solvers to provide a consistent interface for Econox components.
"""

from typing import Callable, Any
import equinox as eqx
import lineax as lx
import optimistix as optx
from jaxtyping import Float, PyTree, Scalar, Array, Bool, Int

# =============================================================================
# 1. Optimization Strategies
# =============================================================================

class MinimizerResult(eqx.Module):
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
        >>> # Default (LBFGS, tol=1e-6)
        >>> opt = Minimizer()
        
        >>> # Custom method (e.g., Nelder-Mead) and tolerances
        >>> opt = Minimizer(method=optx.NelderMead(atol=1e-5, rtol=1e-5))
    """
    method: optx.AbstractMinimiser = optx.LBFGS(rtol=1e-6, atol=1e-6)
    max_steps: int = eqx.field(static=True, default=1000)
    throw: bool = eqx.field(static=True, default=False)

    def minimize(
        self, 
        loss_fn: Callable[[PyTree, Any], Scalar], 
        init_params: PyTree,
        args: Any = None
    ) -> MinimizerResult:
        """
        Minimizes the loss function using the specified method and tolerances.

        Args:
            loss_fn (Callable[[PyTree, Any], Scalar]): The loss function to minimize.
                Takes parameters and additional arguments, returns a scalar loss.
            init_params (PyTree): Initial parameter values for optimization.
            args (Any, optional): Additional arguments passed to the loss function.
                Defaults to None.

        Returns:
            MinimizerResult: Contains the optimized parameters, final loss, 
            success status, and iteration count.
        """
        def wrapped_loss_fn(params, args) -> tuple[Scalar, Scalar]:
            loss = loss_fn(params, args)
            return loss, loss
        
        sol: optx.Solution = optx.minimise(
            fn=wrapped_loss_fn,
            solver=self.method,
            y0=init_params,
            args=args,
            max_steps=self.max_steps,
            throw=self.throw,
            has_aux=True
        )
        params: PyTree = sol.value
        success: Bool[Array, ""] = sol.result == optx.RESULTS.successful
        final_loss: Float[Array, ""] = sol.aux
        steps: Int[Array, ""] = sol.stats["num_steps"]

        result: MinimizerResult = MinimizerResult(
            params=params,
            loss=final_loss,
            success=success,
            steps=steps
        )
        return result
    
    @property
    def method_name(self) -> str:
        """Returns the name of the optimization method used."""
        return self.method.__class__.__name__


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
        >>> # Uses default max_steps (2000) and tolerances (rtol=1e-8, atol=1e-8)
        >>> fp = FixedPoint() 

        >>> # Custom 
        >>> fp = FixedPoint(method=optx.FixedPointIteration(rtol=1e-10, atol=1e-10), max_steps=5000)
    """
    method: optx.AbstractFixedPointSolver = optx.FixedPointIteration(rtol=1e-8, atol=1e-8)
    max_steps: int = eqx.field(static=True, default=2000)
    throw: bool = eqx.field(static=True, default=False)
    adjoint: optx.AbstractAdjoint = optx.ImplicitAdjoint(linear_solver=lx.AutoLinearSolver(well_posed=False))

    def find_fixed_point(
        self, 
        step_fn: Callable[[PyTree, Any], PyTree], 
        init_val: PyTree,
        args: Any = None
    ) -> FixedPointResult:
        r"""
        Solves for :math:`y` such that :math:`y = \text{step\_fn}(y, \text{args})`.
        Returns a FixedPointResult containing the solution and status.

        Args:
            step_fn (Callable[[PyTree, Any], PyTree]): The fixed-point function. Takes current value and args, returns next value.
            init_val (PyTree): Initial guess for the fixed-point iteration.
            args (Any, optional): Additional arguments passed to the fixed-point function.
        
        Returns:
            FixedPointResult: Contains the fixed-point value, success status, and iteration count.
        """
        
        sol = optx.fixed_point(
            fn=step_fn,
            solver=self.method,
            y0=init_val,
            args=args,
            max_steps=self.max_steps,
            throw=self.throw,
            adjoint=self.adjoint
        )

        return FixedPointResult(
            value=sol.value,
            success=(sol.result == optx.RESULTS.successful),
            steps=sol.stats["num_steps"]
        )

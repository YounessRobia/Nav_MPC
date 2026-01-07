"""CasADi solver interface for MPC."""

import casadi as ca
import numpy as np
from typing import Dict, Any, Optional


class CasADiSolver:
    """
    Wrapper around CasADi's Opti interface for MPC problems.

    Provides convenient setup, warm-starting, and solver configuration.
    """

    def __init__(
        self,
        nx: int,
        nu: int,
        N: int,
        solver_opts: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize CasADi solver interface.

        Args:
            nx: State dimension
            nu: Control dimension
            N: Prediction horizon
            solver_opts: IPOPT solver options
        """
        self.nx = nx
        self.nu = nu
        self.N = N

        # Default solver options
        self.solver_opts = {
            "ipopt.max_iter": 100,
            "ipopt.tol": 1e-6,
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.warm_start_bound_push": 1e-9,
            "ipopt.warm_start_mult_bound_push": 1e-9,
            "ipopt.mu_init": 1e-5,
        }
        if solver_opts:
            self.solver_opts.update(solver_opts)

        # Create Opti instance
        self.opti = ca.Opti()
        self._setup_variables()

    def _setup_variables(self):
        """Set up optimization variables."""
        # Decision variables
        self.X = self.opti.variable(self.nx, self.N + 1)  # States
        self.U = self.opti.variable(self.nu, self.N)  # Controls

        # Parameters (set at solve time)
        self.x0_param = self.opti.parameter(self.nx)  # Initial state
        self.x_ref_param = self.opti.parameter(self.nx, self.N + 1)  # Reference trajectory

        # Previous solution for warm start
        self._prev_X = None
        self._prev_U = None

    def set_dynamics_constraint(self, Ad: np.ndarray, Bd: np.ndarray):
        """
        Set linear dynamics constraints.

        x[k+1] = Ad @ x[k] + Bd @ u[k]

        Args:
            Ad: Discrete state matrix (nx, nx)
            Bd: Discrete input matrix (nx, nu)
        """
        for k in range(self.N):
            self.opti.subject_to(
                self.X[:, k + 1] == Ad @ self.X[:, k] + Bd @ self.U[:, k]
            )

    def set_initial_state_constraint(self):
        """Set initial state constraint: x[0] = x0."""
        self.opti.subject_to(self.X[:, 0] == self.x0_param)

    def set_state_bounds(
        self,
        x_min: np.ndarray,
        x_max: np.ndarray,
    ):
        """
        Set state bounds.

        Args:
            x_min: Lower bounds (nx,)
            x_max: Upper bounds (nx,)
        """
        for k in range(self.N + 1):
            self.opti.subject_to(self.opti.bounded(x_min, self.X[:, k], x_max))

    def set_control_bounds(
        self,
        u_min: np.ndarray,
        u_max: np.ndarray,
    ):
        """
        Set control bounds.

        Args:
            u_min: Lower bounds (nu,)
            u_max: Upper bounds (nu,)
        """
        for k in range(self.N):
            self.opti.subject_to(self.opti.bounded(u_min, self.U[:, k], u_max))

    def configure_solver(self):
        """Configure IPOPT solver with current options."""
        self.opti.solver("ipopt", self.solver_opts)

    def set_initial_guess(
        self,
        X_init: Optional[np.ndarray] = None,
        U_init: Optional[np.ndarray] = None,
    ):
        """
        Set initial guess for warm-starting.

        Args:
            X_init: Initial state trajectory (nx, N+1)
            U_init: Initial control trajectory (nu, N)
        """
        if X_init is not None:
            self.opti.set_initial(self.X, X_init)
        if U_init is not None:
            self.opti.set_initial(self.U, U_init)

    def warm_start_from_previous(self):
        """Shift previous solution for warm start."""
        if self._prev_X is not None and self._prev_U is not None:
            # Shift state trajectory
            X_init = np.zeros((self.nx, self.N + 1))
            X_init[:, :-1] = self._prev_X[:, 1:]
            X_init[:, -1] = self._prev_X[:, -1]  # Repeat last

            # Shift control trajectory
            U_init = np.zeros((self.nu, self.N))
            U_init[:, :-1] = self._prev_U[:, 1:]
            U_init[:, -1] = self._prev_U[:, -1]  # Repeat last

            self.set_initial_guess(X_init, U_init)

    def store_solution(self, sol):
        """Store solution for warm-starting next solve."""
        try:
            self._prev_X = sol.value(self.X)
            self._prev_U = sol.value(self.U)
        except Exception:
            pass  # Solution may be invalid

    def solve(
        self,
        x0: np.ndarray,
        x_ref: np.ndarray,
        warm_start: bool = True,
    ) -> Dict[str, Any]:
        """
        Solve the MPC problem.

        Args:
            x0: Initial state (nx,)
            x_ref: Reference trajectory (nx, N+1) or (nx,) for constant
            warm_start: Whether to use warm-starting

        Returns:
            Dictionary with solution data
        """
        # Set parameters
        self.opti.set_value(self.x0_param, x0)

        # Handle constant or trajectory reference
        if x_ref.ndim == 1:
            x_ref_full = np.tile(x_ref.reshape(-1, 1), (1, self.N + 1))
        else:
            x_ref_full = x_ref
        self.opti.set_value(self.x_ref_param, x_ref_full)

        # Warm start
        if warm_start:
            self.warm_start_from_previous()

        # Solve
        try:
            sol = self.opti.solve()
            success = True
            status = "optimal"
        except RuntimeError as e:
            sol = self.opti.debug
            success = False
            status = str(e)

        # Extract solution
        result = {
            "success": success,
            "status": status,
            "X": sol.value(self.X) if success else None,
            "U": sol.value(self.U) if success else None,
            "cost": sol.value(self.opti.f) if success else None,
        }

        # Store for warm start
        if success:
            self.store_solution(sol)

        return result


def create_ipopt_options(
    max_iter: int = 100,
    tolerance: float = 1e-6,
    print_level: int = 0,
    warm_start: bool = True,
) -> Dict[str, Any]:
    """
    Create IPOPT solver options dictionary.

    Args:
        max_iter: Maximum iterations
        tolerance: Convergence tolerance
        print_level: Output verbosity (0=silent, 5=verbose)
        warm_start: Enable warm-starting

    Returns:
        Options dictionary for CasADi
    """
    opts = {
        "ipopt.max_iter": max_iter,
        "ipopt.tol": tolerance,
        "ipopt.print_level": print_level,
        "print_time": 0,
    }

    if warm_start:
        opts.update(
            {
                "ipopt.warm_start_init_point": "yes",
                "ipopt.warm_start_bound_push": 1e-9,
                "ipopt.warm_start_mult_bound_push": 1e-9,
                "ipopt.mu_init": 1e-5,
            }
        )

    return opts

import numpy as np
from typing import Union, Optional
from rich.traceback import install

install()  # for prettier error traces


class Domain:
    """
    Generates the spatial and temporal grid for finite-volume discretization
    on a staggered C‑grid (eta at cell centres, u at cell interfaces).

    The domain is [L0, L] (periodic by default) and the time interval [t0, T].
    The user may provide either (N, M) or (dx, dt); the missing ones are computed.
    """

    def __init__(
        self,
        t0: float = 0.0,
        T: float = 2.0,
        L: float = 4.0,
        L0: Optional[float] = None,
        N: int = 4096,
        M: int = 1024,
        dx: Optional[float] = None,
        dt: Optional[float] = None,
    ):
        # Time parameters
        self.t0 = t0
        self.T = T

        # Space parameters
        self.L = L
        self.L0 = L0 if L0 is not None else -L   # ensure symmetry if not given

        # --- Spatial discretization ---
        if dx is not None:
            self.N = int(np.ceil((self.L - self.L0) / dx))
            self.dx = (self.L - self.L0) / self.N
        else:
            self.N = int(N)
            self.dx = (self.L - self.L0) / self.N

        # --- Temporal discretization ---
        if dt is not None:
            self.M = int(np.ceil((self.T - self.t0) / dt))
            self.dt = (self.T - self.t0) / self.M
        else:
            self.M = int(M)
            self.dt = (self.T - self.t0) / self.M

        # Courant number (for advection with unit speed)
        self.cfl = self.dt / self.dx

        # Time grid: M points, excluding the final one (for periodic time stepping)
        self.time = np.linspace(self.t0, self.T, self.M, endpoint=False)

        # C‑grid coordinates:
        #   - interfaces (u): N+1 points from L0 to L (including both boundaries)
        self.x_interface = np.linspace(self.L0, self.L, self.N + 1)
        #   - centres (eta): N points shifted by dx/2
        self.x_center = self.L0 + (np.arange(self.N) + 0.5) * self.dx

        # For periodic problems, we only need N interfaces (the last one equals the first)
        self.x_interface_periodic = self.x_interface[:-1]

        # Alias for backward compatibility
        self.x = self.x_center

        # Warn if the CFL condition is violated for pure advection
        self._check_cfl()

    def _check_cfl(self):
        """Print a warning if the CFL number exceeds 1.0."""
        if self.cfl > 1.0:
            print(f"Warning: CFL = {self.cfl:.3f} > 1. The scheme may be unstable.")
            print("Suggestions for stable (CFL ≤ 1) combinations:")
            self.suggest_cfl()

    def suggest_cfl(self):
        """
        Display a table of (N, M) combinations that yield a CFL number ≤ 1,
        using the current domain length and time interval.
        """
        from rich.table import Table
        from rich import print

        base_N = 1024
        table = Table(title="Stable (CFL ≤ 1) (N, M) combinations")
        table.add_column("M \\ N", justify="center")
        for exp_n in range(-2, 3):
            n = base_N * (2**exp_n)
            table.add_column(f"N={n}", justify="center")

        for exp_m in range(5, 11):
            m = 2**exp_m
            row = [f"M={m}"]
            for exp_n in range(-2, 3):
                n = base_N * (2**exp_n)
                cfl = (self.T - self.t0) * n / ((self.L - self.L0) * m)
                row.append(f"{cfl:.3f}" if cfl <= 1.0 else "✗")
            table.add_row(*row)

        print(table)

    def get_coordinates(self, variable: str = "eta"):
        """
        Return the appropriate grid coordinates for a given variable.

        Parameters
        ----------
        variable : str
            'eta' or 'u' (or 'h' for the interface, as both scalar and velocity are stored there).

        Returns
        -------
        numpy.ndarray
            Coordinates of the centres (eta) or interfaces (u).
        """
        if variable.lower() in ("eta", "h"):
            return self.x_center
        elif variable.lower() == "u":
            return self.x_interface_periodic
        else:
            raise ValueError("variable must be 'eta' (or 'h') or 'u'.")

    def __repr__(self):
        return (
            f"Domain(N={self.N}, M={self.M}, dx={self.dx:.4f}, dt={self.dt:.4f}, "
            f"CFL={self.cfl:.3f}, L0={self.L0}, L={self.L}, t0={self.t0}, T={self.T})"
        )


class Function2d:
    """
    Factory class that provides all initial condition functions used in the
    research. All methods return η(x,0) for a given spatial coordinate x.
    """

    def __init__(self):
        pass

    def __str__(self) -> str:
        return ("Class designed to build initial conditions for data assimilation. "
                "All functions are consistent with the bibliography used in the thesis.")

    def paper_condition(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """
        Gaussian bump used in Kevlahan & Khan (2019) – typical test case.
        """
        return (1.0 / 20.0) * np.exp(-100.0 * x**2)

    def initial_condition_2(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """
        Narrower Gaussian bump (higher wave number).
        """
        return (1.0 / 20.0) * np.exp(-1000.0 * x**2)

    def square_condition(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """
        Square wave (top‑hat) centred at x=0 with amplitude 0.05.
        Returns 0.05 for -1 ≤ x ≤ 1, and 0 elsewhere.
        """
        if isinstance(x, (np.ndarray, list, tuple)):
            x = np.asarray(x)
            return 0.05 * ((x >= -1.0).astype(float) + (x <= 1.0).astype(float) - 1.0)
        else:
            return 0.05 * (float(x >= -1.0) + float(x <= 1.0) - 1.0)


class ForwardSolution:
    """
    Solver container for the forward (direct) problem.
    It provides the initial conditions for the prognostic variables η and u
    on the appropriate C‑grid locations.
    """

    def __init__(self,
                 domain: Optional[Domain] = None,
                 initial_condition: str = "paper_condition",
                 equation: str = "advection"):
        """
        Parameters
        ----------
        domain : Domain, optional
            Grid and time discretization object. If None, a default domain is created.
        initial_condition : str
            Name of the method in Function2d that provides η(x,0).
        equation : str
            Which equation to solve: 'advection', 'linear_SWE' or 'nonlinear_SWE'.
        """
        self.equation = equation
        self.ic_name = initial_condition

        # Create a default domain if none is given
        if domain is None:
            if equation == "advection":
                self.dom = Domain(N=1024, M=256)
            else:  # linear_SWE or nonlinear_SWE
                self.dom = Domain(N=1024, M=320)
        else:
            self.dom = domain

        # Instantiate the initial condition factory
        self.ic_factory = Function2d()

    def eta_zero(self, 
                 x: Optional[np.ndarray] = None
                 ) -> np.ndarray:
        """
        Returns η(x,0) evaluated on the cell centres (where η is stored).

        Parameters
        ----------
        x : np.ndarray, optional
            Spatial grid. If None, uses self.dom.x_center.

        Returns
        -------
        np.ndarray
            Initial elevation field.
        """
        if x is None:
            x = self.dom.x_center   # η lives at cell centres

        # Dynamically call the requested initial condition method
        method = getattr(self.ic_factory, self.ic_name, None)
        if method is None:
            raise ValueError(f"Method '{self.ic_name}' not found in Function2d.")
        return method(x)

    def u_zero(self,
                x: Optional[np.ndarray] = None
                ) -> np.ndarray:
        """
        Returns u(x,0) = 0 on the cell interfaces (where u is stored).

        Parameters
        ----------
        x : np.ndarray, optional
            Spatial grid. If None, uses self.dom.x_interface_periodic.

        Returns
        -------
        np.ndarray
            Initial velocity field (all zeros).
        """
        if x is None:
            x = self.dom.x_interface_periodic   # u lives at interfaces (periodic version)
        return np.zeros_like(x)

    def analytic_solution(self,
                           iter: Optional[int] = None
                           ) -> np.ndarray:
        """
        Analytic solution for the advection equation (c=1) or the linear shallow‑water
        equations (with initial velocity zero).

        Parameters
        ----------
        iter : int, optional
            Time step index. If None, uses the last time step (M-1).

        Returns
        -------
        np.ndarray
            Analytic solution η at the given time on the cell centres.

        Notes
        -----
        For advection:   η(x, t) = η0(x - t)
        For linear SWE:  η(x, t) = 0.5 * (η0(x - t) + η0(x + t))
        where η0 is the initial condition and t = (iter+1)*dt (as per original code).
        """
        if iter is None:
            iter = self.dom.M - 1   # last time step

        # Time elapsed
        t = (iter + 1) * self.dom.dt   # original logic: iter+1

        # Spatial grid for η
        x = self.dom.x_center   # alias for self.dom.x

        if self.equation == "advection":
            # Advection with velocity c = 1
            return self.eta_zero(x - t)

        elif self.equation == "linear_SWE":
            # Linear shallow water: sum of right‑ and left‑going waves
            return 0.5 * (self.eta_zero(x - t) + self.eta_zero(x + t))

        else:
            print("This equation does not have an analytic solution.")
            return None

    def numeric_solution(
                        self,
                        initial_eta: Optional[np.ndarray] = None,
                        initial_u: Optional[np.ndarray] = None,
                        n_steps: Optional[int] = None,
                        final_time: float = 2.0,
                        mean_depth: float = 1.0,
                    ) -> dict:
        """
        Solve the chosen equation (advection, linear SWE, or non‑linear SWE)
        numerically using finite‑volume schemes on a staggered C‑grid.

        Parameters
        ----------
        initial_eta : np.ndarray, optional
            Initial condition for η. If None, uses self.eta_zero().
        initial_u : np.ndarray, optional
            Initial condition for u. If None, uses self.u_zero() (only for SWE).
        n_steps : int, optional
            Number of time steps. If None, uses self.dom.M.
        final_time : float, optional
            Final physical time for the non‑linear SWE (when variable time stepping is used).
        mean_depth : float, optional
            Mean water depth H for the non‑linear SWE (default 1.0).

        Returns
        -------
        dict
            Dictionary containing the final fields ('eta', 'u') and, for the non‑linear case,
            also 'time' (elapsed time), 'dt_vector' (list of time steps), and 'steps'.
        """
        if n_steps is None:
            n_steps = self.dom.M

        if initial_eta is None:
            initial_eta = self.eta_zero()
        if initial_u is None and self.equation != "advection":
            initial_u = self.u_zero()

        # ------------------------------------------------------------------
        # 1. Advection equation (Lax‑Friedrichs)
        # ------------------------------------------------------------------
        if self.equation == "advection":
            cfl = self.dom.cfl

            def lax_friedrichs(eta):
                return 0.5 * ((1 + cfl) * np.roll(eta, 1) +
                              (1 - cfl) * np.roll(eta, -1))

            eta = initial_eta
            for _ in range(n_steps):
                eta = lax_friedrichs(eta)

            return {'eta': eta}

        # ------------------------------------------------------------------
        # 2. Linear shallow‑water equations (SSP RK33)
        # ------------------------------------------------------------------
        elif self.equation == "linear_SWE":
            dx = self.dom.dx
            dt = self.dom.dt

            # Correct C‑grid derivative: (v[i] - v[i-1]) / dx
            def derivative_u(v):
                return (np.roll(v, 1) - v ) / dx
            
            def derivative_eta(v):
                return (v - np.roll(v, -1)) / dx
            
            def rk_step(eta, u):
                # Stage 1
                eta1 = eta + dt * derivative_u(u)
                u1   = u   + dt * derivative_eta(eta)
                # Stage 2
                eta2 = 0.75 * eta + 0.25 * eta1 + 0.25 * dt * derivative_u(u1)
                u2   = 0.75 * u   + 0.25 * u1   + 0.25 * dt * derivative_eta(eta1)
                # Stage 3
                eta3 = (1.0 / 3.0) * eta + (2.0 / 3.0) * eta2 + (2.0 / 3.0) * dt * derivative_u(u2)
                u3   = (1.0 / 3.0) * u   + (2.0 / 3.0) * u2   + (2.0 / 3.0) * dt * derivative_eta(eta2)
                return eta3, u3

            eta, u = initial_eta, initial_u
            for _ in range(n_steps):
                eta, u = rk_step(eta, u)

            return {
                    'eta': eta,
                     'u': u
                     }

        # ------------------------------------------------------------------
        # 3. Non‑linear shallow‑water equations (SSP RK33 with variable dt)
        # ------------------------------------------------------------------
        elif self.equation == "nonlinear_SWE":
            dx = self.dom.dx
            cfl = self.dom.cfl

            # --- C‑grid interpolation helpers ---
            def u_at_centres(u):
                """Interpolate u from interfaces to cell centres."""
                return 0.5 * (np.roll(u, 1) + u)

            def eta_at_interfaces(eta):
                """Interpolate η from cell centres to interfaces."""
                return 0.5 * (np.roll(eta, -1) + eta)

            def derivative(v):
                """Correct C‑grid derivative: (v[i] - v[i-1]) / dx."""
                return (v - np.roll(v, 1)) / dx

            def rhs(eta, u):
                """Right‑hand side of the non‑linear SWE."""
                h = mean_depth + eta_at_interfaces(eta)   # total depth at interfaces
                u_c = u_at_centres(u)                     # u at centres
                # (h u)_x
                eta_t = -derivative(h * u)
                # (0.5 u² + η)_x
                u_t   = -derivative(0.5 * u_c**2 + eta)
                return eta_t, u_t

            # Initialise
            eta = initial_eta.copy()
            u   = initial_u.copy()

            dt_list = []
            elapsed = 0.0
            step = 0
            max_steps = n_steps if n_steps is not None else 10000  # safety limit

            while elapsed < final_time and step < max_steps:
                # Compute dt based on local CFL
                c_max = np.max(np.abs(u_at_centres(u)) + np.sqrt(mean_depth + eta_at_interfaces(eta)))
                dt = cfl * dx / c_max if c_max > 0 else self.dom.dt
                # Ensure we do not overshoot final_time
                if elapsed + dt > final_time:
                    dt = final_time - elapsed

                # Stage 1
                eta1, u1 = rhs(eta, u)
                eta1 = eta + dt * eta1
                u1   = u   + dt * u1
                # Stage 2
                eta2, u2 = rhs(eta1, u1)
                eta2 = 0.75 * eta + 0.25 * eta1 + 0.25 * dt * eta2
                u2   = 0.75 * u   + 0.25 * u1   + 0.25 * dt * u2
                # Stage 3
                eta3, u3 = rhs(eta2, u2)
                eta3 = (1.0 / 3.0) * eta + (2.0 / 3.0) * eta2 + (2.0 / 3.0) * dt * eta3
                u3   = (1.0 / 3.0) * u   + (2.0 / 3.0) * u2   + (2.0 / 3.0) * dt * u3

                eta, u = eta3, u3
                elapsed += dt
                dt_list.append(dt)
                step += 1

            return {
                'eta': eta,
                'u': u,
                'time': elapsed,
                'dt_vector': dt_list,
                'steps': step
            }

        else:
            raise ValueError(f"Equation '{self.equation}' is not recognised.")



# ----------------------------------------------------------------------
# Example usage (if run as main)
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import hashlib
    import json
    from pathlib import Path

    ### options

    op = 2
    iterations = 2**3

    #### Variables of the problem

    #N=1025; M = 513 #cfl = 0.5
    #N=1024; M = 320 #cfl = 0.8 # Recommended for swe
    #N=512;  M = 160 #cfl = 0.8
    N=1024; M=256   #cfl = 1 # Recommended for advection
    
    amos = 2
    noise = False
    first_sample = 0.2 # paper uses first_sample = 0.2
    Delta_x =  0.09 # paper uses Delta_x = 0.09 end Delta_x = 0.375 for counter-example


    dom = Domain(N = N, M = M)
    sol = ForwardSolution(
                    initial_condition = "paper_condition", # or "square_condition"
                    equation= "linear_SWE" # or  "advection" or "nonlinear_SWE" or "linear_SWE" 
                    )

    if op == 20:
        pass

    elif op == 2:
        sol1 = ForwardSolution(
                    initial_condition = "paper_condition", # or "square_condition"
                    equation= "linear_SWE" # or  "advection" or "nonlinear_SWE" or "linear_SWE" 
                    )

        sol2 = ForwardSolution(
            initial_condition = "paper_condition", # or "square_condition"
            equation= "nonlinear_SWE" # or  "advection" or "nonlinear_SWE" or "linear_SWE" 
            )
        x = dom.x_center 
        y = sol1.numeric_solution()['eta']
        z = sol2.numeric_solution()['eta']
        print(y.shape)
        #plt.plot(x,y, label = 'Linear SWE')
        plt.plot(x,z, label = 'Nonlinear SWE')
        plt.legend()
        plt.show()    


    elif op == 1:
        x = dom.x_center 
        y = sol.analytic_solution()
        print(y.shape)
        plt.plot(x,y)
        plt.show()



    elif op == 0:
        # Test the Domain class
        dom = Domain()
        print(dom)

        # Test the initial conditions
        ic = Function2d()
        x = np.linspace(-1.5, 1.5, 100)
        y1 = ic.paper_condition(x)
        y2 = ic.square_condition(x)
        print("Paper condition at x=0:", ic.paper_condition(0.0))
        print("Square condition at x=0:", ic.square_condition(0.0))

        # Test the ForwardSolution
        fwd = ForwardSolution(equation="advection", initial_condition="square_condition")
        eta0 = fwd.eta_zero()
        u0 = fwd.u_zero()
        print(f"Number of eta points: {len(eta0)}")
        print(f"Number of u points: {len(u0)}")
        print(f"First 5 eta values: {eta0[:5]}")
        print(f"First 5 u values: {u0[:5]}")
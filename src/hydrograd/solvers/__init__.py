"""Time integration and nonlinear solvers of hydrograd.

Modules:
    ode: explicit time integration with diffrax (Tsit5, SSPRK3, ...).
    implicit: implicit theta-scheme (backward Euler, Crank-Nicolson) with IFT adjoints.
    steady: steady states by pseudo-transient Newton with IFT gradients.
    jacobian: exact sparse Jacobian dR/dU by graph coloring and JVPs.
    final: one entry point for the final (or steady) state of a run.
"""

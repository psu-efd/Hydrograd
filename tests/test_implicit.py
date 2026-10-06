"""Implicit theta-scheme stepper: accuracy vs explicit, mass balance, IFT gradient, wetting.

All cases use the hydrograph-inversion valley of ``benchmarks/validation/hydrograph_inversion.py``:
a sloping main channel with gentle floodplains, an inlet-Q hydrograph rising from about
40 to about 180 m^3/s, and an exit-H (or wall) outlet.

What is checked:
- Backward Euler with dt = 20-30 s agrees with the explicit SSPRK3 reference (dt = 0.1 s, CFL
  limited) to about 1-2 cm in water depth / water-surface elevation, keeps h >= 0, and wets
  the floodplain as the discharge rises.
- In a closed channel (wall outlet) the change in stored volume equals the inflow volume to a
  relative error of 1e-5. The inlet face carries exactly Q(t^{n+1}) per step, so the inflow is
  the right-end-rule quadrature of the hydrograph.
- The reverse-mode gradient through the per-step implicit-function-theorem adjoint matches a
  central finite difference to a relative error of 2e-3.
- ``ode_solver = "BackwardEuler()"`` can be selected from ``run_control.json`` through the
  forward app, writes its output frames, and agrees with the explicit run to 2 cm in WSE.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import hydrograd  # noqa: F401
from hydrograd.apps.setup import setup_case
from hydrograd.io.control import read_control
from hydrograd.io.synthetic import write_channel_case
from hydrograd.solvers.implicit import ImplicitSettings, solve_implicit
from hydrograd.solvers.ode import solve_swe


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "validation"))
import hydrograph_inversion as H  # noqa: E402


def _initial_wse(x, y):
    """Initial water-surface elevation: a plane parallel to the bed slope.

    Args:
        x: cell-centre x coordinates (m), shape [n_cells].
        y: cell-centre y coordinates (m), shape [n_cells] (unused).

    Returns:
        WSE (m), shape [n_cells]: S0 (L - x) + 1.03, i.e. 1.03 m at the outlet.
    """
    return H.S0 * (H.L - x) + 1.03


def _case(tmp_path, outlet="EXIT-H", t_end=1200.0):
    """Hydrograph-inversion valley (gentle floodplains), Q rising 40 -> ~180 m^3/s over t_end.

    Args:
        tmp_path: folder where the synthetic SRH-2D case is written.
        outlet: outlet boundary type ("EXIT-H" or "WALL").
        t_end: end time of the simulation and of the hydrograph (s).

    Returns:
        The set-up case (model, initial state, ODE settings).
    """
    # Five hydrograph knots, evenly spaced over [0, t_end]
    hydrograph_times = np.linspace(0.0, t_end, 5)

    # Write the case files (explicit SSPRK3 reference settings) and set it up
    write_channel_case(
        tmp_path,
        H.NX,
        H.NY,
        length=H.L,
        width=H.W,
        n_manning=H.N_MAN,
        zb_fn=H.bed,
        wstill=0.0,
        inlet="INLET-Q",
        outlet=outlet,
        dt=0.1,
        solver="SSPRK3()",
        t_end=t_end,
        wse_out=1.03,
        Q_series=(hydrograph_times, H.q_true(hydrograph_times)),
        ic_wse=_initial_wse,
    )
    return setup_case(read_control(tmp_path / "run_control.json"))


def test_implicit_matches_explicit_and_wets_floodplain(tmp_path) -> None:
    """Backward Euler (dt = 30 s) vs SSPRK3 after 1200 s of a rising hydrograph.

    Checks h >= 0, more cells wetter than 1 cm than initially (the floodplain wets), and an RMS
    depth difference below 1 cm over the cells that are wet in the explicit run.
    """
    case = _case(tmp_path)
    model = case.model

    # Explicit reference run, final state only
    explicit_settings = dataclasses.replace(case.ode, n_save=1)
    U_explicit = np.asarray(
        jax.jit(lambda U0: solve_swe(model, {}, U0, explicit_settings).ys[-1])(case.U0)
    )

    # Implicit run: t in [0, 1200] s, dt = 30 s, Newton tolerance 1e-8
    implicit_settings = ImplicitSettings(0.0, 1200.0, 30.0, n_save=1, tol=1e-8)
    U_implicit = np.asarray(
        jax.jit(lambda U0: solve_implicit(model, {}, U0, implicit_settings)[1][-1])(case.U0)
    )

    # Water depths h = xi + h_still (m)
    h_initial = np.asarray(case.U0[0] + model.hstill)
    h_explicit = U_explicit[0] + np.asarray(model.hstill)
    h_implicit = U_implicit[0] + np.asarray(model.hstill)

    # Positivity and floodplain wetting
    assert h_implicit.min() >= 0.0
    assert (h_implicit > 0.01).sum() > (h_initial > 0.01).sum()           # floodplain wets

    # RMS depth difference over the explicitly wet cells
    wet = h_explicit > 0.01
    rms_difference = np.sqrt(np.mean((h_implicit - h_explicit)[wet] ** 2))
    assert rms_difference < 0.01                                          # within 1 cm (dt = 30 s)


def test_implicit_volume_balance_closed_channel(tmp_path) -> None:
    """Closed channel: storage change equals the hydrograph inflow volume to 1e-5 (relative)."""
    case = _case(tmp_path, outlet="WALL", t_end=600.0)
    model = case.model

    # Implicit run: t in [0, 600] s, dt = 20 s, tight Newton tolerance
    implicit_settings = ImplicitSettings(0.0, 600.0, 20.0, n_save=1, tol=1e-10)
    U_final = np.asarray(
        jax.jit(lambda U0: solve_implicit(model, {}, U0, implicit_settings)[1][-1])(case.U0)
    )

    # Stored volume (m^3) at the start and the end
    area = np.asarray(model.area)
    volume_initial = np.sum(np.asarray(case.U0[0] + model.hstill) * area)
    volume_final = np.sum((U_final[0] + np.asarray(model.hstill)) * area)

    # Interior fluxes cancel and the inlet face carries exactly Q(t^{n+1}) per step, so the
    # storage change equals the right-end-rule quadrature of the hydrograph.
    step_end_times = np.arange(20.0, 600.0 + 1e-9, 20.0)
    knots = np.linspace(0.0, 600.0, 5)
    inflow = 20.0 * np.sum(np.interp(step_end_times, knots, H.q_true(knots)))
    assert abs((volume_final - volume_initial) - inflow) < 1e-5 * inflow


def test_implicit_ift_gradient_matches_finite_differences(tmp_path) -> None:
    """d(sum w h)/dQ_knot via the implicit IFT adjoint vs a central FD (step 0.05 m^3/s).

    The loss is a random weighted sum of the final depths; the relative tolerance is 2e-3.
    """
    case = _case(tmp_path, t_end=600.0)
    model = case.model
    implicit_settings = ImplicitSettings(0.0, 600.0, 60.0, n_save=1, tol=1e-10)

    # Random weights for the scalar loss
    weights = jnp.asarray(np.random.default_rng(0).normal(size=model.n_cells))

    def loss(inlet_q):
        """Weighted sum of the final water depths (m) for the inlet hydrograph values inlet_q."""
        U_final = solve_implicit(model, {"Q": inlet_q}, case.U0, implicit_settings)[1][-1]
        return jnp.sum(weights * (U_final[0] + model.hstill))

    # Reverse-mode gradient through the implicit adjoint
    q0 = jnp.asarray(model.Q)
    gradient = np.asarray(jax.jit(jax.grad(loss))(q0))

    # Central finite difference on hydrograph entry [0, 1], step 0.05
    loss_jit = jax.jit(loss)
    perturbation = jnp.zeros_like(q0).at[0, 1].set(0.05)
    fd = (float(loss_jit(q0 + perturbation)) - float(loss_jit(q0 - perturbation))) / 0.1
    assert abs(gradient[0, 1] - fd) <= 2e-3 * abs(fd)


def test_backward_euler_selectable_from_run_control(tmp_path) -> None:
    """ode_solver = "BackwardEuler()" through the forward app (frames, outputs) vs explicit."""
    from hydrograd.apps.forward import run_forward

    hydrograph_times = np.linspace(0.0, 600.0, 5)
    wse_by_solver = {}

    # Run the same case with the explicit and the implicit solver
    for solver, dt in (("SSPRK3()", 0.1), ("BackwardEuler()", 20.0)):
        case_dir = tmp_path / solver.strip("()")
        write_channel_case(
            case_dir,
            H.NX,
            H.NY,
            length=H.L,
            width=H.W,
            n_manning=H.N_MAN,
            zb_fn=H.bed,
            wstill=0.0,
            inlet="INLET-Q",
            outlet="EXIT-H",
            dt=dt,
            solver=solver,
            t_end=600.0,
            n_save=2,
            wse_out=1.03,
            Q_series=(hydrograph_times, H.q_true(hydrograph_times)),
            ic_wse=_initial_wse,
        )
        result = run_forward(setup_case(read_control(case_dir / "run_control.json")))

        # Three saved frames (t = 0 and n_save = 2) and the last VTK file written
        assert len(result["ts"]) == 3 and (case_dir / "forward_simulation_results_0003.vtk").exists()
        wse_by_solver[solver] = result["fields"]["wse"]

    # WSE agreement within 2 cm everywhere
    assert np.abs(wse_by_solver["BackwardEuler()"] - wse_by_solver["SSPRK3()"]).max() < 0.02

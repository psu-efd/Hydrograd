"""Tests of the 1-D applications: parameters, case file, forward, sensitivity, inversion.

Parameters (``hydrograd.params.transforms``):

1. Every transform is inverted by its inverse; the bounds of the bounded transforms hold;
   softmax fractions sum to one.
2. ``ParameterMap``: whole parameters and single entries of vector parameters; inactive
   parameters and entries keep their reference values; the Gaussian prior.

Case file and run control (``hydrograd.io.oned_case``, ``hydrograd.io.control``):

3. A case written and read back is the same case; misspelled names and wrong shapes are
   reported.

Applications (``hydrograd.apps.oned``), run through the command-line entry point:

4. Forward: the results and truth files are written; the truth volumes are the surveyed
   volumes of the run; noise is added only to the truth file.
5. Sensitivity: the forward-mode Jacobian of the surveyed volumes matches central
   differences.
6. Inversion (twin experiment): the settling multiplier of the fine class is recovered
   from the surveyed deposit volumes of a run with a known value.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from hydrograd.apps.oned.forward import simulate
from hydrograd.apps.oned.observe import load_truth, survey_volumes
from hydrograd.cli import run as run_control_file
from hydrograd.io.control import read_control
from hydrograd.io.oned_case import OneDCase, read_case, write_case
from hydrograd.oned import (
    MorphNumerics,
    OneDConfig,
    UnsteadyNumerics,
    constant_forcing,
    default_parameters,
    network_from_positions,
)
from hydrograd.oned.config import default_parameter_specs
from hydrograd.oned.synthetic import trapezoid_channel, uniform_bed
from hydrograd.params.transforms import (
    ParameterMap,
    ParamSpec,
    to_constrained,
    to_unconstrained,
    with_values,
)
from hydrograd.sediment import SedimentConfig

DAY = 86400.0           # s


def _twin_case(kappa_fine: float = 1.6) -> OneDCase:
    """A small reservoir case: 10 km channel, 20 m pool, silt and sand, 8 steps of 5 days.

    The pool is deep and the discharge moderate, so the silt settles over most of the
    pool and its deposit depends on its settling velocity.

    Args:
        kappa_fine: settling-velocity multiplier of the fine class (-).

    Returns:
        The case.
    """
    n_steps = 8
    config = OneDConfig(sediment=SedimentConfig(diameters=(2.0e-5, 5.0e-4)))
    x, z, s = trapezoid_channel(length=10000.0, n_sections=11, slope=5.0e-4)
    theta = default_parameters(2)
    theta["tau_cd"] = jnp.asarray(1.0)
    theta["kappa_w"] = jnp.asarray([kappa_fine, 1.0])
    load_in = jnp.broadcast_to(jnp.asarray([0.05, 0.002])[:, None], (2, n_steps))
    forcing = constant_forcing(100.0, 120.0, 5.0 * DAY, n_steps)._replace(load_in=load_in)

    return OneDCase(
        config=config,
        network=network_from_positions(x),
        theta=theta,
        forcing=forcing,
        bed0=uniform_bed(z, s, n_classes=2),
    )


def _write_control(case_dir: Path, mode_flag: str, options: dict) -> Path:
    """Write a ``run_control.json`` for the 1-D case ``case.json`` in ``case_dir``.

    Args:
        case_dir: case folder.
        mode_flag: the ``bPerform_*`` flag to set.
        options: option blocks by name, e.g. ``{"inversion_options": {...}}``.

    Returns:
        Path of the control file.
    """
    control = {
        "control_variables": {"model": "1D", "case_file_name": "case.json", mode_flag: True},
    }
    control.update(options)
    path = case_dir / "run_control.json"
    path.write_text(json.dumps(control))
    return path


def test_transforms_round_trip():
    """v -> u -> v for every transform, and the bounds of the bounded ones."""
    cases = [
        (ParamSpec("a", "log"), jnp.asarray([1.0e-8, 0.05, 30.0])),
        (ParamSpec("b", "affine", lo=2.0, hi=3.0), jnp.asarray([2.1, 2.5, 2.9])),
        (ParamSpec("c", "logaffine", lo=1.0e-3, hi=10.0), jnp.asarray([0.01, 1.0, 5.0])),
        (ParamSpec("d", "softmax"), jnp.asarray([0.6, 0.3, 0.1])),
        (ParamSpec("e", "identity"), jnp.asarray([-2.0, 0.0, 3.0])),
    ]
    for spec, value in cases:
        u = to_unconstrained(spec, value)
        np.testing.assert_allclose(to_constrained(spec, u), value, rtol=1e-12, err_msg=spec.name)

    # Any u gives a value inside the bounds; softmax fractions sum to one.
    u = jnp.asarray([-50.0, 0.0, 50.0])
    bounded = to_constrained(ParamSpec("b", "affine", lo=2.0, hi=3.0), u)
    assert bool(jnp.all((bounded >= 2.0) & (bounded <= 3.0)))
    np.testing.assert_allclose(jnp.sum(to_constrained(ParamSpec("d", "softmax"), u)), 1.0)

    with pytest.raises(ValueError, match="Unknown transform"):
        to_constrained(ParamSpec("x", "cubic"), u)


def test_parameter_map():
    """Whole parameters, single entries, fixed values and the prior."""
    theta = default_parameters(3)
    specs = default_parameter_specs()
    specs["a_eh"] = dataclasses.replace(specs["a_eh"], prior_sd=0.5)
    parameter_map = ParameterMap(specs, theta, ["a_eh", "kappa_w[1]", "p_in"])

    # At the reference: the same theta, and a zero log-prior.
    u = parameter_map.unconstrained()
    assert set(u) == {"a_eh", "kappa_w[1]", "p_in"}
    assert u["kappa_w[1]"].shape == ()
    for name, value in parameter_map.theta(u).items():
        np.testing.assert_allclose(value, theta[name], rtol=1e-13, err_msg=name)
    np.testing.assert_allclose(parameter_map.log_prior(u), 0.0, atol=1e-30)

    # Move the variables: a_eh doubles, kappa_w[1] triples; the other entries stay.
    moved = dict(u)
    moved["a_eh"] = u["a_eh"] + jnp.log(2.0)
    moved["kappa_w[1]"] = u["kappa_w[1]"] + jnp.log(3.0)
    theta_moved = parameter_map.theta(moved)
    np.testing.assert_allclose(theta_moved["a_eh"], 0.1, rtol=1e-13)
    np.testing.assert_allclose(theta_moved["kappa_w"], [1.0, 3.0, 1.0], rtol=1e-13)
    np.testing.assert_allclose(theta_moved["tau_cd"], theta["tau_cd"], rtol=1e-15)
    np.testing.assert_allclose(parameter_map.physical(moved)["kappa_w[1]"], 3.0, rtol=1e-13)

    # Prior of a_eh only (sd 0.5 in ln a_eh): -1/2 (ln 2 / 0.5)^2.
    np.testing.assert_allclose(
        parameter_map.log_prior(moved), -0.5 * (np.log(2.0) / 0.5) ** 2, rtol=1e-12,
    )

    # Starting values by label.
    start = with_values(theta, {"kappa_w[0]": 0.7, "gamma1": 1200.0})
    np.testing.assert_allclose(start["kappa_w"], [0.7, 1.0, 1.0])
    np.testing.assert_allclose(start["gamma1"], [1200.0, 1200.0, 1200.0])
    np.testing.assert_allclose(theta["kappa_w"], [1.0, 1.0, 1.0])

    # Errors: unknown parameter, index out of range, malformed label.
    with pytest.raises(ValueError, match="No specification"):
        ParameterMap(specs, theta, ["not_a_parameter"])
    with pytest.raises(ValueError, match="only 3 entries"):
        ParameterMap(specs, theta, ["kappa_w[5]"])
    with pytest.raises(ValueError, match="Invalid parameter label"):
        ParameterMap(specs, theta, ["kappa_w[first]"])


def test_case_file_round_trip(tmp_path: Path):
    """Write, read back, and compare; errors for misspelled names and wrong shapes."""
    case = _twin_case()
    morph = MorphNumerics(hydro_every=2, snapshot_steps=(3, 7), checkpoint=4)
    hydraulics = dataclasses.replace(
        case.config.hydraulics,
        conveyance="section",
        flow_solver="unsteady",
        resistance="keulegan",
        roughness_scale="surface_mean",
        transport_stress="reduced",
        unsteady=UnsteadyNumerics(n_substeps=50, headwater="critical"),
    )
    case = dataclasses.replace(
        case,
        config=dataclasses.replace(case.config, morph=morph, hydraulics=hydraulics),
        theta={**case.theta, "beta_momentum": jnp.asarray(1.1), "roughness_factor": jnp.asarray(2.5)},
    )
    path = tmp_path / "case.json"
    write_case(path, case)
    loaded = read_case(path)

    # Settings (frozen dataclasses compare by value).
    assert loaded.config == case.config

    # Network, bed, forcing and parameters.
    np.testing.assert_array_equal(loaded.network.downstream, case.network.downstream)
    np.testing.assert_allclose(loaded.network.reach_length, case.network.reach_length)
    np.testing.assert_allclose(loaded.bed0.z, case.bed0.z, rtol=1e-15)
    np.testing.assert_allclose(loaded.bed0.p_active, case.bed0.p_active, rtol=1e-15)
    assert loaded.bed0.z_floor is None
    np.testing.assert_allclose(loaded.forcing.load_in, case.forcing.load_in, rtol=1e-15)
    for name, value in case.theta.items():
        np.testing.assert_allclose(loaded.theta[name], value, rtol=1e-15, err_msg=name)

    # A misspelled parameter name is an error.
    data = json.loads(path.read_text())
    data["parameters"]["tau_dc"] = 0.2
    bad = tmp_path / "bad_name.json"
    bad.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="unknown parameter"):
        read_case(bad)

    # A misspelled setting is an error.
    data = json.loads(path.read_text())
    data["numerics"]["morph"]["hydro_evry"] = 3
    bad = tmp_path / "bad_setting.json"
    bad.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="unknown setting"):
        read_case(bad)

    # A forcing series of the wrong length is an error.
    data = json.loads(path.read_text())
    data["forcing"]["dt"] = data["forcing"]["dt"][:-1]
    bad = tmp_path / "bad_shape.json"
    bad.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=r"forcing\.dt has shape"):
        read_case(bad)

    # An unknown conveyance formula and a misspelled hydraulics setting are errors.
    data = json.loads(path.read_text())
    data["hydraulics"]["conveyance"] = "sections"
    bad = tmp_path / "bad_conveyance.json"
    bad.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="hydraulics.conveyance must be one of"):
        read_case(bad)
    data["hydraulics"] = {"conveyence": "section"}
    bad.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="unknown setting"):
        read_case(bad)

    # An unknown flow solver is an error.
    data = json.loads(path.read_text())
    data["hydraulics"]["flow_solver"] = "dynamic"
    bad.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="hydraulics.flow_solver must be one of"):
        read_case(bad)

    # Stations that decrease along a section are an error.
    data = json.loads(path.read_text())
    data["sections"]["s"][0][1] = data["sections"]["s"][0][0] - 1.0
    bad = tmp_path / "bad_stations.json"
    bad.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="non-decreasing"):
        read_case(bad)

    # Parameters, numerics and hydraulics may be left out: defaults apply.
    data = json.loads(path.read_text())
    del data["parameters"]
    del data["numerics"]
    del data["hydraulics"]
    minimal = tmp_path / "minimal.json"
    minimal.write_text(json.dumps(data))
    defaults = read_case(minimal)
    assert defaults.config.morph == MorphNumerics()
    assert defaults.config.hydraulics.conveyance == "segments"
    assert defaults.config.hydraulics.flow_solver == "backwater"
    assert "beta_momentum" not in defaults.theta
    np.testing.assert_allclose(defaults.theta["tau_cd"], 0.1)


def test_forward_application(tmp_path: Path):
    """``hydrograd run`` in forward mode writes the results and the truth file."""
    case = _twin_case()
    write_case(tmp_path / "case.json", case)
    final_time = float(case.forcing.time_end()[-1])
    options = {
        "forward_simulation_options": {
            "survey_times": [final_time / 2.0, final_time],
            "survey_consolidation": False,
        }
    }
    control_path = _write_control(tmp_path, "bPerform_Forward_Simulation", options)

    # The control file selects the 1-D model.
    control = read_control(control_path)
    assert control.model == "1D"
    assert control.case_file == "case.json"

    run_control_file(control_path)

    # Truth file: the surveyed volumes of the run, at both survey times.
    truth = load_truth(tmp_path / "oned_survey_truth.json")
    result = simulate(case)
    expected = survey_volumes(result, case.theta, case.config.sediment, truth.times, False)
    assert truth.volumes.shape == (2, 11)
    np.testing.assert_allclose(truth.volumes, expected, rtol=1e-12)

    # The survey at half time holds the deposits of the first four steps only.
    half = jnp.sum(result.deposition[:4], axis=0)
    solid_fraction = case.theta["gamma1"] / 2650.0
    np.testing.assert_allclose(
        truth.volumes[0], jnp.sum(half / solid_fraction[:, None], axis=0), rtol=1e-12,
    )

    # Results files.
    summary = json.loads((tmp_path / "oned_forward_results.json").read_text())
    np.testing.assert_allclose(summary["deposit_volume_final"], truth.volumes[1], rtol=1e-12)
    assert len(summary["thalweg_final"]) == 11
    arrays = np.load(tmp_path / "oned_forward_results.npz")
    np.testing.assert_allclose(arrays["deposition"], result.deposition, rtol=1e-14)

    # With noise, the truth file changes by about the requested relative amount.
    options["forward_simulation_options"]["survey_noise_relative"] = 0.05
    _write_control(tmp_path, "bPerform_Forward_Simulation", options)
    run_control_file(control_path)
    noisy = load_truth(tmp_path / "oned_survey_truth.json")
    rms = np.sqrt(np.mean(truth.volumes**2, axis=1, keepdims=True))
    relative_noise = np.std((noisy.volumes - truth.volumes) / rms)
    assert 0.02 < relative_noise < 0.1


def test_sensitivity_application(tmp_path: Path):
    """The Jacobian of the surveyed volumes matches central differences."""
    case = _twin_case()
    write_case(tmp_path / "case.json", case)
    options = {
        "sensitivity_analysis_options": {
            "active_param_names": ["kappa_w", "alpha_d", "n_manning"],
            "sensitivity_parameter_values": {"alpha_d": 0.3},
            "survey_consolidation": False,
        }
    }
    control_path = _write_control(tmp_path, "bPerform_Sensitivity_Analysis", options)
    run_control_file(control_path)

    data = json.loads((tmp_path / "sensitivity_results.json").read_text())
    labels = data["parameter_labels"]
    values = np.asarray(data["parameter_values"])
    jacobian = np.asarray(data["jacobian"])

    # The vector parameter is expanded; the evaluation point is the overridden value.
    assert labels == ["kappa_w[0]", "kappa_w[1]", "alpha_d", "n_manning[0]"]
    np.testing.assert_allclose(values, [1.6, 1.0, 0.3, 0.03])
    assert jacobian.shape == (1, 11, 4)
    np.testing.assert_allclose(
        np.asarray(data["jacobian_log"]), jacobian * values[None, None, :], rtol=1e-13,
    )

    # Central differences in each active scalar.
    theta = with_values(case.theta, {"alpha_d": 0.3})
    times = np.asarray(data["survey_times"])
    scale = float(np.abs(jacobian).max())
    for column, label in enumerate(labels):
        step = 1.0e-5 * values[column]
        volumes = []
        for sign in (1.0, -1.0):
            perturbed = with_values(theta, {label: values[column] + sign * step})
            result = simulate(case, perturbed)
            volumes.append(
                np.asarray(survey_volumes(result, perturbed, case.config.sediment, times, False))
            )
        difference = (volumes[0] - volumes[1]) / (2.0 * step)
        column_scale = float(np.abs(difference).max())
        np.testing.assert_allclose(
            jacobian[:, :, column], difference,
            rtol=1e-4, atol=1e-5 * column_scale + 1e-12 * scale, err_msg=label,
        )


# Optimizer stages of the twin experiment. Adam compiles in seconds; L-BFGS needs fewer
# iterations, but compiling its line search around the whole run takes about two minutes.
ADAM_STAGES = {
    "inversion_optimizers": ["Adam", "Adam"],
    "inversion_learning_rates": [0.1, 0.01],
    "inversion_max_iterations": [80, 80],
    "inversion_abs_tols": [0.0, 0.0],
}
LBFGS_STAGES = {
    "inversion_optimizers": ["LBFGS"],
    "inversion_learning_rates": [1.0],
    "inversion_max_iterations": [25],
    "inversion_abs_tols": [1.0e-6],
}


@pytest.mark.parametrize(
    "stages",
    [
        pytest.param(ADAM_STAGES, id="adam"),
        pytest.param(LBFGS_STAGES, id="lbfgs", marks=pytest.mark.slow),
    ],
)
def test_inversion_application_recovers_settling_multiplier(tmp_path: Path, stages: dict):
    """Twin experiment: recover kappa_w of the fine class (truth 1.6, start 0.7)."""
    # Truth run with kappa_w[0] = 1.6 writes the surveyed volumes.
    truth_case = _twin_case(kappa_fine=1.6)
    write_case(tmp_path / "case.json", truth_case)
    forward = {"forward_simulation_options": {"survey_consolidation": False}}
    control_path = _write_control(tmp_path, "bPerform_Forward_Simulation", forward)
    run_control_file(control_path)

    # The case file of the inversion does not know the truth: kappa_w[0] = 1.
    write_case(tmp_path / "case.json", _twin_case(kappa_fine=1.0))
    inversion = {
        "inversion_options": {
            "active_param_names": ["kappa_w[0]"],
            "inversion_parameter_initial_values": {"kappa_w[0]": 0.7},
            "inversion_truth_file_name": "oned_survey_truth.json",
            "inversion_survey_relative_sigma": 0.05,
            "inversion_save_frequency": 5,
            **stages,
        }
    }
    _write_control(tmp_path, "bPerform_Inversion", inversion)
    run_control_file(control_path)

    # The multiplier is recovered and the misfit vanishes.
    results = json.loads((tmp_path / "inversion_results.json").read_text())
    assert results["parameter_labels"] == ["kappa_w[0]"]
    np.testing.assert_allclose(results["parameter_initial_values"], [0.7])
    np.testing.assert_allclose(results["parameter_final_values"], [1.6], atol=0.02)
    assert results["loss_misfit"] < 1.0e-3

    # Loss history: the first row is the start, and the loss decreased.
    lines = (tmp_path / "inversion_loss_history.csv").read_text().strip().split("\n")
    assert lines[0] == "iter,loss_total,loss_misfit,loss_prior,kappa_w[0]"
    first = [float(x) for x in lines[1].split(",")]
    last = [float(x) for x in lines[-1].split(",")]
    np.testing.assert_allclose(first[4], 0.7, rtol=1e-12)
    assert last[1] < 1.0e-3 * first[1]
    assert (tmp_path / "inversion_callback_save_iter_1.json").is_file()

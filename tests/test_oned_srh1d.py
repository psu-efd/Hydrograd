"""Tests of the 1-D mode on the cases of the comparison with SRH-1D (``benchmarks/oned``).

1. The reader of SRH-1D bed profiles (``hydrograd.io.srh1d``) returns the sections,
   output times and initial thalweg of the reservoir case; the last output time, which
   SRH-1D writes twice, is kept once; lengths and volumes are converted from feet.
2. The initial bed of each hydrograd case file is the initial bed of its SRH-1D run.
3. Reservoir: over two years the thalweg differs from that of SRH-1D by less than 0.5 m
   RMS for a bed rise of about 7 m (less than 0.2 m without the dam section, where
   SRH-1D spreads the deposit over half the length), and the largest rise and the bulk
   volume of the deposits agree within 5 % (both formulas of the conveyance).
4. Confluence: after five years the thalweg differs from that of SRH-1D by less than
   0.15 m RMS (0.08 m without the junction and the dam section), and the deposit volumes
   agree within 1 %.
5. Stable adjoint: the relative sensitivity of the deposit volume of the reservoir to the
   coefficient of the sediment rating curve, d ln V / d ln a_r, stays close to one when
   the run is four times as long.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hydrograd.io.oned_case import read_case
from hydrograd.io.srh1d import read_profile, section_indices
from hydrograd.oned import run
from hydrograd.sediment import consolidated_volume

BENCHMARK = Path(__file__).resolve().parents[1] / "benchmarks" / "oned"
CASES = BENCHMARK / "cases"


def _load_comparison():
    """Import ``benchmarks/oned/compare_srh1d.py`` as a module.

    Returns:
        The module.
    """
    spec = importlib.util.spec_from_file_location("compare_srh1d", BENCHMARK / "compare_srh1d.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_profile_reader(tmp_path: Path):
    """Sections, times and values of the reservoir profile; unit conversion."""
    path = CASES / "reservoir" / "srh1d" / "reservoir_OUT_Profile.DAT"
    profile = read_profile(path)

    # One river with 41 cross-sections, 500 m apart, numbered from upstream.
    assert profile.river.tolist() == [1] * 41
    assert profile.section.tolist() == list(range(1, 42))
    np.testing.assert_allclose(profile.location, np.linspace(20000.0, 0.0, 41))

    # Monthly output over two years; the last time is in the file twice and kept once.
    assert profile.time[0] == 0.0
    assert np.all(np.diff(profile.time) > 0.0)
    np.testing.assert_allclose(profile.time[-1], 2.0 * 8766.0)
    assert profile.thalweg.shape == (profile.time.shape[0], 41)

    # Initial bed: slope 0.001 down to 100 m at the dam; nothing deposited yet.
    np.testing.assert_allclose(profile.thalweg[0], np.linspace(120.0, 100.0, 41), atol=1e-6)
    np.testing.assert_allclose(profile.deposit_volume[0], 0.0)
    assert np.all(profile.water_surface[0] > profile.thalweg[0])

    # Sections picked by (river, section), in a given order.
    index = section_indices(profile, [(1, 41), (1, 1)])
    assert index.tolist() == [40, 0]
    with pytest.raises(KeyError, match="no cross-section"):
        section_indices(profile, [(2, 1)])

    # A file in feet: lengths times 0.3048, volumes times 0.3048^3.
    in_feet = read_profile(path, length_unit=0.3048)
    np.testing.assert_allclose(in_feet.thalweg, profile.thalweg * 0.3048)
    np.testing.assert_allclose(in_feet.deposit_volume, profile.deposit_volume * 0.3048**3)

    # A file without records is an error.
    empty = tmp_path / "empty_OUT_Profile.DAT"
    empty.write_text("# TITLE\n t,riv,xcid,xc\n")
    with pytest.raises(ValueError, match="no profile records"):
        read_profile(empty)


@pytest.mark.parametrize("name", ["reservoir", "confluence"])
def test_case_starts_from_the_bed_of_srh1d(name: str):
    """The thalweg of the case file is the thalweg of SRH-1D at its first output."""
    folder = CASES / name
    case = read_case(folder / "case.json")
    meta = json.loads((folder / "benchmark.json").read_text())

    profile = read_profile(folder / meta["srh1d_profile"], meta["srh1d_length_unit_m"])
    index = section_indices(profile, [tuple(pair) for pair in meta["srh1d_sections"]])

    np.testing.assert_allclose(
        np.asarray(case.bed0.z).min(axis=1), profile.thalweg[0][index], atol=1e-4,
    )


@pytest.mark.parametrize("conveyance", ["segments", "section"])
def test_reservoir_agrees_with_srh1d(conveyance: str):
    """Thalweg, largest rise and deposit volume of the reservoir after two years."""
    out = _load_comparison().compare_case(CASES / "reservoir", conveyance)

    assert out["rms_thalweg_max_m"] < 0.5
    assert out["rms_thalweg_without_river_ends_max_m"] < 0.2
    assert out["river_end_sections"] == [40]
    rise = out["largest_rise_m"]
    assert 6.0 < rise["srh1d"] < 7.5
    np.testing.assert_allclose(rise["hydrograd"], rise["srh1d"], rtol=0.05)
    volume = out["bulk_deposit_m3"]
    np.testing.assert_allclose(volume["hydrograd"], volume["srh1d"], rtol=0.05)


def test_confluence_agrees_with_srh1d():
    """Thalweg and deposit volume of the confluence after five years."""
    out = _load_comparison().compare_case(CASES / "confluence")

    assert out["rms_thalweg_final_m"] < 0.15
    assert out["rms_thalweg_without_river_ends_final_m"] < 0.08
    assert out["river_end_sections"] == [30, 40]
    volume = out["bulk_deposit_m3"]
    np.testing.assert_allclose(volume["hydrograd"], volume["srh1d"], rtol=0.01)


def test_sensitivity_to_the_load_does_not_grow_with_the_run_length():
    """d ln V / d ln a_r of the reservoir stays near one from 50 to 200 daily steps."""
    case = read_case(CASES / "reservoir" / "case.json")

    def relative_sensitivity(n_steps: int) -> float:
        """d ln V / d ln a_r after ``n_steps`` steps (-).

        V is the consolidated bulk volume of all deposits (m^3) at the end of the run and
        a_r the coefficient of the rating curve of the inflow load.
        """
        forcing = case.forcing._replace(
            Q_in=case.forcing.Q_in[:n_steps],
            eta_outlet=case.forcing.eta_outlet[:n_steps],
            dt=case.forcing.dt[:n_steps],
        )
        time_survey = float(forcing.time_end()[-1])

        def total_volume(a_r):
            """Consolidated bulk volume of all deposits (m^3) for the coefficient a_r."""
            theta = dict(case.theta)
            theta["a_r"] = a_r
            result = run(case.config, case.network, theta, forcing, case.bed0)
            volume = consolidated_volume(
                result.deposition, result.time_end, time_survey, theta, case.config.sediment,
            )
            return jnp.sum(volume)

        value, derivative = jax.jit(jax.value_and_grad(total_volume))(case.theta["a_r"])
        return float(case.theta["a_r"] * derivative / value)

    short = relative_sensitivity(50)
    long = relative_sensitivity(200)

    # The deposit is nearly proportional to the load, and stays so in the longer run.
    assert 0.8 < short < 1.2
    assert 0.8 < long < 1.2

"""Tests of scattered velocity data (``io.lspiv``, ``apps.velocity_data``).

1. The reader takes PIVlab-style ASCII files (title lines, units in the column names,
   masked NaN vectors) and CSV files with other column names.
2. The frame transform (flip, rotation, scale) maps positions and velocities consistently.
3. Every point is located in the mesh cell that contains it (checked against the
   rectangular cells of a channel mesh); points outside the mesh get -1.
4. Cell averaging: means, point counts, the uncertainty sqrt(sigma^2 / N + floor^2), the
   surface factor, and the minimum point count.
5. The velocity loss is zero for data made from the model's own steady solution and
   grows when Manning's n is changed; its gradient through the steady solve (IFT adjoint)
   matches central differences. This runs the ``"velocity_data"`` run-control block.
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np

from hydrograd.apps.setup import setup_case
from hydrograd.apps.velocity_data import load_velocity_data, velocity_loss
from hydrograd.io.control import read_control
from hydrograd.io.lspiv import (
    PointVelocities,
    bin_to_cells,
    locate_cells,
    read_velocity_points,
    transform_points,
)
from hydrograd.io.synthetic import write_channel_case
from hydrograd.solvers.final import final_state

# Channel of the tests (SI units): 40 x 4 rectangular cells of 5 m x 2.5 m.
LENGTH = 200.0          # m
WIDTH = 10.0            # m
NX = 40
NY = 4
SLOPE = 1.0e-3          # -
Q_TOTAL = 2.0           # m^3/s
N_MANNING = 0.03        # s/m^(1/3)

PIVLAB_TEXT = """PIVlab by W.Th. & E.J.S., ASCII chart output - 2026-10-10
FRAME: 1, filenames: a.tif & b.tif, conversion factor xy (px -> m): 0.001
x [m],y [m],u [m/s],v [m/s],Vector type [-]
1.0,2.0,0.5,0.1,1
3.0,4.0,NaN,NaN,0
5.0,6.0,0.7,-0.2,1
"""


def _normal_depth() -> float:
    """Manning normal depth h_n = (n q / sqrt(S))^(3/5) of the channel (m).

    Returns:
        h_n (m).
    """
    q = Q_TOTAL / WIDTH
    return (N_MANNING * q / np.sqrt(SLOPE)) ** 0.6


def _write_channel(case_dir) -> None:
    """Write a straight sloped channel with a steady solver, exit stage at the normal depth.

    With the exit at the normal depth the flow is close to uniform, so the velocities
    depend on Manning's n everywhere (no backwater from the exit).

    Args:
        case_dir: output folder (pytest ``tmp_path``).
    """
    h_normal = _normal_depth()

    def initial_wse(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Initial water surface: the bed z_b = S (L - x) plus the normal depth (m)."""
        return SLOPE * (LENGTH - x) + h_normal

    write_channel_case(
        case_dir, NX, NY,
        length=LENGTH,
        width=WIDTH,
        slope=SLOPE,
        bump=0.0,
        Q=Q_TOTAL,
        wse_out=_normal_depth(),
        n_manning=N_MANNING,
        solver="SteadyNewton()",
        ic_wse=initial_wse,
        order=2,
    )


def test_read_pivlab_and_csv(tmp_path) -> None:
    """PIVlab ASCII export and a CSV with renamed columns are read; NaN rows dropped."""
    # PIVlab-style file: title lines, units, a masked vector.
    pivlab = tmp_path / "pivlab.txt"
    pivlab.write_text(PIVLAB_TEXT)
    points = read_velocity_points(pivlab)
    np.testing.assert_array_equal(points.x, [1.0, 5.0])
    np.testing.assert_array_equal(points.u, [0.5, 0.7])
    np.testing.assert_array_equal(points.v, [0.1, -0.2])

    # CSV with other names for the velocity columns, and a comment line.
    csv = tmp_path / "probe.csv"
    csv.write_text("# probes\nX,Y,Vx (m/s),Vy (m/s)\n0.5,0.25,1.0,0.0\n")
    points = read_velocity_points(csv, {"u": "vx", "v": "vy"})
    np.testing.assert_array_equal(points.y, [0.25])
    np.testing.assert_array_equal(points.u, [1.0])

    # Space-separated file.
    spaced = tmp_path / "spaced.dat"
    spaced.write_text("x y u v\n1 2 3 4\n")
    np.testing.assert_array_equal(read_velocity_points(spaced).v, [4.0])


def test_transform_points() -> None:
    """Flip, 90-degree rotation, scale and shift of positions and velocities."""
    points = PointVelocities(
        x=np.array([1.0]), y=np.array([2.0]), u=np.array([0.3]), v=np.array([0.4]),
    )

    # Flip y, rotate by 90 degrees: (x, y) -> (x, -y) -> (y, x); then scale and shift.
    moved = transform_points(points, origin=(10.0, 20.0), rotation_deg=90.0, scale=2.0,
                             velocity_scale=0.5, flip_y=True)
    np.testing.assert_allclose(moved.x, [10.0 + 2.0 * 2.0], atol=1e-12)
    np.testing.assert_allclose(moved.y, [20.0 + 2.0 * 1.0], atol=1e-12)
    np.testing.assert_allclose(moved.u, [0.5 * 0.4], atol=1e-12)
    np.testing.assert_allclose(moved.v, [0.5 * 0.3], atol=1e-12)

    # Speeds scale by velocity_scale only.
    speed_before = np.hypot(points.u, points.v)
    speed_after = np.hypot(moved.u, moved.v)
    np.testing.assert_allclose(speed_after, 0.5 * speed_before)


def test_locate_cells(tmp_path) -> None:
    """Random points fall in the rectangle cell that contains them; outside points get -1."""
    _write_channel(tmp_path)
    mesh = setup_case(read_control(tmp_path / "run_control.json")).mesh

    # Random points inside the channel and a few outside.
    rng = np.random.default_rng(2)
    x = rng.uniform(0.0, LENGTH, 2000)
    y = rng.uniform(0.0, WIDTH, 2000)
    x_outside = np.array([-1.0, LENGTH + 1.0, 50.0])
    y_outside = np.array([5.0, 5.0, WIDTH + 1.0])

    # Run.
    cells = locate_cells(mesh, x, y)
    cells_outside = locate_cells(mesh, x_outside, y_outside)

    # Compare: every point lies inside the bounding box of its cell.
    assert np.all(cells >= 0)
    for point, cell in enumerate(cells):
        corners = mesh.nodes[mesh.cell_nodes[cell], :2]
        assert corners[:, 0].min() - 1e-9 <= x[point] <= corners[:, 0].max() + 1e-9
        assert corners[:, 1].min() - 1e-9 <= y[point] <= corners[:, 1].max() + 1e-9
    np.testing.assert_array_equal(cells_outside, [-1, -1, -1])


def test_bin_to_cells(tmp_path) -> None:
    """Cell means, counts, uncertainty, surface factor and the minimum point count."""
    _write_channel(tmp_path)
    mesh = setup_case(read_control(tmp_path / "run_control.json")).mesh

    # Four points in the first cell (around its centroid), one in a second cell.
    first = mesh.centroid[0]
    second = mesh.centroid[10]
    offsets = np.array([[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]])
    xy = np.vstack([first + offsets, second])
    points = PointVelocities(
        x=xy[:, 0], y=xy[:, 1],
        u=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        v=np.array([0.0, 0.0, 0.0, 0.4, 1.0]),
    )

    # Run: surface factor 0.8, point uncertainty 0.04 m/s, floor 0.003 m/s.
    binned = bin_to_cells(mesh, points, surface_factor=0.8, sigma_point=0.04,
                          sigma_floor=0.003)

    # Compare: the first cell has the mean of four points times 0.8.
    np.testing.assert_array_equal(binned.cells, [0, 10])
    np.testing.assert_array_equal(binned.count, [4, 1])
    np.testing.assert_allclose(binned.u, [0.8 * 2.5, 0.8 * 5.0])
    np.testing.assert_allclose(binned.v, [0.8 * 0.1, 0.8 * 1.0])
    expected_sigma = np.sqrt(0.04**2 / np.array([4.0, 1.0]) + 0.003**2)
    np.testing.assert_allclose(binned.sigma, expected_sigma)

    # With at least two points per cell, only the first cell is kept.
    assert bin_to_cells(mesh, points, min_points=2).cells.tolist() == [0]


def test_velocity_loss_and_gradient(tmp_path) -> None:
    """Loss zero at the true n, positive elsewhere; IFT gradient equals central differences."""
    # Setup: the steady solution at the true n gives synthetic "surface" data at random
    # points (surface velocity = depth average / 0.85), written as a PIVlab-style file in
    # a frame shifted by (5, 1) m.
    _write_channel(tmp_path)
    case = setup_case(read_control(tmp_path / "run_control.json"))
    model = case.model
    n_true = jnp.full(len(model.n_zone), N_MANNING)
    U_true, _ = final_state(model, {"n_zone": n_true}, case.U0, case.ode)
    depth = np.asarray(U_true[0] + model.hstill)
    u_cell = np.asarray(U_true[1]) / (depth + model.h_small)

    rng = np.random.default_rng(3)
    x = rng.uniform(20.0, 180.0, 400)
    y = rng.uniform(0.5, 9.5, 400)
    cells = locate_cells(case.mesh, x, y)
    surface_u = u_cell[cells] / 0.85
    rows = [f"{x[k] - 5.0},{y[k] - 1.0},{surface_u[k]},0.0" for k in range(len(x))]
    (tmp_path / "lspiv.csv").write_text("x [m],y [m],u [m/s],v [m/s]\n" + "\n".join(rows))

    # The velocity_data block of the run control.
    control_path = tmp_path / "run_control.json"
    control = json.loads(control_path.read_text())
    control["velocity_data"] = {
        "file": "lspiv.csv",
        "transform": {"origin": [5.0, 1.0]},
        "surface_factor": 0.85,
        "sigma_point": 0.02,
        "sigma_floor": 0.005,
        "binned_file": "lspiv_cells.csv",
    }
    control_path.write_text(json.dumps(control, indent=1))
    case = setup_case(read_control(control_path))
    data = load_velocity_data(case)
    assert (tmp_path / "lspiv_cells.csv").exists()

    def loss_of_n(n_zone: jax.Array) -> jax.Array:
        """Velocity loss (-) of the steady solution for Manning's n per zone."""
        theta = {"n_zone": n_zone}
        U, _ = final_state(model, theta, case.U0, case.ode)
        return velocity_loss(model, theta, U, data)

    # Run: loss at the true n and at a perturbed n; gradient against differences.
    loss_true = float(loss_of_n(n_true))
    n_off = n_true * 1.2
    loss_off = float(loss_of_n(n_off))
    gradient = jax.grad(loss_of_n)(n_off)
    step = 1e-6
    plus = loss_of_n(n_off.at[1].add(step))
    minus = loss_of_n(n_off.at[1].add(-step))
    finite_difference = float((plus - minus) / (2.0 * step))

    # Compare.
    assert loss_true < 1e-12
    assert loss_off > 0.3
    np.testing.assert_allclose(float(gradient[1]), finite_difference, rtol=1e-4)

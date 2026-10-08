"""Static description of one SWE problem: mesh arrays, BCs, ICs, resistance model.

Purpose
-------
``SWEModel`` holds everything that is *not* differentiated (JAX constants): mesh
connectivity and geometry, boundary-face kinds, the bed, the still-water level and the
resistance model. It also holds default values for the quantities that can become
parameters: the bed elevation ``zb``, the Manning's n of each zone, the inlet discharges
and the exit water-surface elevations (WSEs).

Parameter dictionary
--------------------
The residual takes a parameter dictionary ``theta`` that may override any of those
defaults:

    theta = {"zb": [n_cells], "n_zone": [n_materials], "Q": [n_inlets],
             "WSE": [n_exits], "nn": <MLP pytree>}

Mesh arrays
-----------
Interior faces (prefix ``fi_``) connect an owner cell o and a neighbour cell r; the unit
normal (nx, ny) points from o to r. Boundary faces (prefix ``fb_``) belong to one cell and
carry a boundary kind (wall, inlet-Q, exit-H, symmetry). All indices are 0-based.

Units
-----
Units follow the mesh: SI (m, s, g = 9.81 m/s^2, k_n = 1) in all Hydrograd.jl examples,
or US customary (ft, s, g = 32.2 ft/s^2, k_n = 1.486) for ``GridUnit = "EN"``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np
from jax import Array

from hydrograd.bed import BedGeometry, bed_slope
from hydrograd.mesh import Mesh
from hydrograd.params.manning import ManningLaw

logger = logging.getLogger(__name__)

__all__ = ["UNITS", "Resistance", "SWEModel", "Turbulence", "Vegetation", "build_model"]

# GridUnit -> (unit system, k_n, g (length/s^2)). k_n is the unit factor of Manning's
# formula: U = (k_n / n) R^(2/3) S^(1/2).
UNITS = {"SI": ("SI", 1.0, 9.81), "EN": ("EN", 1.486, 32.2)}

# Integer codes of the boundary-face kinds (stored in ``SWEModel.fb_kind``).
KIND_WALL = 0
KIND_INLET = 1
KIND_EXIT = 2
KIND_SYMM = 3

# SRH-2D boundary type names -> kind codes.
_KINDS = {"WALL": KIND_WALL, "INLET-Q": KIND_INLET, "EXIT-H": KIND_EXIT, "SYMM": KIND_SYMM}


@dataclass(frozen=True)
class Resistance:
    """How bed resistance is computed.

    Attributes:
        mode: ``zone`` (n per material), ``law`` (n(h[,U,ks]) via ``law``),
            ``nn_ManningN_h``, ``nn_ManningN_h_Umag_ks`` (NN predicts n), or
            ``nn_FlowResistance`` (NN predicts the friction magnitude).
        law: the variable-n law for ``mode == "law"``.
        activations: hidden-layer activations of the NN.
        out_bounds: NN output bounds (n in s/m^(1/3), or friction in m^2/s^2).
        h_bounds, umag_bounds, ks_bounds, q_bounds: input normalization ranges
            (m, m/s, m, m^2/s).
        ln_axes: LayerNorm axes: ``features`` (local law n_i = f(h_i), default) or
            ``all`` (Hydrograd.jl/Lux whole-array normalization; parity tests only).
    """

    mode: str = "zone"
    law: ManningLaw | None = None
    activations: tuple[str, ...] = ()
    out_bounds: tuple[float, float] = (0.0, 1.0)
    h_bounds: tuple[float, float] = (0.0, 1.0)
    umag_bounds: tuple[float, float] = (0.0, 1.0)
    ks_bounds: tuple[float, float] = (0.0, 1.0)
    q_bounds: tuple[float, float] = (0.0, 1.0)
    ln_axes: str = "features"


@dataclass(frozen=True)
class Vegetation:
    """Drag of rigid emergent vegetation (stems), as a momentum sink per unit plan area.

    The drag force per unit plan area divided by the water density is

        F / rho = 1/2 C_D a h |u| u     (m^2/s^2),

    with a the frontal area per unit volume (1/m), h the depth (m) and u the depth-averaged
    velocity (m/s). With q = h u it equals c_veg q, where c_veg = 1/2 C_D a |u| (1/s); this
    rate is added to the bed-friction rate (see ``physics.vegetation``).

    Attributes:
        mode: ``none`` (no vegetation, default), ``zone`` (C_D per material zone),
            ``nn_CD`` (a local neural-network closure C_D = f(h, |u|, a), weights in
            ``theta["nn"]``) or ``struct_CD`` (the four-parameter law
            C_D = C_0 (a / a_ref)^beta [1 + A exp(-feature / L)], parameters in
            ``theta["veg_struct"]``, see ``physics.vegetation``).
        a_zone: frontal area per unit volume a = d / dS^2 for each material ID (1/m);
            0 where there is no vegetation.
        CD_zone: drag coefficient C_D for each material ID (-), for ``mode == "zone"``.
        activations: hidden-layer activations of the C_D network.
        CD_bounds: output bounds of the network (-).
        h_bounds, umag_bounds, a_bounds: input normalization ranges (m, m/s, 1/m).
        feature: optional fourth input of the C_D network, one value per cell (e.g. the
            frontal area traversed from the leading edge of a patch, a s, which carries
            the sheltering inside a patch); empty for the three-input network.
        feature_bounds: normalization range of ``feature``.
        a_ref: reference frontal area per unit volume of the ``struct_CD`` law (1/m).
    """

    mode: str = "none"
    a_zone: tuple[float, ...] = ()
    CD_zone: tuple[float, ...] = ()
    activations: tuple[str, ...] = ()
    CD_bounds: tuple[float, float] = (0.0, 3.0)
    h_bounds: tuple[float, float] = (0.0, 1.0)
    umag_bounds: tuple[float, float] = (0.0, 1.0)
    a_bounds: tuple[float, float] = (0.0, 10.0)
    feature: tuple[float, ...] = ()
    feature_bounds: tuple[float, float] = (0.0, 1.0)
    a_ref: float = 1.0


@dataclass(frozen=True)
class Turbulence:
    """Horizontal turbulent diffusion of momentum (depth-averaged eddy viscosity).

    The momentum equations get the term div(h nu_t grad u) (m^2/s^2), with

    - ``parabolic``: nu_t = C_t u_* h + nu_0, where u_* = sqrt(g) n |u| / (k_n h^(1/6)) is
      the bed friction velocity (m/s) (the SRH-2D parabolic model, C_t = 0.7 by default);
    - ``constant``: nu_t = ``nu_constant`` + nu_0.

    Attributes:
        mode: ``none`` (no turbulent diffusion, default), ``parabolic`` or ``constant``.
        coefficient: C_t of the parabolic model (-); differentiable as ``theta["eddy_Ct"]``.
        nu0: background (molecular) viscosity nu_0 (m^2/s).
        nu_constant: eddy viscosity of the ``constant`` model (m^2/s).
    """

    mode: str = "none"
    coefficient: float = 0.7
    nu0: float = 1.0e-6
    nu_constant: float = 0.0


@dataclass(frozen=True)
class WallFriction:
    """Shear stress of the side walls (WALL boundary faces) from the log law of the wall.

    Without it a WALL face is free-slip: no mass flux and no tangential stress. With
    ``mode == "loglaw"`` each wall face exerts the stress tau_w / rho = u_*^2 against the
    tangential velocity u_t of its cell, over the wetted wall area h L_face. The friction
    velocity u_* (m/s) solves the combined smooth/rough log law

        u_t / u_* = (1 / kappa) ln(y / y_0),   y_0 = nu / (E u_*) + k_s / 30,

    with y the distance from the cell centroid to the wall (m), and the viscous-sublayer
    law u_t / u_* = y u_* / nu where y+ = y u_* / nu < y+_lam (see ``physics.walls``).

    Attributes:
        mode: ``none`` (free-slip walls, default) or ``loglaw``.
        ks: equivalent sand roughness of the walls k_s (m); 0 for hydraulically smooth
            walls. Differentiable as ``theta["wall_ks"]``.
        nu: kinematic viscosity of water (m^2/s; ft^2/s in EN units).
        kappa: von Karman constant (-).
        E: smooth-wall log-law constant (-) (9.8 as in OpenFOAM wall functions).
        yplus_lam: y+ of the switch between the viscous-sublayer and log laws (-).
        iterations: fixed-point iterations for u_* (unrolled, differentiable).
    """

    mode: str = "none"
    ks: float = 0.0
    nu: float = 1.0e-6
    kappa: float = 0.41
    E: float = 9.8
    yplus_lam: float = 11.53
    iterations: int = 12


@dataclass(frozen=True, eq=False)
class SWEModel:
    """Static arrays of the semi-discrete SWE problem (all JAX arrays, 0-based).

    Units follow the mesh (SI in all Hydrograd.jl examples): lengths m, time s.
    n_fi is the number of interior faces, n_fb the number of boundary faces.

    Attributes:
        n_cells: number of cells.
        g: gravitational acceleration (m/s^2).
        k_n: unit factor of Manning's formula (1 in SI, 1.486 in EN).
        h_small: dry threshold depth (m).
        area: cell areas (m^2), [n_cells].
        fi_o, fi_r: owner and neighbour cell of each interior face, [n_fi].
        fi_nx, fi_ny: unit normal of each interior face, from owner to neighbour, [n_fi].
        fi_L: interior face lengths (m), [n_fi].
        fb_c: cell of each boundary face, [n_fb].
        fb_nx, fb_ny: outward unit normal of each boundary face, [n_fb].
        fb_L: boundary face lengths (m), [n_fb].
        fb_kind: boundary kind code (``KIND_*``) of each boundary face, [n_fb].
        fb_inlet: index into theta["Q"] per boundary face (0 for non-inlet faces), [n_fb].
        fb_exit: index into theta["WSE"] per boundary face (0 for non-exit faces), [n_fb].
        n_inlets, n_exits: number of inlet-Q and exit-H boundaries.
        bed: index arrays for bed face values and slopes.
        zb: default bed elevation at cells (m), [n_cells].
        S0: bed slope of ``zb`` (-), [n_cells, 2].
        hstill: still-water depth h_still = w_still - zb (m), [n_cells].
        wstill: still-water surface elevation (m), [n_cells].
        mat_id: material (zone) ID per cell, [n_cells].
        ks: roughness height per cell (m), [n_cells]; zeros if unused.
        n_zone: default Manning's n per material ID (s/m^(1/3)), [n_materials].
        Q: inlet discharges (m^3/s), sorted BC-ID order: [n_in] or [n_in, n_t].
        WSE: exit WSEs (m), sorted BC-ID order: [n_ex] or [n_ex, n_t].
        inlet_ids, exit_ids: SRH-2D boundary IDs of the inlets and exits, sorted.
        resistance: the bed-resistance model.
        inlet_split: how Q is split over inlet faces: "conveyance" (L h^(5/3)/n) or
            "julia" (L^(5/3) h/n, Hydrograd.jl).
        scheme: "hr_hllc" (wet/dry, default) or "julia_xi_roe" (Hydrograd.jl).
        order: spatial order of "hr_hllc": 2 (MUSCL, default) or 1.
        limiter: order-2 limiter: "venkatakrishnan" (smooth) or "barth_jespersen".
        dry_smoothing: smoothing width (m); > 0 smooths max(h, 0) at wet/dry fronts
            (robustness of the implicit Newton solve).
        fi_ro: interior-face midpoint minus owner centroid (m), [n_fi, 2].
        fi_rr: interior-face midpoint minus neighbour centroid (m), [n_fi, 2].
        fb_ro: boundary-face midpoint minus its cell centroid (m), [n_fb, 2].
        loss_velocity_eps: eps (m) in u = q/(h + eps) in the losses; None -> h_small
            (as in the truth files).
        Q_times: time grid (s) when Q is a time series, [n_t].
        WSE_times: time grid (s) when WSE is a time series, [n_t].
    """

    n_cells: int
    g: float
    k_n: float
    h_small: float
    area: Array
    fi_o: Array
    fi_r: Array
    fi_nx: Array
    fi_ny: Array
    fi_L: Array
    fb_c: Array
    fb_nx: Array
    fb_ny: Array
    fb_L: Array
    fb_kind: Array
    fb_inlet: Array        # index into theta["Q"] (0 for non-inlet faces)
    fb_exit: Array         # index into theta["WSE"] (0 for non-exit faces)
    n_inlets: int
    n_exits: int
    bed: BedGeometry
    zb: Array              # default bed elevation at cells (m)
    S0: Array              # its slope, [n_cells, 2]
    hstill: Array          # still-water depth h_still = w_still - zb (m)
    wstill: Array          # still-water surface elevation (m)
    mat_id: Array          # material (zone) ID per cell
    ks: Array              # roughness height per cell (m); zeros if unused
    n_zone: Array          # default Manning's n per material ID
    Q: Array               # inlet discharges (m^3/s), sorted BC-ID order: [n_in] or [n_in, n_t]
    WSE: Array             # exit WSEs (m), sorted BC-ID order: [n_ex] or [n_ex, n_t]
    inlet_ids: tuple[int, ...] = ()
    exit_ids: tuple[int, ...] = ()
    resistance: Resistance = field(default_factory=Resistance)
    inlet_split: str = "conveyance"    # "conveyance" (L h^(5/3)/n) or "julia" (L^(5/3) h/n)
    scheme: str = "hr_hllc"            # "hr_hllc" (wet/dry, default) or "julia_xi_roe"
    order: int = 2                     # spatial order of "hr_hllc": 2 (MUSCL, default) or 1
    limiter: str = "venkatakrishnan"   # order-2 limiter: "venkatakrishnan" or "barth_jespersen"
    dry_smoothing: float = 0.0         # m; > 0 smooths max(h, 0) at wet/dry fronts
    fi_ro: Array | None = None         # interior-face midpoint - owner centroid (m), [n_fi, 2]
    fi_rr: Array | None = None         # interior-face midpoint - neighbour centroid (m)
    fb_ro: Array | None = None         # boundary-face midpoint - cell centroid (m), [n_fb, 2]
    loss_velocity_eps: float | None = None  # u = q/(h + eps) in losses; None -> h_small
    Q_times: Array | None = None       # time grid (s) when Q is a time series
    WSE_times: Array | None = None     # time grid (s) when WSE is a time series
    vegetation: Vegetation = field(default_factory=Vegetation)  # stem drag; none by default
    turbulence: Turbulence = field(default_factory=Turbulence)  # eddy viscosity; none by default
    wall_friction: WallFriction = field(default_factory=WallFriction)  # free-slip by default

    def __hash__(self) -> int:
        """Identity hash, so that a model can be a static argument of ``jax.jit``.

        Returns:
            ``id(self)``.
        """
        return id(self)


def _boundary_values(values: dict, ids: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray | None]:
    """Numbers and/or TimeSeries per boundary -> (array, time grid or None).

    If any boundary has a series, all are sampled on the union of the series times
    (exact for piecewise-linear series; constants are repeated).

    Args:
        values: boundary ID -> a number or a ``TimeSeries`` (m^3/s for Q, m for WSE).
        ids: boundary IDs in the order wanted (sorted).

    Returns:
        (values, times): values of shape [n_boundaries] when all are constant (times is
        then None), or of shape [n_boundaries, n_times] sampled on the time grid
        ``times`` (s), shape [n_times].
    """
    from hydrograd.io.srh2d import TimeSeries

    # All boundaries constant: one value per boundary, no time grid.
    series = [values[i] for i in ids if isinstance(values[i], TimeSeries)]
    if not series:
        constants = [float(values[i]) for i in ids]
        return np.array(constants, dtype=float), None

    # Common time grid: the union of all series times.
    grid = np.unique(np.concatenate([ts.t for ts in series]))

    # Sample each boundary on the grid (series are interpolated, constants repeated).
    rows = []
    for i in ids:
        if isinstance(values[i], TimeSeries):
            row = np.interp(grid, values[i].t, values[i].v)
        else:
            row = np.full(len(grid), float(values[i]))
        rows.append(row)

    return np.stack(rows), grid


def build_model(mesh: Mesh, zb_cells: np.ndarray, wstill: np.ndarray, mat_id: np.ndarray,
                n_zone: Sequence[float], inlet_Q: dict, exit_WSE: dict,
                resistance: Resistance = Resistance(), ks_cells: np.ndarray | None = None,
                unit_system: str = "SI", h_small: float = 1.0e-3) -> SWEModel:
    """Assemble an :class:`SWEModel`.

    Args:
        mesh: the mesh (its ``bc_types`` define the boundary kinds).
        zb_cells: bed elevation at cells (m), shape [n_cells].
        wstill: still-water surface elevation at cells (m), from the IC, shape [n_cells].
        mat_id: material ID per cell, shape [n_cells].
        n_zone: Manning's n per material ID 0..n_materials-1 (s/m^(1/3)).
        inlet_Q: inlet-Q boundary ID -> total discharge (m^3/s), a number or a TimeSeries.
        exit_WSE: exit-H boundary ID -> water-surface elevation (m), a number or a TimeSeries.
        resistance: friction model.
        ks_cells: roughness height per cell (m), for ``h_Umag_ks`` laws, shape [n_cells].
        unit_system: ``SI`` or ``EN`` (sets g and k_n).
        h_small: dry threshold depth (m).

    Returns:
        The assembled model.

    Raises:
        ValueError: on an unsupported boundary type, or a material ID without a Manning's n.
    """
    _, k_n, g = UNITS[unit_system]

    # Interior faces have a neighbour cell (index >= 0); boundary faces do not.
    inner = mesh.neigh >= 0
    bfaces, bids = mesh.boundary_arrays()

    # Inlet-Q and exit-H boundary IDs, sorted; they index theta["Q"] and theta["WSE"].
    inlet_ids = tuple(sorted(i for i, t in mesh.bc_types.items() if t == "INLET-Q"))
    exit_ids = tuple(sorted(i for i, t in mesh.bc_types.items() if t == "EXIT-H"))

    # Check the inputs.
    bad = sorted({t for t in mesh.bc_types.values() if t not in _KINDS})
    if bad:
        raise ValueError(
            f"unsupported boundary condition type(s) {bad}; supported: {sorted(_KINDS)}"
        )
    if int(np.max(mat_id)) >= len(n_zone):
        raise ValueError(
            f"material ID {int(np.max(mat_id))} has no Manning's n ({len(n_zone)} zones given)"
        )

    # Per boundary face: kind code, and position of its boundary in theta["Q"] / ["WSE"].
    kind = np.array([_KINDS[mesh.bc_types[b]] for b in bids])
    inlet_pos = {b: k for k, b in enumerate(inlet_ids)}
    exit_pos = {b: k for k, b in enumerate(exit_ids)}
    fb_inlet = np.array([inlet_pos.get(b, 0) for b in bids])
    fb_exit = np.array([exit_pos.get(b, 0) for b in bids])

    # Boundary values: constants, or series sampled on a common time grid.
    Q_values, Q_times = _boundary_values(inlet_Q, inlet_ids)
    WSE_values, WSE_times = _boundary_values(exit_WSE, exit_ids)

    # Bed geometry and the default bed elevation.
    bed = BedGeometry(mesh)
    zb = jnp.asarray(zb_cells, dtype=jnp.float64)

    # Roughness heights (zeros when no law uses them).
    ks = ks_cells if ks_cells is not None else np.zeros(mesh.n_cells)

    # Vectors from the owner / neighbour centroid to the interior-face midpoint (MUSCL).
    owner_inner = mesh.owner[inner]
    neigh_inner = mesh.neigh[inner]
    fi_ro = mesh.midpoint[inner] - mesh.centroid[owner_inner]
    fi_rr = mesh.midpoint[inner] - mesh.centroid[neigh_inner]

    # Vector from the cell centroid to each boundary-face midpoint (wall distance).
    fb_ro = mesh.midpoint[bfaces] - mesh.centroid[mesh.owner[bfaces]]

    return SWEModel(
        n_cells=mesh.n_cells,
        g=g,
        k_n=k_n,
        h_small=h_small,
        area=jnp.asarray(mesh.area),
        fi_o=jnp.asarray(owner_inner),
        fi_r=jnp.asarray(neigh_inner),
        fi_nx=jnp.asarray(mesh.normal[inner, 0]),
        fi_ny=jnp.asarray(mesh.normal[inner, 1]),
        fi_L=jnp.asarray(mesh.length[inner]),
        fb_c=jnp.asarray(mesh.owner[bfaces]),
        fb_nx=jnp.asarray(mesh.normal[bfaces, 0]),
        fb_ny=jnp.asarray(mesh.normal[bfaces, 1]),
        fb_L=jnp.asarray(mesh.length[bfaces]),
        fb_kind=jnp.asarray(kind),
        fb_inlet=jnp.asarray(fb_inlet),
        fb_exit=jnp.asarray(fb_exit),
        n_inlets=len(inlet_ids),
        n_exits=len(exit_ids),
        bed=bed,
        zb=zb,
        S0=bed_slope(bed, zb),
        hstill=jnp.asarray(wstill - zb_cells),
        wstill=jnp.asarray(wstill),
        mat_id=jnp.asarray(mat_id),
        ks=jnp.asarray(ks),
        n_zone=jnp.asarray(np.asarray(n_zone, dtype=float)),
        Q=jnp.asarray(Q_values),
        WSE=jnp.asarray(WSE_values),
        inlet_ids=inlet_ids,
        exit_ids=exit_ids,
        resistance=resistance,
        Q_times=None if Q_times is None else jnp.asarray(Q_times),
        WSE_times=None if WSE_times is None else jnp.asarray(WSE_times),
        fi_ro=jnp.asarray(fi_ro),
        fi_rr=jnp.asarray(fi_rr),
        fb_ro=jnp.asarray(fb_ro),
    )

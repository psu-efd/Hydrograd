"""Static description of one SWE problem: mesh arrays, BCs, ICs, resistance model.

``SWEModel`` holds everything that is *not* differentiated (JAX constants), plus
default values for the quantities that can become parameters (bed ``zb``, zone
Manning's n, inlet discharges, exit WSEs). The residual takes a parameter
dictionary ``theta`` that may override any of those defaults:

    theta = {"zb": [n_cells], "n_zone": [n_materials], "Q": [n_inlets],
             "WSE": [n_exits], "nn": <MLP pytree>}
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

__all__ = ["UNITS", "Resistance", "SWEModel", "build_model"]

# GridUnit -> (unit system, k_n, g (length/s^2))
UNITS = {"SI": ("SI", 1.0, 9.81), "EN": ("EN", 1.486, 32.2)}
KIND_WALL, KIND_INLET, KIND_EXIT, KIND_SYMM = 0, 1, 2, 3
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


@dataclass(frozen=True, eq=False)
class SWEModel:
    """Static arrays of the semi-discrete SWE problem (all JAX arrays, 0-based).

    Units follow the mesh (SI in all Hydrograd.jl examples): lengths m, time s.
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
    inlet_split: str = "conveyance"   # "conveyance" (L h^(5/3)/n) or "julia" (L^(5/3) h/n)
    scheme: str = "hr_hllc"            # "hr_hllc" (wet/dry, default) or "julia_xi_roe" (Hydrograd.jl)
    order: int = 2                     # spatial order of "hr_hllc": 2 (MUSCL, default) or 1
    limiter: str = "venkatakrishnan"   # order-2 limiter: "venkatakrishnan" (smooth) or "barth_jespersen"
    fi_ro: Array | None = None      # interior-face midpoint - owner centroid (m), [n_fi, 2]
    fi_rr: Array | None = None      # interior-face midpoint - neighbour centroid (m), [n_fi, 2]
    loss_velocity_eps: float | None = None
    Q_times: Array | None = None     # time grid (s) when Q is a time series
    WSE_times: Array | None = None   # time grid (s) when WSE is a time series  # u = q/(h + eps) in losses; None -> h_small (as in truth files)

    def __hash__(self) -> int:
        return id(self)


def _boundary_values(values: dict, ids: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray | None]:
    """Numbers and/or TimeSeries per boundary -> (array, time grid or None).

    If any boundary has a series, all are sampled on the union of the series times
    (exact for piecewise-linear series; constants are repeated).
    """
    from hydrograd.io.srh2d import TimeSeries

    series = [values[i] for i in ids if isinstance(values[i], TimeSeries)]
    if not series:
        return np.array([float(values[i]) for i in ids], dtype=float), None
    grid = np.unique(np.concatenate([ts.t for ts in series]))
    rows = [np.interp(grid, values[i].t, values[i].v) if isinstance(values[i], TimeSeries)
            else np.full(len(grid), float(values[i])) for i in ids]
    return np.stack(rows), grid


def build_model(mesh: Mesh, zb_cells: np.ndarray, wstill: np.ndarray, mat_id: np.ndarray,
                n_zone: Sequence[float], inlet_Q: dict, exit_WSE: dict,
                resistance: Resistance = Resistance(), ks_cells: np.ndarray | None = None,
                unit_system: str = "SI", h_small: float = 1.0e-3) -> SWEModel:
    """Assemble an :class:`SWEModel`.

    Args:
        mesh: the mesh (its ``bc_types`` define the boundary kinds).
        zb_cells: bed elevation at cells (m).
        wstill: still-water surface elevation at cells (m), from the IC.
        mat_id: material ID per cell.
        n_zone: Manning's n per material ID 0..n_materials-1.
        inlet_Q: inlet-Q boundary ID -> total discharge (m^3/s), a number or a TimeSeries.
        exit_WSE: exit-H boundary ID -> water-surface elevation (m), a number or a TimeSeries.
        resistance: friction model.
        ks_cells: roughness height per cell (m), for ``h_Umag_ks`` laws.
        unit_system: ``SI`` or ``EN`` (sets g and k_n).
        h_small: dry threshold depth (m).
    """
    _, k_n, g = UNITS[unit_system]
    inner = mesh.neigh >= 0
    bfaces, bids = mesh.boundary_arrays()
    inlet_ids = tuple(sorted(i for i, t in mesh.bc_types.items() if t == "INLET-Q"))
    exit_ids = tuple(sorted(i for i, t in mesh.bc_types.items() if t == "EXIT-H"))
    bad = sorted({t for t in mesh.bc_types.values() if t not in _KINDS})
    if bad:
        raise ValueError(f"unsupported boundary condition type(s) {bad}; supported: {sorted(_KINDS)}")
    if int(np.max(mat_id)) >= len(n_zone):
        raise ValueError(f"material ID {int(np.max(mat_id))} has no Manning's n ({len(n_zone)} zones given)")
    kind = np.array([_KINDS[mesh.bc_types[b]] for b in bids])
    inlet_pos = {b: k for k, b in enumerate(inlet_ids)}
    exit_pos = {b: k for k, b in enumerate(exit_ids)}
    Qv, Qt = _boundary_values(inlet_Q, inlet_ids)
    Wv, Wt = _boundary_values(exit_WSE, exit_ids)
    bed = BedGeometry(mesh)
    zb = jnp.asarray(zb_cells, dtype=jnp.float64)
    j = lambda a: jnp.asarray(a)
    return SWEModel(
        n_cells=mesh.n_cells, g=g, k_n=k_n, h_small=h_small, area=j(mesh.area),
        fi_o=j(mesh.owner[inner]), fi_r=j(mesh.neigh[inner]),
        fi_nx=j(mesh.normal[inner, 0]), fi_ny=j(mesh.normal[inner, 1]), fi_L=j(mesh.length[inner]),
        fb_c=j(mesh.owner[bfaces]), fb_nx=j(mesh.normal[bfaces, 0]), fb_ny=j(mesh.normal[bfaces, 1]),
        fb_L=j(mesh.length[bfaces]), fb_kind=j(kind),
        fb_inlet=j(np.array([inlet_pos.get(b, 0) for b in bids])),
        fb_exit=j(np.array([exit_pos.get(b, 0) for b in bids])),
        n_inlets=len(inlet_ids), n_exits=len(exit_ids), bed=bed, zb=zb, S0=bed_slope(bed, zb),
        hstill=j(wstill - zb_cells), wstill=j(wstill), mat_id=j(mat_id),
        ks=j(ks_cells if ks_cells is not None else np.zeros(mesh.n_cells)),
        n_zone=j(np.asarray(n_zone, dtype=float)),
        Q=j(Qv), WSE=j(Wv),
        inlet_ids=inlet_ids, exit_ids=exit_ids, resistance=resistance,
        Q_times=None if Qt is None else j(Qt), WSE_times=None if Wt is None else j(Wt),
        fi_ro=j(mesh.midpoint[inner] - mesh.centroid[mesh.owner[inner]]),
        fi_rr=j(mesh.midpoint[inner] - mesh.centroid[mesh.neigh[inner]]))

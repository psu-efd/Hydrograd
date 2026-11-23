"""Calibrated parameters in unconstrained variables: transforms, selection and priors.

Why
---
An optimizer works best on variables without bounds and of comparable size. Each physical
parameter value v is therefore written as a function of an unconstrained variable u:

    transform    v(u)                                           use
    log          exp(u)                                         positive values
    affine       lo + (hi - lo) sigmoid(u)                      values in (lo, hi)
    logaffine    exp(ln lo + (ln hi - ln lo) sigmoid(u))        bounded on a log scale
    softmax      exp(u_k) / sum_j exp(u_j)                      fractions that sum to one
    identity     u                                              no constraint

A derivative with respect to u of a ``log`` parameter is a derivative per unit of ln v,
so the entries of a gradient are directly comparable between parameters.

Selection
---------
A model has many parameters (``theta``, a dictionary of arrays); a calibration frees a
few of them. The active parameters are named by labels: ``"a_eh"`` (a whole parameter)
or ``"kappa_w[0]"`` (one entry of a vector parameter). :class:`ParameterMap` holds the
reference ``theta``, the specifications and the labels, and converts between the
optimizer's variables (a dictionary label -> unconstrained array) and the full ``theta``.

Priors
------
A Gaussian prior on the unconstrained variable, centred on the reference value, with the
standard deviation ``prior_sd`` of the specification:

    log p(u) = -1/2 sum ((u - u_ref) / prior_sd)^2          (constants dropped).

For a ``log`` parameter, prior_sd = 0.7 means "within a factor of about 2".

The module is independent of the flow model: the 1-D mode uses it, and 2-D parameters
can use it in the same way.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

__all__ = [
    "ParamSpec",
    "ParameterMap",
    "parse_label",
    "to_constrained",
    "to_unconstrained",
    "with_values",
]

# Differentiable parameters: name -> array of physical values.
Theta = dict[str, Any]

# Clip of the normalized value y in the inverse of the sigmoid transforms (-).
Y_CLIP = 1.0e-6

# Floor of a fraction in the inverse of the softmax transform (-).
FRACTION_FLOOR = 1.0e-12

# A label is a parameter name, optionally followed by an index in square brackets.
_LABEL = re.compile(r"^([A-Za-z_]\w*)(?:\[(\d+)\])?$")


@dataclass(frozen=True)
class ParamSpec:
    """How one parameter is transformed and regularized.

    Attributes:
        name: parameter name (key of ``theta``).
        transform: ``"log"``, ``"affine"``, ``"logaffine"``, ``"softmax"`` or
            ``"identity"`` (see the module docstring).
        lo: lower bound of the ``affine`` and ``logaffine`` transforms (units of the
            parameter).
        hi: upper bound (units of the parameter).
        prior_sd: standard deviation of the Gaussian prior on the unconstrained variable
            (-); None for no prior.
        doc: one-line description with units.
    """

    name: str
    transform: str = "log"
    lo: float = 0.0
    hi: float = 1.0
    prior_sd: float | None = None
    doc: str = ""


def _logit(y: Array) -> Array:
    """Inverse of the sigmoid, with y clipped away from 0 and 1.

    Args:
        y: values in [0, 1] (-), any shape.

    Returns:
        ln(y / (1 - y)), same shape.
    """
    y = jnp.clip(y, Y_CLIP, 1.0 - Y_CLIP)
    return jnp.log(y / (1.0 - y))


def to_unconstrained(spec: ParamSpec, value: Array) -> Array:
    """Unconstrained variable u of a physical value v.

    Args:
        spec: specification of the parameter.
        value: physical value(s) (units of the parameter), any shape.

    Returns:
        u (-), same shape as ``value``.
    """
    value = jnp.asarray(value, dtype=float)

    if spec.transform == "log":
        return jnp.log(value)

    if spec.transform == "affine":
        return _logit((value - spec.lo) / (spec.hi - spec.lo))

    if spec.transform == "logaffine":
        log_lo = math.log(spec.lo)
        log_hi = math.log(spec.hi)
        return _logit((jnp.log(value) - log_lo) / (log_hi - log_lo))

    if spec.transform == "softmax":
        return jnp.log(jnp.maximum(value, FRACTION_FLOOR))

    if spec.transform == "identity":
        return value

    raise ValueError(f"Unknown transform {spec.transform!r} of parameter {spec.name!r}")


def to_constrained(spec: ParamSpec, u: Array) -> Array:
    """Physical value v of an unconstrained variable u.

    Args:
        spec: specification of the parameter.
        u: unconstrained variable(s) (-), any shape (``softmax``: the size classes are
            along axis 0).

    Returns:
        v (units of the parameter), same shape as ``u``.
    """
    if spec.transform == "log":
        return jnp.exp(u)

    if spec.transform == "affine":
        return spec.lo + (spec.hi - spec.lo) * jax.nn.sigmoid(u)

    if spec.transform == "logaffine":
        log_lo = math.log(spec.lo)
        log_hi = math.log(spec.hi)
        return jnp.exp(log_lo + (log_hi - log_lo) * jax.nn.sigmoid(u))

    if spec.transform == "softmax":
        return jax.nn.softmax(u, axis=0)

    if spec.transform == "identity":
        return u

    raise ValueError(f"Unknown transform {spec.transform!r} of parameter {spec.name!r}")


def parse_label(label: str) -> tuple[str, int | None]:
    """Split a label into the parameter name and the optional entry index.

    Args:
        label: ``"name"`` or ``"name[i]"``.

    Returns:
        Tuple ``(name, index)``; ``index`` is None for a whole parameter.
    """
    match = _LABEL.match(label.strip())
    if match is None:
        raise ValueError(f"Invalid parameter label {label!r}; use 'name' or 'name[index]'")

    name = match.group(1)
    index = None if match.group(2) is None else int(match.group(2))
    return name, index


def with_values(theta: Theta, values: Mapping[str, Any]) -> Theta:
    """Copy of ``theta`` with the physical values of some labels replaced.

    Args:
        theta: parameter dictionary (physical values).
        values: label -> physical value: a scalar for ``"name[i]"``, a scalar or an array
            of the parameter's shape for ``"name"`` (a scalar fills the whole parameter).

    Returns:
        A new parameter dictionary; ``theta`` is not changed.
    """
    updated = dict(theta)
    for label, value in values.items():
        name, index = parse_label(label)
        if name not in updated:
            raise ValueError(f"Parameter {name!r} (label {label!r}) is not in theta")

        current = jnp.asarray(updated[name], dtype=float)
        if index is None:
            updated[name] = jnp.broadcast_to(jnp.asarray(value, dtype=float), current.shape)
        else:
            flat = jnp.reshape(current, (-1,)).at[index].set(value)
            updated[name] = jnp.reshape(flat, current.shape)

    return updated


class ParameterMap:
    """Conversion between the optimizer's variables and the full parameter dictionary.

    The optimizer's variables are a dictionary ``label -> unconstrained array``: a scalar
    for a label with an index, and an array of the parameter's shape for a whole
    parameter. All other parameters keep their reference values.
    """

    def __init__(
        self, specs: Mapping[str, ParamSpec], theta_reference: Theta, labels: Sequence[str],
    ):
        """Set up the map.

        Args:
            specs: specification of every parameter that may be active, by name.
            theta_reference: reference values of all parameters (physical). They are the
                fixed values of the inactive parameters and the centre of the priors.
            labels: active parameters, e.g. ``["a_eh", "kappa_w[0]"]``.
        """
        self.specs = dict(specs)
        self.theta_reference = {
            name: jnp.asarray(value, dtype=float) for name, value in theta_reference.items()
        }
        self.labels = tuple(labels)

        # Parameter name and entry index of every label, with checks.
        self.targets: dict[str, tuple[str, int | None]] = {}
        for label in self.labels:
            name, index = parse_label(label)
            if name not in self.specs:
                raise ValueError(f"No specification for parameter {name!r} (label {label!r})")
            if name not in self.theta_reference:
                raise ValueError(f"Parameter {name!r} (label {label!r}) is not in theta")
            size = int(self.theta_reference[name].size)
            if index is not None and index >= size:
                raise ValueError(f"Label {label!r}: {name} has only {size} entries")
            self.targets[label] = (name, index)

        # Unconstrained reference values of the parameters that have active labels.
        self.u_reference: dict[str, Array] = {}
        for name, _ in self.targets.values():
            if name not in self.u_reference:
                self.u_reference[name] = to_unconstrained(
                    self.specs[name], self.theta_reference[name],
                )

    def unconstrained(self, theta: Theta | None = None) -> dict[str, Array]:
        """Optimizer variables for given physical values.

        Args:
            theta: physical values of (at least) the active parameters; None uses the
                reference values.

        Returns:
            Dictionary label -> unconstrained value (-).
        """
        u = {}
        for label, (name, index) in self.targets.items():
            if theta is None or name not in theta:
                u_full = self.u_reference[name]
            else:
                u_full = to_unconstrained(self.specs[name], theta[name])

            u[label] = u_full if index is None else jnp.reshape(u_full, (-1,))[index]

        return u

    def _unconstrained_full(self, u: Mapping[str, Array]) -> dict[str, Array]:
        """Unconstrained arrays of the active parameters, with their inactive entries.

        Args:
            u: optimizer variables, label -> unconstrained value.

        Returns:
            Dictionary parameter name -> unconstrained array of the parameter's shape.
        """
        full = dict(self.u_reference)
        for label, (name, index) in self.targets.items():
            if index is None:
                full[name] = jnp.reshape(u[label], self.u_reference[name].shape)
            else:
                flat = jnp.reshape(full[name], (-1,)).at[index].set(u[label])
                full[name] = jnp.reshape(flat, self.u_reference[name].shape)

        return full

    def theta(self, u: Mapping[str, Array]) -> Theta:
        """Full parameter dictionary for the optimizer variables.

        Args:
            u: optimizer variables, label -> unconstrained value.

        Returns:
            ``theta`` with physical values: the active parameters from ``u``, the others
            at their reference values.
        """
        theta = dict(self.theta_reference)
        for name, u_full in self._unconstrained_full(u).items():
            theta[name] = to_constrained(self.specs[name], u_full)

        return theta

    def physical(self, u: Mapping[str, Array]) -> dict[str, Array]:
        """Physical value of every label, for reports.

        Args:
            u: optimizer variables, label -> unconstrained value.

        Returns:
            Dictionary label -> physical value (a scalar for a label with an index).
        """
        theta = self.theta(u)
        values = {}
        for label, (name, index) in self.targets.items():
            value = theta[name]
            values[label] = value if index is None else jnp.reshape(value, (-1,))[index]

        return values

    def log_prior(self, u: Mapping[str, Array]) -> Array:
        """Sum of the Gaussian log-priors of the active variables (constants dropped).

        Args:
            u: optimizer variables, label -> unconstrained value.

        Returns:
            Scalar log-prior (-); zero if no active parameter has a ``prior_sd``.
        """
        reference = self.unconstrained()
        total = jnp.asarray(0.0)
        for label, (name, _) in self.targets.items():
            prior_sd = self.specs[name].prior_sd
            if prior_sd is None:
                continue

            deviation = (u[label] - reference[label]) / prior_sd
            total = total - 0.5 * jnp.sum(deviation * deviation)

        return total

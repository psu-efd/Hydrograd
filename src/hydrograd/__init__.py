"""Hydrograd: differentiable shallow-water and sediment-transport solvers in JAX.

- 2-D: finite-volume solver of the shallow water equations on unstructured meshes (the
  Python/JAX version of Hydrograd.jl).
- 1-D: flow and sediment transport on a network of cross-sections
  (:mod:`hydrograd.oned`), with the sediment closures of :mod:`hydrograd.sediment`.

Importing the package enables float64 in JAX (required for the adjoints).
"""

import jax

jax.config.update("jax_enable_x64", True)

__version__ = "2.0.1"
__all__ = ["__version__"]

"""Hydrograd: differentiable 2D shallow water solver (Python/JAX port of Hydrograd.jl).

Importing the package enables float64 in JAX (required for SWE adjoints).
"""

import jax

jax.config.update("jax_enable_x64", True)

__version__ = "1.0.0"
__all__ = ["__version__"]

"""Run modes driven by Hydrograd.jl's ``run_control.json``.

* :mod:`~hydrograd.apps.setup`: shared case setup (files -> mesh -> model -> IC);
* :mod:`~hydrograd.apps.forward`: forward simulation;
* :mod:`~hydrograd.apps.sensitivity`: Jacobian of the final state w.r.t. parameters;
* :mod:`~hydrograd.apps.inversion`: inversion of Manning's n, bed elevation or inflow;
* :mod:`~hydrograd.apps.ude`: training of a neural-network flow-resistance closure;
* :mod:`~hydrograd.apps.losses`, :mod:`~hydrograd.apps.params`,
  :mod:`~hydrograd.apps.optimize`: loss terms, active parameters, optimizer loop.
"""

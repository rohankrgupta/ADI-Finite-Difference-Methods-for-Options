"""ADI finite-difference solvers for American options under the Heston model.

Conventions shared by every module:

* A 2D solution is a NumPy array ``U`` of shape ``(m+1, n+1)`` with
  ``U[i, j] = u(s_i, v_j)``: axis 0 is the stock price s, axis 1 the variance v.
* Everything is float64.
* ``theta`` is the Heston long-run variance; the ADI weight is ``theta_adi``.
"""

from .params import HestonParams, OptionSpec

__all__ = ["HestonParams", "OptionSpec"]
__version__ = "0.1.0"

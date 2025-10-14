import numpy as np
from scipy.stats import gaussian_kde
from scipy.linalg import LinAlgError

# This helper class is a picklable replacement for the lambda function.
# It holds the constant bandwidth value and returns it when called.
class _PicklableScalarBandwidth:
    def __init__(self, const):
        self.const = const
    def __call__(self, *args, **kwargs):
        return self.const

class PicklableGaussianKDE(gaussian_kde):
    """
    A subclass of `scipy.stats.gaussian_kde` that is compatible with
    `multiprocessing` by avoiding the use of un-picklable lambda functions
    when a scalar bandwidth is provided.
    """
    def set_bandwidth(self, bw_method=None):
        if bw_method is None:
            pass
        elif bw_method == 'scott':
            self.covariance_factor = self.scotts_factor
        elif bw_method == 'silverman':
            self.covariance_factor = self.silverman_factor
        # This is the patched part:
        elif np.isscalar(bw_method) and not isinstance(bw_method, str):
            self._bw_method = 'use constant'
            # Instead of `lambda: bw_method`, we use our picklable class.
            self.covariance_factor = _PicklableScalarBandwidth(bw_method)
        elif callable(bw_method):
            self._bw_method = bw_method
            # This will still fail if pickled, but it is not your use case.
            self.covariance_factor = lambda: self._bw_method(self)
        else:
            msg = ("`bw_method` should be 'scott', 'silverman', a scalar "
                   "or a callable.")
            raise ValueError(msg)

        self._compute_covariance()

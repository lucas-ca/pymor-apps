import numpy as np

from pymor.core.interfaces import BasicInterface
from pymor.vectorarrays.numpy import NumpyVectorArray


class GenericStokesRBReconstructor(BasicInterface):
    """Simple reconstructor forming linear combinations with a reduced basis."""

    def __init__(self, velocity_rb, pressure_rb):
        self.velocity_rb = velocity_rb.copy()
        self.pressure_rb = pressure_rb.copy()
        # find a better way
        self.velocity_rb_size = self.velocity_rb._array.shape[0]
        self.pressure_rb_size = self.pressure_rb._array.shape[0]

    def reconstruct(self, U):
        """Reconstruct high-dimensional vector from reduced vector `U`."""
        assert isinstance(U, NumpyVectorArray)
        data = self.slice_reduced_solution(U)
        U_r = self.velocity_rb.lincomb(data['velocity'].data)
        P_r = self.pressure_rb.lincomb(data['pressure'].data)

        u = U_r._array[0]
        p = P_r._array[0]

        reconstructed_sol = np.concatenate((u, p))
        return NumpyVectorArray(reconstructed_sol)

    def restricted_to_subbasis(self, dim):
        """See :meth:`~pymor.operators.numpy.NumpyMatrixOperator.projected_to_subbasis`."""
        raise NotImplementedError
        assert dim <= len(self.RB)
        return GenericStokesRBReconstructor(self.RB.copy(ind=list(range(dim))))

    def slice_reduced_solution(self, U):
        array = U._array[0]
        u = array[0:self.velocity_rb_size]
        p = array[self.velocity_rb_size:]

        u2 = NumpyVectorArray(u)
        p2 = NumpyVectorArray(p)

        return {'velocity': u2, 'pressure': p2}
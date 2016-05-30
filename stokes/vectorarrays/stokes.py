from __future__ import absolute_import, division, print_function

from pymor.grids.tria import TriaGrid
from pymor.vectorarrays.interfaces import VectorArrayInterface
from pymor.vectorarrays.numpy import NumpyVectorArray


class StokesSolution:

    def __init__(self, grid, U):
        #assert isinstance(grid, TriaGrid)

        num_p1_knots = grid.size(grid.dim)
        num_p2_knots = grid.size(grid.dim) + grid.size(grid.dim - 1)

        # one vector
        if isinstance(U, VectorArrayInterface):
            assert U._array.shape in ((1, 3*num_p1_knots), (1, 2*num_p2_knots + num_p1_knots))
            # P1P1
            if U._array.shape == (1, 3*num_p1_knots):
                u = NumpyVectorArray(U._array[:, 0:num_p1_knots])
                v = NumpyVectorArray(U._array[:, num_p1_knots:2*num_p1_knots])
                p = NumpyVectorArray(U._array[:, 2*num_p1_knots:3*num_p1_knots])
                self.type = 'P1P1'
            # P2P1
            elif  U._array.shape == (1, 2*num_p2_knots + num_p1_knots):
                u = NumpyVectorArray(U._array[:, 0:num_p2_knots])
                v = NumpyVectorArray(U._array[:, num_p2_knots:2*num_p2_knots])
                p = NumpyVectorArray(U._array[:, 2*num_p2_knots:])
                self.type = 'P2P1'
            else:
                raise ValueError
        # multiple vectors
        elif isinstance(U, tuple):
            assert all(u._array.shape in ((1, 3*num_p1_knots), (1, 2*num_p2_knots + num_p1_knots)) for u in U)
            # P1P1
            if all(u._array.shape == (1, 3*num_p1_knots) for u in U):
                u = tuple([NumpyVectorArray(u0._array[:, 0:num_p1_knots]) for u0 in U])
                v = tuple([NumpyVectorArray(u0._array[:, num_p1_knots:2*num_p1_knots]) for u0 in U])
                p = tuple([NumpyVectorArray(u0._array[:, 2*num_p1_knots:]) for u0 in U])
                self.type = 'P1P1'
            # case P2P1
            elif all(u._array.shape == (1, 2*num_p2_knots + num_p1_knots) for u in U):
                u = tuple([NumpyVectorArray(u0._array[:, 0:num_p2_knots]) for u0 in U])
                v = tuple([NumpyVectorArray(u0._array[:, num_p2_knots:2*num_p2_knots]) for u0 in U])
                p = tuple([NumpyVectorArray(u0._array[:, 2*num_p2_knots:]) for u0 in U])
                self.type = 'P2P1'
            else:
                raise ValueError
        else:
            raise ValueError

        self.u = u
        self.v = v
        self.p = p
        self.solution = {'u': u, 'v': v, 'p': p, 'type': self.type}

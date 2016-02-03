# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.grids.interfaces import AffineGridInterface
from pymor.grids.referenceelements import triangle
#from matplotlib.pyplot import grid

from pymor.grids.tria import TriaGrid


class AffineTransformedGrid(TriaGrid):

    #reference_element = triangle

    def __init__(self, grid, transformation):

        assert isinstance(transformation, np.ndarray)
        assert transformation.shape == (2,2)

        num_i = grid.num_intervals
        domain = grid.domain
        ibt = grid.identify_bottom_top
        ilr = grid.identify_left_right
        super(AffineTransformedGrid, self).__init__(num_i, domain, ilr, ibt)

        self.t = transformation
        self.grid = grid
        self.dim = grid.dim
        self.dim_outer = grid.dim_outer

        self.reference_element = grid.reference_element
        #self.lock()


    #def __str__(self):
    #    return self.grid.__str__()

    #def size(self, codim=0):
    #    return self.grid.size(codim)

    #def subentities(self, codim=0, subentity_codim=None):
    #    return self.grid.subentities(codim, subentity_codim)

    def embeddings(self, codim=0):
        A, B = self.grid.embeddings(codim)
        B_t = np.einsum('ij, ej->ei', self.t, B)
        if codim == 2:
            A_t = A
        else:
            A_t = np.einsum('ij, ejk->eik', self.t, A)
        #embed0 = np.einsum('ij,ejk->eik',self.t,embed[0])
        #embed1 = np.einsum('ij,ej->ei',self.t,embed[1])
        return (A_t, B_t)

    def centers(self, codim):
        C = self.grid.centers(codim)
        C_t = np.einsum('ij,ej->ei', self.t, C)
        return C_t

    def calculate_bounding_box(self):
        d = self.domain
        ll = d[0,:]
        lr = np.array([d[1, 0], d[0, 1]])
        ul = np.array([d[0, 0], d[1, 1]])
        ur = d[1, :]

        box = np.array([ll, lr, ul, ur])

        box_t = np.einsum('ij,aj->ai', self.t, box)

        max = box_t.max(axis=0)
        min = box_t.min(axis=0)

        return np.array([min, max])


    def visualize(self, U, codim=2, **kwargs):
        """Visualize scalar data associated to the grid as a patch plot.
        Parameters
        ----------
        U
            |VectorArray| of the data to visualize. If `len(U) > 1`, the data is visualized
            as a time series of plots. Alternatively, a tuple of |VectorArrays| can be
            provided, in which case a subplot is created for each entry of the tuple. The
            lengths of all arrays have to agree.
        codim
            The codimension of the entities the data in `U` is attached to (either 0 or 2).
        kwargs
            See :func:`~pymor.gui.qt.visualize_patch`
        """
        #mu = kwargs.pop('mu', None)
        from pymor.gui.qt import visualize_patch
        from pymor.vectorarrays.numpy import NumpyVectorArray
        if not isinstance(U, NumpyVectorArray):
            U = NumpyVectorArray(U, copy=False)
        #bounding_box = kwargs.pop('bounding_box', self.domain*2)
        bounding_box = self.calculate_bounding_box()
        visualize_patch(self, U, codim=codim, bounding_box=bounding_box, **kwargs)

    #@staticmethod
    #def test_instances():
    #    pass



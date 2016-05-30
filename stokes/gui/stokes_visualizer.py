from pymor.core.interfaces import BasicInterface
from pymor.vectorarrays.interfaces import VectorArrayInterface
from pymor.vectorarrays.numpy import NumpyVectorArray
from pymor.grids.referenceelements import triangle, square
from pymor.tools.vtkio import write_vtk
from pymor.gui.qt import visualize_patch
from stokes.gui.qt import visualize_patch as visualize_patch_trafo
from pymor.parameters.base import Parameter

from stokes.vectorarrays.stokes import StokesSolution
from stokes_alt.grids.transformed_tria import AffineTransformedGrid
from stokes.grids.affine_transformed_tria import AffineTransformedTriaGrid

import numpy as np
from matplotlib import pyplot as plt

class StokesVisualizer(BasicInterface):
    """Visualize scalar data associated to a two-dimensional |Grid| as a patch plot.
    The grid's |ReferenceElement| must be the triangle or square. The data can either
    be attached to the faces or vertices of the grid.
    Parameters
    ----------
    grid
        The underlying |Grid|.
    bounding_box
        A bounding box in which the grid is contained.
    codim
        The codimension of the entities the data in `U` is attached to (either 0 or 2).
    backend
        Plot backend to use ('gl' or 'matplotlib').
    block
        If `True` block execution until the plot window is closed.
    plot_type
        Different options to plot velocity.
            0: Separate plots for u_1 and u_2
            1: Quiver plot
            2: Streamline plot
    resolution:
        Number of arrows in quiver plot.
    mu:
        Parameter for which to plot the solution. Must be 'transformation'.
    """

    def __init__(self, grid, bounding_box=([0, 0], [1, 1]), codim=2, backend=None, block=False, plot_type=0,
                 resolution=None):
        assert grid.reference_element in (triangle,)
        assert grid.dim_outer == 2
        assert codim in (0, 2)
        self.grid = grid
        self.bounding_box = bounding_box
        self.codim = codim
        self.backend = backend
        self.block = block
        self.plot_type = plot_type
        self.resolution = resolution

    def visualize_quiver(self, grid, u, v, title, solution_type=None, mu=None):
        assert isinstance(u, VectorArrayInterface)
        assert isinstance(v, VectorArrayInterface)
        assert solution_type is not None
        assert solution_type == 'P1P1' or solution_type == 'P2P1'

        #(ni_x, ni_y) = grid.num_intervals
        #dx = ni_x/self.resolution
        #dy = ni_y/self.resolution

        # plot only on P1 knots
        np1 = grid.size(2)

        if mu is None:
            X = grid.centers(2)[..., 0]
            Y = grid.centers(2)[..., 1]
            if solution_type == 'P1P1':
                np1 = grid.size(2)
                U = u._array[0, 0:np1]
                V = v._array[0, 0:np1]
            else:
                np1 = grid.size(2)
                np2 = grid.size(2) + grid.size(1)
                U = u._array[0, 0:np1]
                V = v._array[0, 0:np1]
        else:
            #raise NotImplementedError
            t = mu['transformation']
            # transform nodes
            xy_trans = np.einsum('ij,aj->ai', t, grid.centers(2))
            X = xy_trans[..., 0]
            Y = xy_trans[..., 1]
            # Piola transformation
            uv_trans = 1./np.abs(np.linalg.det(t)) * np.einsum('ij,aj->ai', t,
                                                               np.vstack((u._array[0, :], v._array[0, :])).T)
            #uv_trans = 1./np.abs(np.linalg.det(t)) * uv_trans
            U = uv_trans[:, 0]
            V = uv_trans[:, 1]

        # resolution
        #X = X[0::dx]
        #Y = Y[0::dy]
        #U = U[0::dx]
        #V = V[0::dy]

        #assert len(X) == U.shape[1]
        #assert len(Y) == V.shape[1]

        plt.figure()
        plt.title(title)
        plt.quiver(X, Y, U, V)
    """
    def visualize_quiver_transformed(self, grid, u, v, mu, title):
        assert isinstance(u, VectorArrayInterface)
        assert isinstance(v, VectorArrayInterface)

        assert isinstance(mu, Parameter)

        t = mu['transformation']

        (ni_x, ni_y) = grid.num_intervals
        dx = ni_x/self.resolution
        dy = ni_y/self.resolution

        X = grid.centers(2, mu)[..., 0]
        Y = grid.centers(2, mu)[..., 1]
        U = u._array[0, 0:grid.size(grid.dim)]
        V = v._array[0, 0:grid.size(grid.dim)]

        # Piola transformation
        #uv_trans = 1./np.abs(np.linalg.det(t)) * np.einsum('ij,aj->ai', t,
        #                                                   np.vstack((u._array[0, :], v._array[0, :])).T)
        #uv_trans = 1./np.abs(np.linalg.det(t)) * uv_trans
        #U = uv_trans[:, 0]
        #V = uv_trans[:, 1]

        # resolution
        X = X[0::dx]
        Y = Y[0::dy]
        U = U[0::dx]
        V = V[0::dy]

        #assert len(X) == U.shape[1]
        #assert len(Y) == V.shape[1]

        plt.title(title)
        plt.quiver(X, Y, U, V)
    """

    def visualize_streamplot(self, grid, u, v, mu, title, linewidth):
        assert isinstance(u, VectorArrayInterface)
        assert isinstance(v, VectorArrayInterface)

        (ni_x, ni_y) = grid.num_intervals
        dx = ni_x/self.resolution
        dy = ni_y/self.resolution
        X = grid.embeddings(2)[1][0:ni_x+1,:][..., 0]
        Y = grid.embeddings(2)[1][0::ni_x+1,:][0:ni_y+1][..., 1]
        U = u._array[0,0:(ni_x+1)*(ni_y+1)].reshape((ni_x+1, ni_y+1))
        V = v._array[0,0:(ni_x+1)*(ni_y+1)].reshape((ni_x+1, ni_y+1))

        #assert len(X) == U.shape[1]
        #assert len(Y) == V.shape[1]

        plt.title(title)
        plt.streamplot(X, Y, U, V)

    def visualize(self, U, discretization, mu=None, title=None, legend=None, separate_colorbars=False,
                  rescale_colorbars=False, block=None, filename=None, columns=2):
        """Visualize the provided data.
        Parameters
        ----------
        U
            |VectorArray| of the data to visualize. If `len(U) > 1`, the data is visualized
            as a time series of plots. Alternatively, a tuple of |VectorArrays| can be
            provided, in which case a subplot is created for each entry of the tuple. The
            lengths of all arrays have to agree.
        discretization
            Filled in :meth:`pymor.discretizations.DiscretizationBase.visualize` (ignored).
        title
            Title of the plot.
        legend
            Description of the data that is plotted. Most useful if `U` is a tuple in which
            case `legend` has to be a tuple of strings of the same length.
        separate_colorbars
            If `True`, use separate colorbars for each subplot.
        rescale_colorbars
            If `True`, rescale colorbars to data in each frame.
        block
            If `True`, block execution until the plot window is closed. If `None`, use the
            default provided during instantiation.
        filename
            If specified, write the data to a VTK-file using
            :func:`pymor.tools.vtkio.write_vtk` instead of displaying it.
        columns
            The number of columns in the visualizer GUI in case multiple plots are displayed
            at the same time.
        """
        assert isinstance(U, VectorArrayInterface) and hasattr(U, 'data') \
            or (isinstance(U, tuple) and all(isinstance(u, VectorArrayInterface) and hasattr(u, 'data') for u in U)
                and all(len(u) == len(U[0]) for u in U))

        U2 = StokesSolution(self.grid, U)

        num_knots_p1 = self.grid.size(self.grid.dim)
        num_knots_p2 = self.grid.size(self.grid.dim) + self.grid.size(self.grid.dim - 1)
        """
        if isinstance(U, VectorArrayInterface):
            assert U._array.shape in ((1, 3*num_knots_p1), (1, 2*num_knots_p2 + num_knots_p1))
            if U._array.shape == (1, 3*num_knots_p1):
                u = NumpyVectorArray(U._array[:, 0:num_knots_p1])
                v = NumpyVectorArray(U._array[:, num_knots_p1:2*num_knots_p1])
                p = NumpyVectorArray(U._array[:, 2*num_knots_p1:3*num_knots_p1])
            elif U._array.shape == (1, 2*num_knots_p2 + num_knots_p1):
                u = NumpyVectorArray(U._array[:, 0:num_knots_p1])
                v = NumpyVectorArray(U._array[:, num_knots_p2:num_knots_p2+num_knots_p1])
                p = NumpyVectorArray(U._array[:, 2*num_knots_p2:])

        elif isinstance(U, tuple):
            assert all(u._array.shape in ((1, 3*num_knots_p1), (1, 2*num_knots_p2 + num_knots_p1)) for u in U)
            # case P1P1
            if all(u._array.shape == (1, 3*num_knots_p1) for u in U):
                u = tuple([NumpyVectorArray(u0._array[:, 0:num_knots_p1]) for u0 in U])
                v = tuple([NumpyVectorArray(u0._array[:, num_knots_p1:2*num_knots_p1]) for u0 in U])
                p = tuple([NumpyVectorArray(u0._array[:, 2*num_knots_p1:]) for u0 in U])
            # case P2P1
            elif all(u._array.shape == (1, 2*num_knots_p2 + num_knots_p1) for u in U):
                u = tuple([NumpyVectorArray(u0._array[:, 0:num_knots_p1]) for u0 in U])
                v = tuple([NumpyVectorArray(u0._array[:, num_knots_p2:num_knots_p2+num_knots_p1]) for u0 in U])
                p = tuple([NumpyVectorArray(u0._array[:, 2*num_knots_p2:]) for u0 in U])
        """
        u = U2.u
        v = U2.v
        p = U2.p
        t = U2.type

        if filename:
            raise NotImplementedError

        else:
            block = self.block if block is None else block
            block = False
            if mu is None:
                if self.plot_type == 0:  # separate plot for u1, u2 and p
                    visualize_patch(self.grid, u, bounding_box=self.bounding_box, codim=self.codim, title="{} u_1".format(title),
                                    legend=legend, separate_colorbars=separate_colorbars, rescale_colorbars=rescale_colorbars,
                                    backend=self.backend, block=block, columns=columns)
                    visualize_patch(self.grid, v, bounding_box=self.bounding_box, codim=self.codim, title="{} u_2".format(title),
                                    legend=legend, separate_colorbars=separate_colorbars, rescale_colorbars=rescale_colorbars,
                                    backend=self.backend, block=block, columns=columns)
                    visualize_patch(self.grid, p, bounding_box=self.bounding_box, codim=self.codim, title="{} p".format(title),
                                    legend=legend, separate_colorbars=separate_colorbars, rescale_colorbars=rescale_colorbars,
                                    backend=self.backend, block=block, columns=columns)
                elif self.plot_type == 1:  # quiver plot for (u1,u2) and separate plot for p
                    self.visualize_quiver(self.grid, u, v, mu=mu, title="{} $u$".format(title), solution_type=t)
                    visualize_patch(self.grid, p, bounding_box=self.bounding_box, codim=self.codim, title="{} p".format(title),
                                    legend=legend, separate_colorbars=separate_colorbars, rescale_colorbars=rescale_colorbars,
                                    backend=self.backend, block=block, columns=columns)
                elif self.plot_type == 2:  # streamline plot for (u1,u2) and separate plot for p
                    self.visualize_streamplot(self.grid, u, v, mu=mu, title="{} $u$".format(title), linewidth=None)
                    visualize_patch(self.grid, p, bounding_box=self.bounding_box, codim=self.codim, title="{} p".format(title),
                                    legend=legend, separate_colorbars=separate_colorbars, rescale_colorbars=rescale_colorbars,
                                    backend=self.backend, block=block, columns=columns)
            else:
                bounding_box = self.grid.bounding_box(mu)
                if self.plot_type == 0:  # separate plot for u1, u2 and p
                    # ugly
                    u = NumpyVectorArray(u._array[0:num_knots_p1])
                    v = NumpyVectorArray(v._array[0:num_knots_p1])
                    visualize_patch_trafo(self.grid, u, mu, bounding_box=bounding_box, codim=self.codim,
                                          title="{} u for mu={}".format(title, mu), legend=legend,
                                          separate_colorbars=separate_colorbars, rescale_colorbars=rescale_colorbars,
                                          backend=self.backend, block=block, columns=columns)
                    visualize_patch_trafo(self.grid, v, mu, bounding_box=bounding_box, codim=self.codim,
                                          title="{} v for mu={}".format(title, mu), legend=legend,
                                          separate_colorbars=separate_colorbars, rescale_colorbars=rescale_colorbars,
                                          backend=self.backend, block=block, columns=columns)
                    visualize_patch_trafo(self.grid, p, mu, bounding_box=bounding_box, codim=self.codim,
                                          title="{} p for mu={}".format(title, mu), legend=legend,
                                          separate_colorbars=separate_colorbars, rescale_colorbars=rescale_colorbars,
                                          backend=self.backend, block=block, columns=columns)
                elif self.plot_type == 1: # quiver plot vor (u,v)
                    # ugly
                    u = NumpyVectorArray(u._array[0:num_knots_p1])
                    v = NumpyVectorArray(v._array[0:num_knots_p1])

                    # quiver plot
                    #self.visualize_quiver_transformed(self.grid, u, v, mu, title)
                    # p
                    visualize_patch_trafo(self.grid, p, mu, bounding_box=bounding_box, codim=self.codim,
                                          title="{} p for mu={}".format(title, mu), legend=legend,
                                          separate_colorbars=separate_colorbars, rescale_colorbars=rescale_colorbars,
                                          backend=self.backend, block=block, columns=columns)

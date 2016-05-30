from __future__ import absolute_import, division, print_function

from pymor.grids.tria import TriaGrid
from pymor.parameters.base import Parameter

from stokes.functions.affine_transformation import AffineTransformation


class AffineTransformedTriaGrid(TriaGrid):

    def __init__(self, grid, transformation):

        # grid must be a tria grid
        assert isinstance(grid, TriaGrid)

        # transformation must be affine
        assert isinstance(transformation, AffineTransformation)

        super(AffineTransformedTriaGrid, self).__init__(num_intervals=grid.num_intervals,
                                                        domain=grid.domain,
                                                        identify_left_right=grid.identify_left_right,
                                                        identify_bottom_top=grid.identify_bottom_top)

        self.grid = grid
        self.transformation = transformation

    def centers(self, codim, mu=None):
        assert mu is None or isinstance(mu, Parameter)

        centers = self.grid.centers(codim)
        if mu is None:
            return centers
        else:
            centers_transformed = self.transformation.apply(centers, mu)
            return centers_transformed

    def embeddings(self, codim=0, mu=None):
        assert mu is None or isinstance(mu, Parameter)

        a, b = self.grid.embeddings(codim)
        if mu is None:
            return a, b
        else:
            if codim == 2:
                a_transformed = a
            else:
                a_transformed = self.transformation.apply(a, mu)
            b_transformed = self.transformation.apply(b, mu)
            return a_transformed, b_transformed

    def bounding_box(self, mu=None):
        assert mu is None or isinstance(mu, Parameter) or isinstance(mu, tuple) and \
                                                          all(isinstance(m, Parameter) for m in mu)

        if mu is None:
            return self.domain
        else:
            return self.transformation.bounding_box(self.domain, mu)

    def visualize(self, U, codim=2, **kwargs):
        #
        from stokes.gui.qt import visualize_patch

        mu = kwargs.pop("mu", None)

        bounding_box = self.bounding_box(mu)

        visualize_patch(grid=self,
                        U=U,
                        mu=mu,
                        bounding_box=bounding_box,
                        codim=codim,
                        **kwargs)

from __future__ import absolute_import, division, print_function

from pymor.domaindescriptions.basic import RectDomain
from pymor.domaindescriptions.interfaces import DomainDescriptionInterface
from pymor.functions.basic import FunctionBase
from pymor.parameters.base import Parameter

from stokes.domaindescriptions.polygonal import PolygonalDomain

import numpy as np

from stokes.functions.affine_transformation import AffineTransformation


class PiecewiseAffineTransformation(FunctionBase):

    def __init__(self, domain, transformations, mapping):
        """

        Parameters
        ----------
        domain The domain.
        transformations A list of transformations. Every subdomain of the domain has its own transformation.
        mapping A function that maps the parameter mu to an affine transformation on every subdomain.

        Returns
        -------

        """
        assert isinstance(domain, DomainDescriptionInterface)
        assert isinstance(transformations, list)
        assert len(transformations) > 0
        assert all(isinstance(t, AffineTransformation) for t in transformations)

        if isinstance(domain, RectDomain):
            # one transformation for the whole rectangular domain
            assert len(transformations) == 1
        elif isinstance(domain, PolygonalDomain):
            if len(domain.subdomains) > 0:
                # every subdomain has its own transformation
                if len(transformations) > 1:
                    assert len(transformations) == len(domain.subdomains)
                else:
                    # one transformation for the whole domain
                    assert len(transformations) == 1
        else:
            raise NotImplementedError

        self.domain = domain
        self.transformations = transformations
        self.mapping = mapping

    def evaluate(self, x, mu=None):
        assert isinstance(mu, Parameter)
        assert len(mu) == len(self.transformations)

        # TODO: berechne die einzelnen Transformationsparameter

        if len(self.transformations) == 1:
            return self.transformations[0].evaluate(x, mu)
        else:
            if isinstance(self.domain, PolygonalDomain):
                # transformations
                transformations = np.array(self.transformations)

                # get sub rectangles from domain
                points = np.array(self.domain.points)
                subdomains = [np.array(sd) for sd in self.domain.subdomains]
                subrectangles = [points[sd] for sd in subdomains]

                # get lower left and upper right corners
                lower_left = [np.min(sr, axis=0) for sr in subrectangles]
                upper_right = [np.max(sr, axis=0) for sr in subrectangles]

                # TODO: ab hier wird x benutzt
                # get boolean array indicating whether a point is in rectangle i
                boolean_inside_masks = [np.all(np.logical_and(lower_left[i] <= x, x <= upper_right[i]), axis=1)
                                for i in range(len(subrectangles))]

                # get integer array indicating in which rectangle a point is (starting with 1)
                integer_inside_masks = [ism*(i+1) for i, ism in enumerate(boolean_inside_masks)]

                # TODO: nur den index aus dem ersten array nutzen; DONE
                subdomain_mask = np.zeros_like(integer_inside_masks[0])
                for i in range(len(subrectangles)):
                    subdomain_mask += np.where((subdomain_mask == 0) * boolean_inside_masks[i], integer_inside_masks[i],
                                               0)

                # ziehe 1 ab wegen 1 indizierung
                subdomain_mask -= 1

                res = np.zeros(shape=x.shape[:-1] + transformations[0].shape_range)

                for i in range(len(subrectangles)):
                    m = subdomain_mask == i
                    res[m] = transformations[i].evaluate(x[m], mu[i])

                # return
                return res



if __name__ == '__main__':

    from pymor.domaindescriptions.basic import RectDomain
    from pymor.domaindescriptions.boundarytypes import BoundaryType

    from stokes.domaindescriptions.polygonal import PolygonalDomain

    d = PolygonalDomain(points=[[0, 0], [1, 0], [2, 0], [2, 1], [1, 1], [0, 1]],
                        boundary_types={BoundaryType('dirichlet'): [0, 1, 2, 3, 4, 5]},
                        holes=[],
                        inner_edges=[[1, 4]],
                        subdomains=[[0, 1, 4, 5], [1, 2, 3, 4]])

    A = np.array([[True, False, True], [False, True, False]])
    f1 = np.array([[1, 0], [0, 0]])

    t0 = AffineTransformation('transformation', 0.1, 1.0, None, 't0')
    t1 = AffineTransformation('transformation', 0.1, 1.0, None, 't1')



    ll = d.lower_left
    ur = d.upper_right

    z=0
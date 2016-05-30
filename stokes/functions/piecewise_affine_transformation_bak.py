from __future__ import absolute_import, division, print_function

from pymor.domaindescriptions.basic import RectDomain
from pymor.domaindescriptions.interfaces import DomainDescriptionInterface
from pymor.functions.basic import FunctionBase
from pymor.parameters.base import Parameter

from stokes.domaindescriptions.polygonal import PolygonalDomain

import numpy as np

from stokes.functions.affine_transformation import AffineTransformation







def apply_affine_transformation(x, mu):
    pass


class PiecewiseAffineTransformation(FunctionBase):

    def __init__(self, domain, mapping):
        """

        Parameters
        ----------
        domain The domain.
        mapping A function that maps the parameter mu to an affine transformation on every subdomain.

        Returns
        -------

        """
        assert isinstance(domain, PolygonalDomain)
        assert len(domain.subdomains) > 0

        self.domain = domain
        self.mapping = mapping

    def characterisitc_function(self, x, index):
        """
        A function that returns a boolean array of shape (x.shape[:-1],) which indicates whether point x is in a
        specific subdoman.
        Parameters
        ----------
        x The points to be checked.
        index The index of the subdomain.

        Returns
        -------

        """
        # get sub rectangles from domain
        points = np.array(self.domain.points)
        subdomains = [np.array(sd) for sd in self.domain.subdomains]
        subrectangles = [points[sd] for sd in subdomains]

        # get lower left and upper right corners
        lower_left = [np.min(sr, axis=0) for sr in subrectangles]
        upper_right = [np.max(sr, axis=0) for sr in subrectangles]

        # get boolean array indicating whether a point is in rectangle i
        boolean_inside_masks = [np.all(np.logical_and(lower_left[i] <= x, x <= upper_right[i]), axis=-1)
                                for i in range(len(subrectangles))]

        # get integer array indicating in which rectangle a point is (starting with 1)
        integer_inside_masks = [ism*(i+1) for i, ism in enumerate(boolean_inside_masks)]

        subdomain_mask = np.zeros_like(integer_inside_masks[0])
        for i in range(len(subrectangles)):
            subdomain_mask += np.where((subdomain_mask == 0) * boolean_inside_masks[i], integer_inside_masks[i], 0)

        # ziehe 1 ab wegen 1 indizierung
        subdomain_mask -= 1

        return subdomain_mask == index

    def parse_subdomains(self, x):
        """

        Parameters
        ----------
        x The points to be parsed.

        Returns
        -------
        subdomain_mask Integer array, which indicates to which subdomian a point belongs.
        l Number of subdomains.

        """

        # get sub rectangles from domain
        points = np.array(self.domain.points)
        subdomains = [np.array(sd) for sd in self.domain.subdomains]
        subrectangles = [points[sd] for sd in subdomains]

        # get lower left and upper right corners
        lower_left = [np.min(sr, axis=0) for sr in subrectangles]
        upper_right = [np.max(sr, axis=0) for sr in subrectangles]

        # get boolean array indicating whether a point is in rectangle i
        boolean_inside_masks = [np.all(np.logical_and(lower_left[i] <= x, x <= upper_right[i]), axis=1)
                                for i in range(len(subrectangles))]

        # get integer array indicating in which rectangle a point is (starting with 1)
        integer_inside_masks = [ism*(i+1) for i, ism in enumerate(boolean_inside_masks)]

        # TODO: nur den index aus dem ersten array nutzen; DONE
        subdomain_mask = np.zeros_like(integer_inside_masks[0])
        for i in range(len(subrectangles)):
            subdomain_mask += np.where((subdomain_mask == 0) * boolean_inside_masks[i], integer_inside_masks[i], 0)

        # ziehe 1 ab wegen 1 indizierung
        subdomain_mask -= 1

        return subdomain_mask, len(subrectangles)

    def evaluate(self, x, mu=None):
        assert isinstance(mu, Parameter)

        parameter_list = self.mapping(mu)
        """
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
            subdomain_mask += np.where((subdomain_mask == 0) * boolean_inside_masks[i], integer_inside_masks[i], 0)

        # ziehe 1 ab wegen 1 indizierung
        subdomain_mask -= 1
        """

        subdomain_mask, num_subdomains = self.parse_subdomains(x)

        res = np.zeros(shape=x.shape[:-1] + (2, 2))

        for i in range(num_subdomains):
            m = subdomain_mask == i
            res[m] = self._apply(x[m], mu[i])

        # return
        return res

    def _apply(self, x, mu=None):
        # parse mu
        mu = self.parse_parameter(mu)
        a = mu['matrix']
        b = mu['vector']

        assert x.shape[-1] == 2

        # res = np.dot(x, mu)
        res = np.einsum('ij,ej->ei', a, x)
        res += b

        return res

    def _jacobian(self, mu=None):
        if mu is None:
            raise ValueError

        assert isinstance(mu, Parameter)

        # parse mu
        mu = self.parse_parameter(mu)
        a = mu['matrix']
        # b = mu['vector']

        return a

    def _jacobian_inverse(self, mu=None):
        if mu is None:
            raise ValueError

        assert isinstance(mu, Parameter)

        # parse mu
        mu = self.parse_parameter(mu)
        a = mu['matrix']
        # b = mu['vector']

        return np.linalg.inv(a)

    def _jacobian_determinant(self, mu=None):
        if mu is None:
            raise ValueError

        assert isinstance(mu, Parameter)

        # parse mu
        mu = self.parse_parameter(mu)
        a = mu['matrix']
        # b = mu['vector']

        return np.linalg.det(a)

    def _diffusion_functional(self, mu, index):
        """
        Returns an entry of the diffusion transformation matrix.

        Parameters
        ----------
        mu parameter for which to assemble the matrix
        index the entry to be returned

        Returns
        -------

        """

        # index must be between (0, 0) and (1, 1)
        assert isinstance(index, tuple)
        assert (0, 0) <= index
        assert (1, 1) >= index

        # mu must be a parameter
        assert isinstance(mu, Parameter)

        # parse parameter
        # mu = self.parse_parameter(mu)
        # a = mu['matrix']
        # b = mu['vector']

        # jacobian_inverse
        jac_inv = self._jacobian_inverse(mu)
        # jacobian inverse transposed
        jac_inv_transposed = jac_inv.swapaxes(-1, -2)
        # jacobian determinant
        det = self._jacobian_determinant(mu)
        det = np.abs(det)

        # transformation matrix
        # res = np.einsum('eij,ejk,e->eik', jac_inv, jac_inv_transposed, det)
        res = det * np.dot(jac_inv, jac_inv_transposed)

        return res[index]

    def _advection_functional(self, mu, index):
        """
        Returns an entry of the advection transformation matrix.

        Parameters
        ----------
        mu parameter for which to assemble the matrix
        index the entry to be returned

        Returns
        -------

        """

        # index must be between (0, 0) and (1, 1)
        assert isinstance(index, tuple)
        assert (0, 0) <= index
        assert (1, 1) >= index

        # mu must be a parameter
        assert isinstance(mu, Parameter)

        # parse parameter
        # mu = self.parse_parameter(mu)
        # a = mu['matrix']
        # b = mu['vector']

        # jacobian_inverse
        jac_inv = self._jacobian_inverse(mu)
        # jacobian inverse transposed
        jac_inv_transposed = jac_inv.swapaxes(-1, -2)
        # jacobian determinant
        det = self._jacobian_determinant(mu)
        det = np.abs(det)

        # transformation matrix
        # res = np.einsum('eij,ejk,e->eik', jac_inv, jac_inv_transposed, det)
        res = det * jac_inv_transposed

        return res[index]



if __name__ == '__main__':

    from pymor.domaindescriptions.basic import RectDomain
    from pymor.domaindescriptions.boundarytypes import BoundaryType

    from stokes.domaindescriptions.polygonal import PolygonalDomain

    d = PolygonalDomain(points=[[0, 0], [1, 0], [2, 0], [2, 1], [1, 1], [0, 1]],
                        boundary_types={BoundaryType('dirichlet'): [0, 1, 2, 3, 4, 5]},
                        holes=[],
                        inner_edges=[[1, 4]],
                        subdomains=[[0, 1, 4, 5], [1, 2, 3, 4]])

    t0 = AffineTransformation('transformation', 0.1, 1.0, None, 't0')
    t1 = AffineTransformation('transformation', 0.1, 1.0, None, 't1')

    t = PiecewiseAffineTransformation(d, None)
    b = np.array([[True, False, False], [False, True, True]])
    x = np.array([[0, 0], [0.5, 0.5], [1, 0], [2, 1], [1.0000000001, 0]])

    b0 = t.characterisitc_function(x, 0)
    f0 = np.outer(b0, np.array([[1.0, 0.0], [0.0, 0.0]])).reshape(x.shape[:-1] + (2, 2))


    z=0
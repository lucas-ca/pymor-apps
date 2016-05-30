from __future__ import absolute_import, division, print_function

from functools import partial
from itertools import product

from pymor.domaindescriptions.basic import RectDomain
from pymor.domaindescriptions.interfaces import DomainDescriptionInterface
from pymor.functions.basic import FunctionBase, GenericFunction
from pymor.parameters.base import Parameter
from pymor.parameters.functionals import GenericParameterFunctional

from stokes.domaindescriptions.polygonal import PolygonalDomain

import numpy as np

from stokes.functions.affine_transformation import AffineTransformation


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
        self.num_subdomains = len(domain.subdomains)

    def diffusion_functions(self):
        return self._functions()

    def advection_functions(self):
        return self._functions()

    def rhs_functions(self):
        return self._functions()

    def dirichlet_data_functions(self):
        return self._functions()

    def diffusion_functionals(self, mu=None):
        functionals = [GenericParameterFunctional(mapping=partial(self._diffusion_functional, subdomain_index=si,
                                                                  index=(i, j)),
                                                  parameter_type={'transformation': (2, 2)}, # TODO type richtig???
                                                  name='Transformation')
                       for si in range(self.num_subdomains) for i, j in product(xrange(2), xrange(2))]

        return functionals

    def advection_functionals(self, mu=None):
        functionals = [GenericParameterFunctional(mapping=partial(self._advection_functional, subdomain_index=si,
                                                                  index=(i, j)),
                                                  parameter_type={'transformation': (2, 2)}, # TODO type richtig???
                                                  name='Transformation')
                       for si in range(self.num_subdomains) for i, j in product(xrange(2), xrange(2))]

        return functionals

    def rhs_functionals(self, mu=None):
        functionals = [GenericParameterFunctional(mapping=partial(self._rhs_functional, subdomain_index=si,
                                                                  index=(i, j)),
                                                  parameter_type={'transformation': (2, 2)}, # TODO type richtig???
                                                  name='Transformation')
                       for si in range(self.num_subdomains) for i, j in product(xrange(2), xrange(2))]

        return functionals

    def dirichlet_data_functionals(self, mu=None):
        functionals = [GenericParameterFunctional(mapping=partial(self._dirichlet_data_functional, subdomain_index=si,
                                                                  index=(i, j)),
                                                  parameter_type={'transformation': (2, 2)}, # TODO type richtig???
                                                  name='Transformation')
                       for si in range(self.num_subdomains) for i, j in product(xrange(2), xrange(2))]

        return functionals

    def jacobian(self, mu=None):
        if mu is None:
            raise NotImplementedError
        else:
            return mu

    def jacobian_inverse(self, mu=None):

        if mu is None:
            raise NotImplementedError
        else:
            return np.linalg.inv(mu)

    def jacobian_determinant(self, mu=None):

        if mu is None:
            raise NotImplementedError
        else:
            return np.linalg.det(mu)

    def _characterisitc_function(self, x, subdomain_index):
        """
        A function that returns a boolean array of shape (x.shape[:-1],) which indicates whether point x is in a
        specific subdoman.
        Parameters
        ----------
        x The points to be checked.
        subdomain_index The index of the subdomain.

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

        return subdomain_mask == subdomain_index

    def _function(self, x, subdomain_index, true_value):
        assert isinstance(true_value, np.ndarray)

        return np.outer(self._characterisitc_function(x, subdomain_index), true_value).\
            reshape(x.shape[:-1] + true_value.shape)

    def _functions(self):
        return [GenericFunction(mapping=partial(self._functions, subdomian_index=si, true_value=tv), dim_domain=2,
                                shape_range=(2, 2)) for si in range(self.num_subdomains) for tv in
                [np.array([[1.0, 0.0], [0.0, 0.0]]),
                 np.array([[0.0, 1.0], [0.0, 0.0]]),
                 np.array([[0.0, 0.0], [1.0, 0.0]]),
                 np.array([[0.0, 0.0], [0.0, 1.0]])]]

    def _diffusion_functional(self, mu, subdomain_index, index):
        """
        Returns an entry of the diffusion transformation matrix

        Parameters
        ----------
        mu parameter for which to assemble the matrix
        subdomain_index Index of the subdomain.
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
        # mu = mu['transformation']

        # get parameter_list
        parameter_list = self.mapping(mu)
        parameter = parameter_list[subdomain_index]
        p = self.parse_parameter(parameter)
        a = p['matrix']
        # b = p['vector']

        # jacobian_inverse
        jac_inv = self.jacobian_inverse(a)
        # jacobian inverse transposed
        jac_inv_transposed = jac_inv.swapaxes(-1, -2)
        # jacobian determinant
        det = self.jacobian_determinant(a)
        det = np.abs(det)

        # transformation matrix
        # res = np.einsum('eij,ejk,e->eik', jac_inv, jac_inv_transposed, det)
        res = det * np.dot(jac_inv, jac_inv_transposed)

        return res[index]

    def _advection_functional(self, mu, subdomain_index, index):
        """
        Returns an entry of the advection transformation matrix

        Parameters
        ----------
        mu parameter for which to assemble the matrix
        subdomain_index Index of the subdomain.
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
        # mu = mu['transformation']

        # get parameter_list
        parameter_list = self.mapping(mu)
        parameter = parameter_list[subdomain_index]
        p = self.parse_parameter(parameter)
        a = p['matrix']
        # b = p['vector']

        # jacobian_inverse
        jac_inv = self.jacobian_inverse(a)
        # jacobian inverse transposed
        jac_inv_transposed = jac_inv.swapaxes(-1, -2)
        # jacobian determinant
        det = self.jacobian_determinant(a)
        det = np.abs(det)

        # transformation matrix
        # res = np.einsum('eij,ejk,e->eik', jac_inv, jac_inv_transposed, det)
        res = det * jac_inv_transposed

        return res[index]

    def _rhs_functional(self, mu, subdomain_index, index):
        """
        Returns an entry of the rhs transformation matrix

        Parameters
        ----------
        mu parameter for which to assemble the matrix
        subdomain_index Index of the subdomain.
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
        # mu = mu['transformation']

        # get parameter_list
        parameter_list = self.mapping(mu)
        parameter = parameter_list[subdomain_index]
        p = self.parse_parameter(parameter)
        a = p['matrix']
        # b = p['vector']

        # jacobian_inverse
        jac = self.jacobian(a)
        # jacobian determinant
        det = self.jacobian_determinant(a)
        det = np.abs(det)
        det_inv = 1.0/det

        # transformation matrix
        # res = np.einsum('eij,ejk,e->eik', jac_inv, jac_inv_transposed, det)
        res = det_inv * det * jac

        return res[index]

    def _dirichlet_data_functional(self, mu, subdomain_index, index):
        """
        Returns an entry of the dirichlet data transformation matrix

        Parameters
        ----------
        mu parameter for which to assemble the matrix
        subdomain_index Index of the subdomain.
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
        # mu = mu['transformation']

        # get parameter_list
        parameter_list = self.mapping(mu)
        parameter = parameter_list[subdomain_index]
        p = self.parse_parameter(parameter)
        a = p['matrix']
        # b = p['vector']

        # jacobian_inverse
        jac = self.jacobian(a)
        # jacobian determinant
        det = self.jacobian_determinant(a)
        det = np.abs(det)
        det_inv = 1.0/det

        # transformation matrix
        # res = np.einsum('eij,ejk,e->eik', jac_inv, jac_inv_transposed, det)
        res = det_inv * det * jac

        return res[index]

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

        return subdomain_mask, len(subrectangles), lower_left, upper_right

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

        subdomain_mask, num_subdomains, ll, ur = self.parse_subdomains(x)

        res = np.zeros(shape=x.shape[:-1] + (2,))

        for i in range(num_subdomains):
            m = subdomain_mask == i
            x2 = x[m] - ll[i]
            res[m] = self._apply(x2, parameter_list[i]) + ll[i]

        # return
        return res

    def _apply(self, x, mu=None):
        # parse mu
        mu = self.parse_parameter(mu)
        a = mu['matrix']
        b = mu['translation']

        assert x.shape[-1] == 2

        # res = np.dot(x, mu)
        res = np.einsum('ij,ej->ei', a, x)
        res += b

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

    t0 = AffineTransformation('transformation', 0.1, 1.0, None, 't0')
    t1 = AffineTransformation('transformation', 0.1, 1.0, None, 't1')

    t = PiecewiseAffineTransformation(d, None)
    b = np.array([[True, False, False], [False, True, True]])
    x = np.array([[0, 0], [0.5, 0.5], [1, 0], [2, 1], [1.0000000001, 0]])

    b0 = t.characterisitc_function(x, 0)
    f0 = np.outer(b0, np.array([[1.0, 0.0], [0.0, 0.0]])).reshape(x.shape[:-1] + (2, 2))


    z=0
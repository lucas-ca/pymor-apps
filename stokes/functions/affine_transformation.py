from __future__ import absolute_import, division, print_function

from pymor.functions.basic import ConstantFunction
from pymor.functions.basic import FunctionBase
from pymor.parameters.base import Parameter
from pymor.parameters.functionals import GenericParameterFunctional
from pymor.parameters.spaces import CubicParameterSpace

from functools import partial
from itertools import product

import numpy as np


class AffineTransformation(FunctionBase):

    def __init__(self, parameter_type, mu_min=None, mu_max=None, ranges=None, parameter_mapping=None, name=None):

        assert (ranges is None and mu_min is not None and mu_max is not None) or\
               (ranges is not None and mu_min is None and mu_max is None)

        if ranges is not None:
            assert set(ranges.keys()) == set(parameter_type.keys())

        assert parameter_mapping is not None

        super(AffineTransformation, self).__init__()

        self.build_parameter_type(parameter_type, local_global=True)
        self.parameter_space = CubicParameterSpace(parameter_type, mu_min, mu_max, ranges)
        self.parameter_mapping = parameter_mapping
        self.name = name

    def evaluate(self, x, mu=None):
        return self.apply(x, mu)

    def apply(self, x, mu=None):
        # parse mu
        mu = self.parse_parameter(mu)
        mu = self.parameter_mapping(mu)
        a = mu['matrix']
        b = mu['translation']

        assert x.shape[-1] == 2

        res = np.dot(x, a)
        res2 = np.einsum('ij,ej->ei', a, x)
        res2 += b

        return res2

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

    def bounding_box(self, domain, mu=None):

        ll = domain[0, :]
        ur = domain[1, :]
        ul = np.array([domain[0, 0], domain[1, 1]])
        lr = np.array([domain[1, 0], domain[0, 1]])

        box = np.array([ll, lr, ul, ur])

        mu = (mu,) if isinstance(mu, Parameter) else mu

        transformed_boxes = []
        for m in mu:
            transformed_boxes.append(self.apply(box, m))

        tb = np.array(transformed_boxes)
        v_min = tb.min(axis=(0, 1))
        v_max = tb.max(axis=(0, 1))

        return np.array([v_min, v_max])

    def diffusion_functions(self):
        return self._functions()

    def advection_functions(self):
        return self._functions()

    def rhs_functions(self):
        return self._functions()

    def dirichlet_data_functions(self):
        return self._functions()

    #def mass_functions(self):
    #    return [ConstantFunction(np.array(1.0), dim_domain=2)]

    def _functions(self):
        f1 = ConstantFunction(np.array([[1.0, 0.0], [0.0, 0.0]]), dim_domain=2)
        f2 = ConstantFunction(np.array([[0.0, 1.0], [0.0, 0.0]]), dim_domain=2)
        f3 = ConstantFunction(np.array([[0.0, 0.0], [1.0, 0.0]]), dim_domain=2)
        f4 = ConstantFunction(np.array([[0.0, 0.0], [0.0, 1.0]]), dim_domain=2)

        return [f1, f2, f3, f4]

    def diffusion_functionals(self, mu=None):
        functionals = [GenericParameterFunctional(mapping=partial(self._diffusion_functional, index=(i, j)),
                                                  parameter_type=self.parameter_type, name='Transformation')
                       for i, j in product(xrange(2), xrange(2))]

        return functionals

    def advection_functionals(self, mu=None):
        functionals = [GenericParameterFunctional(mapping=partial(self._advection_functional, index=(i, j)),
                                                  parameter_type=self.parameter_type, name='Transformation')
                       for i, j in product(xrange(2), xrange(2))]

        return functionals

    def rhs_functionals(self, mu=None):
        functionals = [GenericParameterFunctional(mapping=partial(self._rhs_functional, index=(i, j)),
                                                  parameter_type=self.parameter_type, name='Transformation')
                       for i, j in product(xrange(2), xrange(2))]

        return functionals

    def dirichlet_data_functionals(self, mu=None):
        functionals = [GenericParameterFunctional(mapping=partial(self._dirichlet_data_functional, index=(i, j)),
                                                  parameter_type=self.parameter_type, name='Transformation')
                       for i, j in product(xrange(2), xrange(2))]

        return functionals

    #def mass_functionals(self, mu=None):
    #    functionals = [GenericParameterFunctional(mapping=self._mass_functional,
    #                                              parameter_type=self.parameter_type, name='Transformation')]

    #    return functionals

    def _diffusion_functional(self, mu, index):
        """
        Returns an entry of the diffusion transformation matrix

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
        mu = self.parse_parameter(mu)
        mu = self.parameter_mapping(mu)

        assert isinstance(mu, dict)
        assert len(mu) == 2
        assert 'matrix' in mu.keys()
        assert 'translation' in mu.keys()

        a = mu['matrix']
        b = mu['translation']

        assert isinstance(a, np.ndarray)
        assert isinstance(b, np.ndarray)
        assert a.shape == (2, 2)
        assert b.shape == (2,)

        # jacobian_inverse
        jac_inv = self.jacobian_inverse(a)
        # jacobian inverse transposed
        jac_inv_transposed = jac_inv.swapaxes(-1, -2)
        # jacobian determinant
        det = self.jacobian_determinant(a)
        det = np.abs(det)

        # transformation matrix
        res = det * np.dot(jac_inv, jac_inv_transposed)

        return res[index]

    def _advection_functional(self, mu, index):
        """
        Returns an entry of the advection transformation matrix

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
        mu = self.parse_parameter(mu)
        mu = self.parameter_mapping(mu)

        assert isinstance(mu, dict)
        assert len(mu) == 2
        assert 'matrix' in mu.keys()
        assert 'translation' in mu.keys()

        a = mu['matrix']
        b = mu['translation']

        assert isinstance(a, np.ndarray)
        assert isinstance(b, np.ndarray)
        assert a.shape == (2, 2)
        assert b.shape == (2,)

        # jacobian_inverse
        jac_inv = self.jacobian_inverse(a)
        # jacobian inverse transposed
        jac_inv_transposed = jac_inv.swapaxes(-1, -2)
        # jacobian determinant
        det = self.jacobian_determinant(a)
        det = np.abs(det)

        # transformation matrix
        res = det * jac_inv_transposed

        return res[index]

    def _rhs_functional(self, mu, index):
        """
        Returns an entry of the rhs transformation matrix

        Parameters
        ----------
        mu parameter for which to assemble the matrix
        index the entry to be returned

        Returns
        -------

        """

        # mu must be a parameter
        assert isinstance(mu, Parameter)

        # parse parameter
        mu = self.parse_parameter(mu)
        mu = self.parameter_mapping(mu)

        assert isinstance(mu, dict)
        assert len(mu) == 2
        assert 'matrix' in mu.keys()
        assert 'translation' in mu.keys()

        a = mu['matrix']
        b = mu['translation']

        assert isinstance(a, np.ndarray)
        assert isinstance(b, np.ndarray)
        assert a.shape == (2, 2)
        assert b.shape == (2,)

        # jacobian
        jac = self.jacobian(a)
        # jacobian determinant
        det = self.jacobian_determinant(a)
        det = np.abs(det)
        # jacobian determinant inverse
        det_inv = 1.0/det

        # transformation matrix
        res = det_inv * det * jac

        return res[index]

    def _dirichlet_data_functional(self, mu, index):
        """
        Returns an entry of the dirichlet data transformation matrix

        Parameters
        ----------
        mu parameter for which to assemble the matrix
        index the entry to be returned

        Returns
        -------

        """

        # mu must be a parameter
        assert isinstance(mu, Parameter)

        # parse parameter
        mu = self.parse_parameter(mu)
        mu = self.parameter_mapping(mu)

        assert isinstance(mu, dict)
        assert len(mu) == 2
        assert 'matrix' in mu.keys()
        assert 'translation' in mu.keys()

        a = mu['matrix']
        b = mu['translation']

        assert isinstance(a, np.ndarray)
        assert isinstance(b, np.ndarray)
        assert a.shape == (2, 2)
        assert b.shape == (2,)

        # jacobian_inverse
        jac = self.jacobian(a)
        # jacobian determinant
        det = self.jacobian_determinant(a)
        det = np.abs(det)
        # jacobian determinant inverse
        det_inv = 1.0/det

        # transformation matrix
        res = det_inv * jac

        return res[index]

    #def _mass_functional(self, mu):
        """
        Returns the mass transformation functional

        Parameters
        ----------
        mu parameter for which to assemble the functional

        Returns
        -------

        """

        # mu must be a parameter
     #   assert isinstance(mu, Parameter)

        # parse parameter
    #    mu = self.parse_parameter(mu)
    #    mu = self.parameter_mapping(mu)

    #    assert isinstance(mu, dict)
    #    assert len(mu) == 2
    #    assert 'matrix' in mu.keys()
    #    assert 'translation' in mu.keys()

    #    a = mu['matrix']
    #    b = mu['translation']

    #    assert isinstance(a, np.ndarray)
    #    assert isinstance(b, np.ndarray)
    #    assert a.shape == (2, 2)
    #    assert b.shape == (2,)

        # jacobian determinant
    #    det = self.jacobian_determinant(a)
    #    det = np.abs(det)
    #    res = det

    #    return res

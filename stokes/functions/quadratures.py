from __future__ import absolute_import, division, print_function

import numpy as np


def high_order_triangle_quadrature(order=None, npoints=None, quadrature_type='default'):
        assert order is not None or npoints is not None, 'must specify "order" or "npoints"'
        assert order is None or npoints is None, 'cannot specify "order" and "npoints"'
        if quadrature_type == 'default':
            if order == 1 or npoints == 1:
                quadrature_type = 'center'
            else:
                quadrature_type = 'edge_centers'

        if quadrature_type == 'edge_centers':
            assert order is None or order <= 2
            assert npoints is None or npoints == 3
            # this would work for arbitrary reference elements
            # L, A = self.subentity_embedding(1)
            # return np.array(L.dot(self.sub_reference_element().center()) + A), np.ones(3) / len(A) * self.volume
            return np.array(([0.5, 0.5], [0, 0.5], [0.5, 0])), np.ones(3) / 3 * 0.5
        elif quadrature_type == 'interior':
            assert order is None or order <= 2
            assert npoints is None or npoints == 3
            # this would work for arbitrary reference elements
            # L, A = self.subentity_embedding(1)
            # return np.array(L.dot(self.sub_reference_element().center()) + A), np.ones(3) / len(A) * self.volume
            return np.array(([1./6., 1./6.], [2./3., 1./6.], [1./6., 2./3.])), np.ones(3) / 3 * 0.5
        elif quadrature_type == 'interior_cubic':
            assert order is None or order <= 3
            assert npoints is None or npoints == 4
            # this would work for arbitrary reference elements
            # L, A = self.subentity_embedding(1)
            # return np.array(L.dot(self.sub_reference_element().center()) + A), np.ones(3) / len(A) * self.volume
            return np.array(([1./3., 1./3.], [1./5., 3./5.], [1./5., 1./5.], [3./5., 1./5.])),\
                   np.array((-27./48., 25./48, 25./48, 25./48)) * 0.5
        else:
            raise NotImplementedError('quadrature_type must be "center" or "edge_centers"')

# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.functions.interfaces import FunctionInterface


def P1ShapeFunctions(dim):

    assert isinstance(dim, int)
    assert dim > 0

    #assert isinstance(x, np.ndarray)

    if dim == 1:
        return lambda X: np.array((
            1.0 - X[..., 0],
            X[..., 0]
        ))
    elif dim == 2:
        return lambda X: np.array([
            1.0 - X[..., 0],
            X[..., 0],
            X[..., 1]
        ])
    else:
        raise NotImplementedError


def P1ShapeFunctionGradients(dim):

    assert isinstance(dim, int)
    assert dim > 0

    if dim == 1:
        return np.array([
            [-1.0],
            [1.0]
        ])
    elif dim == 2:
        return np.array([
            [-1.0, -1.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ])
    else:
        raise NotImplementedError


def P2ShapeFunctions(dim):

    assert isinstance(dim, int)
    assert dim > 0

    #assert isinstance(x, np.ndarray)

    if dim == 1:
        return lambda X: np.array([
            2.0*X[..., 0]**2 - 3.0*X[..., 0] + 1.0,
            2.0*X[..., 0]**2 - X[..., 0],
            -4.0*X[..., 0]**2 + 4.0*X[..., 0]
        ])
    elif dim == 2:
        return lambda X: np.array([
            (X[..., 0] + X[..., 1] - 1.0)*(2.0*X[..., 0] + 2.0*X[..., 1] - 1.0),
            X[..., 0]*(2.0*X[..., 0] - 1.0),
            X[..., 1]*(2.0*X[..., 1] - 1.0),
            4.0*X[..., 0]*X[..., 1],
            -4.0*X[..., 1]*(X[..., 1] + X[..., 0] - 1.0),
            -4.0*X[..., 0]*(X[..., 0] + X[..., 1] - 1.0)
        ])
    else:
        raise NotImplementedError


def P2ShapeFunctionGradients(dim):

    assert isinstance(dim, int)
    assert dim > 0

    #assert isinstance(x, np.ndarray)

    if dim == 1:
        return lambda X: np.array([
            [4.0*X[..., 0] - 3.0],
            [4.0*X[..., 0] - 1.0],
            [-8.0*X[..., 0] + 4.0]
        ])
    elif dim == 2:
        return lambda X: np.array([
            [4.0*X[..., 0] + 4.0*X[..., 1] - 3.0, 4.0*X[..., 1] + 4.0*X[..., 0] - 3.0],
            [4.0*X[..., 0] - 1.0, np.zeros_like(X[..., 0])],
            [np.zeros_like(X[..., 0]), 4.0*X[..., 1] - 1.0],
            [4.0*X[..., 1], 4.0*X[..., 0]],
            [-4.0*X[..., 1], -8.0*X[..., 1] - 4.0*X[..., 0] + 4.0],
            [-8.0*X[..., 0] - 4.0*X[..., 1] + 4.0, -4.0*X[..., 0]]
        ])
    else:
        raise NotImplementedError

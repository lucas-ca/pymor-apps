# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Module containing a block operator."""

from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.sparse import bmat

from pymor.operators.numpy import NumpyMatrixBasedOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace


class StokesLhsBlockOperator(NumpyMatrixBasedOperator):
    """Represents the following Stokes Operator:
        A  B1
        B2 C
    """

    def __init__(self, blocks):
        assert isinstance(blocks, (tuple, list))
        assert len(blocks) == 4, "blocks must be of the form [A, B, Bt, C]"
        self.blocks = blocks  # [A, B, Bt, C]
        #s = 2*blocks[0].source.dim + blocks[1].source.dim
        #r = 2*blocks[0].range.dim + blocks[2].range.dim
        s = blocks[0].source.dim + blocks[1].source.dim
        r = blocks[0].range.dim + blocks[2].range.dim
        self.source = NumpyVectorSpace(s)
        self.range = NumpyVectorSpace(r)
        self.build_parameter_type(inherits=blocks)

    def _assemble(self, mu=None):
        b = self.blocks

        # reduced case with dense matrices
        if type(b[0].assemble(mu)._matrix) == np.ndarray:
            return np.bmat([[b[0].assemble(mu)._matrix, b[1].assemble(mu)._matrix],
                            [b[2].assemble(mu)._matrix, b[3].assemble(mu)._matrix]])
        # sparse case
        else:
            return bmat([[b[0].assemble(mu)._matrix, b[1].assemble(mu)._matrix],
                         [b[2].assemble(mu)._matrix, b[3].assemble(mu)._matrix]])


class StokesRhsBlockOperator(NumpyMatrixBasedOperator):
    """Represents the following Stokes Functional:
        F
        Z
    """

    def __init__(self, blocks):
        assert isinstance(blocks, (tuple, list))
        assert len(blocks) == 2  # [F, Fz]
        self.blocks = blocks
        s = blocks[0].source.dim + blocks[1].source.dim
        self.source = NumpyVectorSpace(s)
        self.range = NumpyVectorSpace(1)
        self.build_parameter_type(inherits=blocks)

    def _assemble(self, mu=None):
        # return np.hstack((self.blocks[0]._assemble(mu), self.blocks[1]._assemble(mu)))
        f1 = self.blocks[0].assemble(mu)._matrix
        f2 = self.blocks[1].assemble(mu)._matrix

        return np.hstack((self.blocks[0].assemble(mu)._matrix, self.blocks[1].assemble(mu)._matrix))


class DiagonalBlockOperator(NumpyMatrixBasedOperator):
    """Represents a diagonal block operator:
        O_0  0   0
         0  O_1  0
         0   0  O_2
    """

    def __init__(self, blocks):
        assert isinstance(blocks, (tuple, list))
        self.blocks = blocks  # [A, B, Bt, C]
        #s = 2*blocks[0].source.dim + blocks[1].source.dim
        #r = 2*blocks[0].range.dim + blocks[2].range.dim
        s = sum([op.source.dim for op in blocks])
        r = sum([op.range.dim for op in blocks])
        self.source = NumpyVectorSpace(s)
        self.range = NumpyVectorSpace(r)
        self.build_parameter_type(inherits=blocks)

    def _assemble(self, mu=None):
        b = self.blocks

        if len(b) == 2:
            A = bmat([[b[0].assemble(mu)._matrix, None],
                      [None, b[1].assemble(mu)._matrix]])
        elif len(b) == 3:
            A = bmat([[b[0].assemble(mu)._matrix, None, None],
                      [None, b[1].assemble(mu)._matrix, None],
                      [None, None, b[2].assemble(mu)._matrix]])
        else:
            raise NotImplementedError

        return A

    #def projected(self, range_basis, source_basis, product=None, name=None):
    #    proj_operators = [op.projected(range_basis=range_basis, source_basis=source_basis, product=product)
    #                      for op in self.blocks]

    #    return super(DiagonalBlockOperator).__init__(proj_operators)



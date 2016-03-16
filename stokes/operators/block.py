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
        A 0 B
        0 A B
        B B C
    """

    def __init__(self, blocks):
        assert isinstance(blocks, (tuple, list))
        assert len(blocks) == 4, "blocks must be of the form [A, B, Bt, C]"
        self.blocks = blocks  # [A, B, Bt, C]
        s = 2*blocks[0].source.dim + blocks[1].source.dim
        r = 2*blocks[0].range.dim + blocks[2].range.dim
        self.source = NumpyVectorSpace(s)
        self.range = NumpyVectorSpace(r)
        self.build_parameter_type(inherits=blocks)

    def _assemble(self, mu=None):
        b = self.blocks

        A = bmat([[b[0].assemble(mu)._matrix, None],[None, b[0].assemble(mu)._matrix]])
        return bmat([[A, b[1].assemble(mu)._matrix], [b[2].assemble(mu)._matrix, b[3].assemble(mu)._matrix]])


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
        self.range=NumpyVectorSpace(1)
        self.build_parameter_type(inherits=blocks)

    def _assemble(self, mu=None):
        #return np.hstack((self.blocks[0]._assemble(mu), self.blocks[1]._assemble(mu)))
        return np.hstack((self.blocks[0].assemble(mu)._matrix, self.blocks[1].assemble(mu)._matrix))


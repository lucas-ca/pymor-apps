# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
#
# Contributors: Michael Laier <m_laie01@uni-muenster.de>

from __future__ import absolute_import, division, print_function

import numpy as np

import collections
from pymor.domaindescriptions.boundarytypes import BoundaryType
from pymor.domaindescriptions.interfaces import DomainDescriptionInterface


class PolygonalDomain(DomainDescriptionInterface):
    """Describes a domain with a polygonal boundary and polygonal holes inside the domain.

    Parameters
    ----------
    points
        List of points [x_0, x_1] that describe the polygonal chain that bounds the domain.
    boundary_types
        Either a dictionary {BoundaryType: [i_0, ...], BoundaryType: [j_0, ...], ...} with i_0, ... being the
        id of the line (starting with 0) that connects the corresponding points,
        or a function that returns the |BoundaryType| for a given coordinate.
    holes
        List of lists of points that describe the polygonal chains that bound the holes inside the domain.
    inner_edges
        Force inner edges. List of list of indices of points that describe the inner edges. No boundary conditions can
        be provided.

    Attributes
    ----------
    points
    boundary_types
    holes
    inner_edges
    """

    dim = 2

    def __init__(self, points, boundary_types, holes=[], inner_edges=[], subdomains=[]):
        self.points = points
        self.holes = holes
        self.inner_edges = inner_edges
        self.subdomains = subdomains

        if isinstance(boundary_types, dict):
            self.boundary_types = boundary_types
        # if the |BoundaryTypes| are not given as a dict, try to evaluate at the edge centers to get a dict.
        else:
            points = [points]
            points.extend(holes)
            # shift points 1 entry to the left.
            points_deque = [collections.deque(ps) for ps in points]
            for ps_d in points_deque:
                ps_d.rotate(-1)
            # compute edge centers.
            centers = [[(p0[0]+p1[0])/2, (p0[1]+p1[1])/2] for ps, ps_d in zip(points, points_deque)
                       for p0, p1 in zip(ps, ps_d)]
            # evaluate the boundary at the edge centers and save the |BoundaryTypes| together with the
            # corresponding edge id.
            self.boundary_types = dict(zip([boundary_types(centers)], [range(1, len(centers)+1)]))

        # check if the dict keys are given as |BoundaryType|
        assert all(isinstance(bt, BoundaryType) for bt in self.boundary_types.iterkeys())

    def __repr__(self):
        return 'PolygonalDomain({}, {}, {}, {})'.format(repr(self.points), repr(self.boundary_types), repr(self.holes),
                                                        repr(self.inner_edges), repr(self.subdomains))


from __future__ import absolute_import, division, print_function

from pymor.domaindescriptions.basic import RectDomain
from pymor.grids.tria import TriaGrid

import numpy as np


def composite_tria_grid(domains, num_intervals):
    assert isinstance(domains, RectDomain) or \
           domains in (tuple, list) and all(isinstance(d, RectDomain) for d in domains)
    assert num_intervals in (list, tuple)
    assert len(num_intervals) == 2

# 2 1 1 3

def validate_edges(domains):
    edges = {}
    for d in domains:
        for e in get_edges(d):
            if edges.has_key(e):
                edges[e] += 1
            else:
                edges[e] = 1

    return all(value <= 2 for value in edges.values())


def validate_domains(domains):
    assert isinstance(domains, RectDomain) or \
           isinstance(domains, (list, tuple)) and all(isinstance(d, RectDomain) for d in domains)

    if isinstance(domains, RectDomain):
        return True

    if not validate_edges(domains):
        return False

    for d1 in domains:
        for d2 in domains:
            if d1 == d2:
                break
            for e1 in get_edges(d1):
                for e2 in get_edges(d2):
                    if coincide_edges(e1, e2):
                        pass




def get_edges(domain):
    assert isinstance(domain, RectDomain)

    d = domain.domain

    ll = tuple(d[0])
    ur = tuple(d[1])
    lr = (d[1][0], d[0][1])
    ul = (d[0][0], d[1][1])

    eb = (ll, lr)
    er = (lr, ur)
    et = (ul, ur)
    el = (ll, ul)

    return (eb, er, et, el)


def coincide_edges(e1, e2):
    assert isinstance(e1, tuple)
    assert isinstance(e2, tuple)
    assert all(isinstance(e, tuple) for e in e1)
    assert all(isinstance(e, tuple) for e in e2)

    return np.allclose(e1, e2) or np.allclose(e1, e2[::-1])


if __name__ == '__main__':

    d1 = RectDomain()
    d2 = RectDomain(domain=([1, 0], [2, 1]))

    d = (d1, d2)

    v = validate_edges(d)
    v = validate_domains(d)

    from stokes.domaindescriptions.polygonal import PolygonalDomain
    from pymor.domaindescriptions.boundarytypes import BoundaryType
    from pymor.grids.gmsh import load_gmsh
    from stokes.domaindiscretizers.gmsh import discretize_gmsh

    #p = PolygonalDomain([[0, 0], [0.5, 0], [1, 0], [1, 0.5], [1, 1], [0.5, 1], [0, 1], [0, 0.5]],
    #                    {BoundaryType('dirichlet'): [0, 1, 2, 3, 4, 5, 6, 7]}, [], [[1, 5], [3, 7]])
    p2 = PolygonalDomain([[0, 0], [1, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2], [0, 1]],
                         {BoundaryType('dirichlet'): [0, 1, 2, 3, 4, 5, 6, 7]},
                         [],
                         [[1, 4], [4, 7]])
    #g, b = discretize_gmsh(geo_file=open('/home/lucas/test45.geo'))
    g, b = discretize_gmsh(p2, geo_file_path='/home/lucas/test42.geo')


    from matplotlib import pyplot as plt
    c = g.centers(2)
    i = g.subentities(0, 2)

    plt.triplot(c[..., 0], c[..., 1], i)

    b = 0

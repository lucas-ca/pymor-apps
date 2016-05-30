import numpy_ as np

from pymor.grids.tria import TriaGrid

from stokes.analyticalproblems.poiseuille import PoiseuilleProblem
from stokes.operators.cg import DiffusionOperatorP2

PROBLEM = 1
FEM_ORDER = 2
NUM_ELEMENTS = 10

if PROBLEM == 1:
    p = PoiseuilleProblem()

grid = TriaGrid(num_intervals=(NUM_ELEMENTS, NUM_ELEMENTS),
                domain=p.domain)

# operators:
if FEM_ORDER == 1:
    pass
elif FEM_ORDER == 2:
    A = DiffusionOperatorP2
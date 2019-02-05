"""
Finis
=====

Finite Element (French: elements finis) package for MAP 584 course at Ecole Polytechnique, Paris. This package relies on meshes generated with triangle: https://pypi.org/project/triangle/ .

Quickstart
----------
Create a mesh with the simple wrapper around triangle:
>>> import numpy as np
>>> import finis
>>> mesh = finis.triangulate()

Plot mesh using vertex and triangle numbers:
>>> finis.plot_mesh(mesh, vertex_numbers=True, triangle_numbers=True)

Build finite element (fe) space with shape functions of order 2:
>>> fe = finis.fe_space(mesh, order=2)

Integrate simple function:
>>> integrand = lambda x,y: np.exp(x)
>>> I = finis.integrate(fe, integrand)
>>> print("Numeric : {}".format(I))
>>> print("Analytic: {}".format(np.expm1(1)))
Numeric : 1.7182772860575093
Analytic: 1.718281828459045

Use finis.integrate to evaluate symbolic expression
and validate integration by parts:
>>> integrand = "dx_f*g + f*dx_g"
>>> funs = {
>>>     'f': lambda x,y: np.sin(x * (1-x) * np.exp(y)),
>>>     'g': lambda x,y: np.cos(x**2 + y**3),
>>> }
>>> I = finis.integrate(fe, integrand, funs)
>>> print("Numeric : {}".format(I))
>>> print("Analytic: {}".format(0.0))
Numeric : 9.972496077168339e-05
Analytic: 0.0
"""

__version__ = "0.0.1"

from ._mesh import triangulate, plot_mesh
from ._fem import fe_space, integrate

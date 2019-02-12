import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg
from ._fem import fe_space
from ._mesh import triangulate

def model_problem(fe, f, dirichlet=False):
    F_i = f(fe['integ'][:,0], fe['integ'][:,1])
    W = sp.spdiags(fe['w'], [0], m=fe['w'].size, n=fe['w'].size)
    A = fe['U'].transpose().dot(W).dot(fe['U']) + \
        fe['DUX'].transpose().dot(W).dot(fe['DUX']) + \
        fe['DUY'].transpose().dot(W).dot(fe['DUY'])
    F = fe['U'].transpose().dot(W).dot(F_i)
    
    if not dirichlet:
        # Neumann Boundary
        u_h = sp.linalg.spsolve(A, F)
    else:
        #Dirichlet
        row = np.where(fe['markers']==0)[0] #before: inner_to_all
        #all_to_inner = -np.ones(shape=markers.shape, dtype=np.int32)
        #all_to_inner[inner_to_all] = np.arange(inner_to_all.size)

        col = np.arange(row.size)
        data = np.ones((row.size, ), dtype=np.float)
        P = sp.csr_matrix((data, (row, col)), shape=(fe['markers'].size, row.size))
        
        Ad = P.transpose().dot(A).dot(P)
        Fd = P.transpose().dot(F)
        u_h = P.dot(sp.linalg.spsolve(Ad, Fd))
    
    return u_h

def error_plot(vertices, f, u, dux, duy, order=1, min_max_area=0.01, max_max_area=1.0, num=50, dirichlet=False):
    """
    vertices
    f: rhs
    u: solution
    dux: d/dx u
    duy: d/dy u
    min_max_area
    max_max_area
    steps
    order
    dirichlet=False (either dirichlet or neumann)
    """
    
    if dirichlet:
        mesh = triangulate(vertices=vertices, max_area=min_max_area)
        fe = fe_space(mesh, order=order, return_h=True)
        assert np.allclose(u(fe['dof'][fe['markers']==1,0], fe['dof'][fe['markers']==1,1]), 0), "Analytic u does not vanish on boundary."
    
    max_area = np.logspace(np.log10(max_max_area), np.log10(min_max_area), base=10.0, num=num)
    L2 = np.zeros(max_area.shape, dtype=np.float64)
    H1 = np.zeros(max_area.shape, dtype=np.float64)
    hs = np.zeros(max_area.shape, dtype=np.float64)

    for i in range(max_area.size):
        mesh = triangulate(vertices=vertices, max_area=max_area[i])
        fe = fe_space(mesh, order=order, return_h=True)
        u_h = model_problem(fe, f, dirichlet=dirichlet)
        u_ana = u(fe['dof'][:,0], fe['dof'][:,1])

        xi = fe['integ'][:,0]
        yi = fe['integ'][:,1]
        # L2
        integrand = (fe['U'].dot(u_h) - u(xi, yi))**2
        L2[i] = np.sqrt(np.sum(fe['w'] * integrand))
        # H1
        integrand = (fe['DUX'].dot(u_h) - dux(xi, yi))**2 + (fe['DUY'].dot(u_h) - duy(xi, yi))**2
        H1[i] = np.sqrt(L2[i]**2 + np.sum(fe['w'] * integrand))
        hs[i] = np.mean(fe['h'])
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.loglog(hs, L2, '+-', label="L2 error")
    plt.loglog(np.array([hs[-1], hs[0]]), np.array([L2[-1], L2[-1]*(hs[0]/hs[-1])**(order+1)]), 'r:', label="h^{}".format(order+1))
    plt.xlabel("h [mean edge length]");
    plt.ylabel("L2 error");
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.loglog(hs, H1, '+-', label="H1 error")
    plt.loglog(np.array([hs[-1], hs[0]]), np.array([H1[-1], H1[-1]*(hs[0]/hs[-1])**(order)]), 'r:', label="h^{}".format(order))
    plt.xlabel("h [mean edge length]");
    plt.ylabel("H1 error");
    plt.legend()
    
    if dirichlet:
        plt.suptitle("Dirichlet FEM P{}".format(order))
    else:
        plt.suptitle("Neumann FEM P{}".format(order))
    plt.show()
    

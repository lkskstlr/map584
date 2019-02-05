import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


def triangle_x(m):
    return m['vertices'][m['triangles'],0]

def triangle_y(m):
    return m['vertices'][m['triangles'],1]

def triangle_cog_x(m):
    return np.mean(triangle_x(m), axis=1)

def triangle_cog_y(m):
    return np.mean(triangle_y(m), axis=1)

def triangle_cog(m):
    return np.hstack((triangle_cog_x(m)[:,None], triangle_cog_y(m)[:,None]))

def plot_mesh(m, vertex_numbers=False, triangle_numbers=False, edge_numbers=False, edge_labels=False):
    plt.triplot(m['vertices'][:,0], m['vertices'][:,1], m['triangles'])
    
    if vertex_numbers:
        for i in range(m['vertices'].shape[0]):
            plt.text(m['vertices'][i, 0], m['vertices'][i, 1], str(i),
                     color='r',
                     horizontalalignment='center',
                     verticalalignment='center')
            
    if triangle_numbers:
        cogs = triangle_cog(m)
        for i in range(cogs.shape[0]):
            plt.text(cogs[i, 0], cogs[i, 1], str(i))
            
    if edge_labels or edge_numbers:
        from edges import meshEdges
        edge, edge_markers, ElementEdges = meshEdges(m)
        
    if edge_numbers:
        for i in range(edge.shape[0]):
            _x = np.mean(m['vertices'][edge[i,:], 0])
            _y = np.mean(m['vertices'][edge[i,:], 1])
            plt.text(_x, _y, str(i), color='g')
            
    if edge_labels:
        for i in range(edge.shape[0]):
            _x = np.mean(m['vertices'][edge[i,:], 0])
            _y = np.mean(m['vertices'][edge[i,:], 1])
            plt.text(_x, _y, edge_markers[i,0], color='g')
            
            
            
def form_int_1d(order=2):
    if order == 1:
        x = np.array([1/2])
        w = np.array([1.0])
        
        return x, w
    
    if (order == 2) or (order == 3):
        x = np.array([1/2 - 1/sqrt(12), 1/2 + 1/sqrt(12)])
        w = np.array([1/2, 1/2])
        
        return x, w
    
    if (order == 4) or (order == 5):
        x = np.array([1/2 - 1/2*sqrt(3/5), 1/2, 1/2 + 1/2*sqrt(3/5)])
        w = np.array([5/18, 8/18, 5/18])
        
        return x, w
    

    raise NotImplementedError("Only orders 1, ..., 5 are implemented")

def form_int_2d(order=2):
    if order == 1:
        x = np.array([1/3])
        y = np.array([1/3])
        w = np.array([1/2])
        
        return x, y, w
    
    if order == 2:
        x = np.array([1/6, 2/3, 1/6])
        y = np.array([1/6, 1/6, 2/3])
        w = np.array([1/6, 1/6, 1/6])
        
        return x, y, w
    
    if order == 3:
        x = np.array([1/3, 1/5, 3/5, 1/5])
        y = np.array([1/3, 1/5, 1/5, 3/5])
        w = np.array([-9/32, 25/96, 25/96, 25/96])
        
        return x, y, w
    
    if order == 4:
        x = np.array([1/2, 1/2, 0, 1/6, 1/6, 2/3])
        y = np.array([1/2, 0, 1/2, 1/6, 2/3, 1/6])
        w = np.array([1/60, 1/60, 1/60, 3/20, 3/20, 3/20])
        
        return x, y, w
    
    if (order == 5) or (order == 6):
        a = (6 + sqrt(15))/21
        b = (6 - sqrt(15))/21
        A = (155 + sqrt(15))/2400
        B = (155 - sqrt(15))/2400
        
        x = np.array([1/3, a, 1-2*a, a, b, 1-2*b, b])
        y = np.array([1/3, a, a, 1-2*a, b, b, 1-2*b])
        w = np.array([9/80, A, A, A, B, B, B])
        
        return x, y, w
    
    raise NotImplementedError("Only orders 1, ..., 6 are implemented")
    
    
def integrate(fun, order=2):
    x,y,w = form_int_2d(order=order)
    f = fun(x,y)
    return np.sum(f*w)


def build_integ(mesh, order=2):
    xh, yh, wh = form_int_2d(order=order)
    Nint = xh.size
    N = mesh['triangles'].shape[0]
    x = np.zeros((Nint * N, ))
    y = np.zeros((Nint * N, ))
    w = np.zeros((Nint * N, ))
    
    xT = triangle_x(mesh)
    yT = triangle_y(mesh)
    
    d1x = xT[:,1]-xT[:,0]
    d2x = xT[:,2]-xT[:,0]
    
    d1y = yT[:,1]-yT[:,0]
    d2y = yT[:,2]-yT[:,0]
    
    c_T = np.abs(d1x*d2y - d2x*d1y) # area transformation
    
    for i in range(N):
        x[Nint*i:Nint*(i+1)] = xT[i,0] + d1x[i]*xh + d2x[i]*yh
        y[Nint*i:Nint*(i+1)] = yT[i,0] + d1y[i]*xh + d2y[i]*yh
        w[Nint*i:Nint*(i+1)] = wh * c_T[i]
    
    return {'x':x, 'y':y, 'w':w}



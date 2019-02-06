import numpy as np


def meshEdges(m):
    """
        Input: un maillage m

        Outputs: 'edge' est un tableau avec la liste de toutes les arêtes du maillage
                 'edge_markers' renvoie le label des arêtes avec
                        0 pour une arête intérieure qui n'est pas un segment fixé par le maillage
                        i>0 où i est le label du bord sur lequel se trouve l'arête
                 'ElementEdges' est un tableau contenant, pour chaque triangle, la liste de ses arêtes

       Note : Pour avoir les labels des arêtes il faut garder le champ 'segments' dans le maillage
              donc il faut mettre l'option 'p' lorsqu'on construit le maillage m
              m = triangle.triangulate(A,'p')

    """

    tri = m['triangles']
    nb_tri, _ = tri.shape

    e01 = np.array([tri[:, 0], tri[:, 1]]).T
    e01 = np.array([np.amin(e01, axis=1), np.amax(e01, axis=1)]).T

    e12 = np.array([tri[:, 1], tri[:, 2]]).T
    e12 = np.array([np.amin(e12, axis=1), np.amax(e12, axis=1)]).T

    e02 = np.array([tri[:, 0], tri[:, 2]]).T
    e02 = np.array([np.amin(e02, axis=1), np.amax(e02, axis=1)]).T

    etot = np.concatenate((e01, e12, e02))

    edge, indices = np.unique(etot, axis=0, return_inverse=True)

    ElementEdges = np.array(
        [indices[nb_tri:2 * nb_tri], indices[2 * nb_tri:], indices[:nb_tri]]).T

    if 'segments' not in m:
        raise ValueError(
            "Attention : il faut ajouter l'option 'p' dans la commande triangle.triangulate")

    seg = m['segments']
    seg = np.array([np.amin(seg, axis=1), np.amax(seg, axis=1)]).T

    segedg = np.concatenate((edge, seg))
    edge_lab, ind_lab = np.unique(segedg, axis=0, return_inverse=True)

    ne = edge.shape[0]

    edge_markers = np.zeros((ne, 1), dtype='int')
    edge_markers[ind_lab[ne:]] = m['segment_markers']

    return edge, edge_markers, ElementEdges


def mesh_edges(mesh):
    """
        Input: mesh

        Outputs: 'edge' est un tableau avec la liste de toutes les arêtes du maillage
                 'ElementEdges' est un tableau contenant, pour chaque triangle, la liste de ses arêtes

        Note : See meshEdges for reference implementation. This avoids one ouput to gain speed.

    """

    tri = mesh['triangles']
    N_tri = tri.shape[0]

    etot = np.concatenate((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [0, 2]]))
    etot.sort(axis=1)

    edge, indices = np.unique(etot, axis=0, return_inverse=True)
    # The require is needed for C continuous data
    element_edges = np.require(
        np.reshape(
            np.roll(
                indices, -N_tri), (N_tri, 3), order='F'), requirements='C')

    return edge, element_edges


def test_mesh_edges(mesh):
    """ Test new vs reference implementation """
    edge_old, _, element_edges_old = meshEdges(mesh)
    edge, element_edges = mesh_edges(mesh)
    assert np.array_equal(edge_old, edge)
    assert np.array_equal(element_edges_old, element_edges)
    print("Correct")

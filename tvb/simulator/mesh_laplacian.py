# -*- coding: utf-8 -*-

"""
Calculates Discrete Laplacian on triangulated surface.

Belkinetal2008 "Discrete Laplace Operator On Meshed Surface"

ARGUMENTS:
    TR -- Surface as a TriRep object
    Neighbourhood -- The N-ring to truncate approximation.  
    AverageNeighboursPerVertex -- <description>

OUTPUT: 
    lap_op -- Discrete approximation to Laplace-Beltrami operator.
    Convergence -- <description>

REQUIRES:
    dis() -- euclidean distance.
    GetLocalSurface() -- extracts sub-region of surface.
    perform_fast_marching_mesh() -- Calculates geodesic distances 
    between vertices on a mesh 
    surface.

USAGE::
    
    ThisSurface = 'reg13';
    load(['Cortex_' ThisSurface '.mat'], 'Vertices', 'Triangles'); # Contains: 'Vertices', 'Triangles', 'VertexNormals', 'TriangleNormals' 
    tr = TriRep(Triangles, Vertices);     # Convert to TriRep object
    load(['SummaryInfo_Cortex_' ThisSurface '.mat'], 'meanEdgeLength'); #
    
    [lap_op, Convergence] = MeshLaplacian(tr, 8, meanEdgeLength);
    
    #Plot to check, ratio max outer ring / dominant contribution, the
    #closer to zero these values are the better.
    figure, plot(Convergence ./ max(lap_op))

.. moduleauthor:: Stuart A. Knock <stuart.knock@gmail.com>

"""

#TODO: this will make most sense as a method on Surface datatypes... 

import numpy
from scipy import sparse


def MeshLaplacian(surface, cutoff):  #function [lap_op, Convergence] = 
    """
    Calculate a discrete approximation of the Laplace-Beltrami operator for a 
    Surface object from TVB's datatypes.

    """
    nv = surface.number_of_vertices
    
    # NOTE: 'h' needs to be set such that the last vertices inside cutoff contribute ~ 0...
    h = cutoff / 3.0 #~3std
    h4 = h * 4.0; #QUERY: should it be h^2 so exp(...) is dimensionless???
    
    c1 = 1.0 / (12.0 * numpy.pi * h**2) #NOTE: 12 = 4 * 3, where 3 is vertices per triangle
    
    #NOTE: the 1/h^2 in C1 has the role of  division by dx^2,  as it corresponds to an effective
    #      neighbourhood considered by the Laplacian. ?THINK THIS IS TRUE?
    #      So don't do: lap_op(GlobalVertexIndices(2:end),i) = lap_op(GlobalVertexIndices(2:end),i).' ./ DeltaX.^2
    
    #Get distance to vertices in neighbourhood of current vertex
    gauss_dis_weight = surface.geodesic_distance_matrix.copy()
    gauss_dis_weight.data = numpy.exp(-gauss_dis_weight.data**2 / h4)
    
    vertex_onering_area = numpy.zeros(nv)
    for vertex in range(nv):
        vertex_triangles = list(surface.vertex_triangles[vertex])
        vertex_onering_area[vertex] = numpy.sum(surface.triangle_areas[vertex_triangles])
    
    weighted_area = c1 * vertex_onering_area
    
    ind = numpy.arange(nv, dtype=int)
    weighted_area_mat = sparse.csc_matrix((weighted_area, (ind, ind)), shape=(nv, nv))
    
    #
    lap_op = weighted_area_mat * gauss_dis_weight
    #import pdb; pdb.set_trace()
    diag_vals = numpy.array(lap_op.sum(axis=1)).squeeze()
    #import pdb; pdb.set_trace()
    lap_op = lap_op - sparse.csc_matrix((diag_vals, (ind, ind)), shape=(nv, nv))
    
    #TODO: Should reintroduce the convergence checks...
    
    return lap_op



if __name__ == '__main__':
    # Do some stuff that tests or makes use of this module... 
    print 'testing module...'



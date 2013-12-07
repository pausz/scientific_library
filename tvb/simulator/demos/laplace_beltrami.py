

"""

For the Mayavi based visualisation to be functional under Linux you'll 
probably want to start ipython with:
    
    ipython --pylab=qt



"""

import scipy.sparse

import tvb.datatypes.surfaces as surfaces
import tvb.simulator.mesh_laplacian as mlb

from tvb.simulator.plot.tools import surface_pattern



#setting coupling_strength to 1.0 as we don't what to rescale the 
#Laplace-Beltrami operator -- even though we're reusing the surface's
#local_coupling, it isn't one...

default_cortex = surfaces.Cortex(coupling_strength=1.0)

#Also we don't want it to auto-calculate a default local connectivity for us
#at configure time, so set a dummy default
default_cortex.local_connectivity = surfaces.LocalConnectivity()
default_cortex.local_connectivity.matrix = scipy.sparse.eye(42)


default_cortex.configure()

surface.compute_geodesic_distance_matrix(max_dist=42.0)

lapop = mlb.MeshLaplacian(default_cortex, 42.0)


local_lapop = lapop[:, 0].todense()

surface_pattern(default_cortex, local_lapop)

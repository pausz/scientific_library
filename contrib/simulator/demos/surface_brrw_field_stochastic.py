# -*- coding: utf-8 -*-
#

"""
Demonstrate using the simulator for a surface simulation, stochastic 
integration, using the BRRW field equations.

Noise enters through a single state-variable approximating the phi_n term...

``Run time``: approximately 142 seconds (workstation circa 2010).
``Memory requirement``: ~ 2 GB

.. moduleauthor:: Stuart A. Knock <stuart.knock@gmail.com>

"""


# Third party python libraries
import numpy
import scipy.sparse

from tvb.simulator.common import get_logger
LOG = get_logger(__name__)

#Import from tvb.simulator modules
#import tvb.simulator #For eeg_projection hack...
import tvb.simulator.simulator as simulator
import tvb.simulator.coupling as coupling
import tvb.simulator.integrators as integrators
import tvb.simulator.monitors as monitors

import tvb.datatypes.connectivity as connectivity
import tvb.datatypes.surfaces as surfaces

# Add the contributed models directory to the PYTHONPATH
sys.path += ["../models"]

from brrw import BRRW
import tvb.simulator.mesh_laplacian as mlb


from matplotlib.pyplot import figure, plot, title, show
from tvb.simulator.plot.tools import surface_timeseries


##----------------------------------------------------------------------------##
##-                      Perform the simulation                              -##
##----------------------------------------------------------------------------##

LOG.info("Configuring...")
#Initialise a Model, Coupling, and Connectivity.
field = BRRW()
white_matter = connectivity.Connectivity()
white_matter.speed = numpy.array([4.0])

white_matter_coupling = coupling.Linear(a=0.0)

#Initialise an Integrator
heunint = integrators.HeunStochastic(dt=2**-4)

#Yet Another Ugly Hack
heunint.noise.nsig = numpy.zeros(8)
heunint.noise.nsig[5] = field.alfa * field.btta *  field.nu_sn

#Initialise some Monitors with period in physical time
mon_tavg = monitors.TemporalAverage(period=2**-2)
mon_savg = monitors.SpatialAverage(period=2**-2)
mon_eeg = monitors.EEG(period=2**-2)

#Bundle them
what_to_watch = (mon_tavg, mon_savg, mon_eeg)


#setting coupling_strength to 1.0 as we don't what to rescale the 
#Laplace-Beltrami operator -- even though we're reusing the surface's
#local_coupling, it isn't one...

default_cortex = surfaces.Cortex(coupling_strength=1.0)

#Also we don't want it to auto-calculate a default local connectivity for us
#at configure time, so set a dummy default
default_cortex.local_connectivity = surfaces.LocalConnectivity()
default_cortex.local_connectivity.matrix = scipy.sparse.eye(42)


default_cortex.configure()

default_cortex.compute_geodesic_distance_matrix(max_dist=42.0)

lapop = mlb.MeshLaplacian(default_cortex, 42.0)

default_cortex.local_connectivity.matrix = lapop

#Initialise Simulator -- Model, Connectivity, Integrator, Monitors, and surface.
sim = simulator.Simulator(model = field, 
                          connectivity = white_matter,
                          coupling = white_matter_coupling, 
                          integrator = heunint, 
                          monitors = what_to_watch,
                          surface = default_cortex)

sim.configure()

LOG.info("Starting simulation...")
#Perform the simulation
tavg_data = []
tavg_time = []
savg_data = []
savg_time = []
eeg_data = []
eeg_time = []
for tavg, savg, eeg in sim(simulation_length=2**2):
    if not tavg is None:
        tavg_time.append(tavg[0])
        tavg_data.append(tavg[1])
        
    if not savg is None:
        savg_time.append(savg[0])
        savg_data.append(savg[1])
        
    if not eeg is None:
        eeg_time.append(eeg[0])
        eeg_data.append(eeg[1])

LOG.info("finished simulation.")

##----------------------------------------------------------------------------##
##-               Plot pretty pictures of what we just did                   -##
##----------------------------------------------------------------------------##

#Make the lists numpy.arrays for easier use.
TAVG = numpy.array(tavg_data)
SAVG = numpy.array(savg_data)
EEG = numpy.array(eeg_data)

#Plot region averaged time series
figure(3)
plot(savg_time, SAVG[:, 0, :, 0])
title("Region average")

#Plot EEG time series
figure(4)

color_idx = numpy.linspace(0, 1, EEG.shape[2])
for i in color_idx:
    plot(eeg_time, EEG[:, 0, :, 0], color=cm.cool(i), lw=3, alpha=0.2)
title("EEG")

#Show them
show()

#Surface movie, requires mayavi.malb
if IMPORTED_MAYAVI:
    st = surface_timeseries(sim.surface, TAVG[:, 0, :, 0])

###EoF###
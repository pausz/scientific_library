# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under 
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General 
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
Make use of the correlation_coefficient analyzer to compute functional connectivity, using
the demo data at the region level.
``Run time``: 

``Memory requirement``: 

.. moduleauthor:: Paula Sanz Leon <paula.sanz-leon@univ-amu.fr>

"""

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(["-profile", "LIBRARY_PROFILE"], try_reload=False)

import numpy
import tvb.datatypes.connectivity as connectivity
import tvb.analyzers.correlation_coefficient as corr_coeff
from tvb.datatypes.time_series import TimeSeriesRegion
from tvb.basic.logger.builder import get_logger
from tvb.simulator.plot.tools import *


LOG = get_logger(__name__)

#Load the demo region timeseries dataset 
try:
    data = numpy.load("demo_data_region_16s_2048Hz.npy")
except IOError:
    LOG.error("Can't load demo data. Run demos/generate_region_demo_data.py")
    raise

period = 0.00048828125  # s

#Put the data into a TimeSeriesRegion datatype
white_matter = connectivity.Connectivity()
tsr = TimeSeriesRegion(connectivity=white_matter,
                       data=data,
                       sample_period=period)
tsr.configure()

#Create and run the analyser
corrcoeff_analyser = corr_coeff.CorrelationCoefficient(time_series=tsr)
corrcoeff_data = corrcoeff_analyser.evaluate()

#Generate derived data, if any...
corrcoeff_data.configure()

# Plot matrix with numbers
# For visualization purposes, the diagonal is set to zero.
FC = corrcoeff_data.array_data[:, :, 0, 0]
numpy.fill_diagonal(FC, 0.)
pyplot.matshow(FC, cmap='RdBu', vmin=-0.5, vmax=0.5, interpolation='nearest')
pyplot.colorbar()
pyplot.show()
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
A contributed model: Hindmarsh-Rose

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Paula Sanz Leon <Paula@tvb.invalid>
.. moduleauthor:: Gaurav Malhotra <Gaurav@tvb.invalid>
.. moduleauthor:: Marmaduke Woodman <Marmaduke@tvb.invalid>

"""

# Third party python libraries
import numpy
import numexpr

#The Virtual Brain
from tvb.simulator.common import psutil, get_logger
LOG = get_logger(__name__)

import tvb.datatypes.arrays as arrays
import tvb.basic.traits.types_basic as basic 
import tvb.simulator.models as models


class HindmarshRose(models.Model):
    """
    The Hindmarsh-Rose model is a mathematically simple model for repetitive
    bursting.
    
    .. [HR_1984] Hindmarsh, J. L., and Rose, R. M., *A model of neuronal 
        bursting using three coupled first order differential equations*, 
        Proceedings of the Royal society of London. Series B. Biological 
        sciences 221: 87, 1984.
    
    The models (:math:`x`, :math:`y`) phase-plane, including a representation of
    the vector field as well as its nullclines, using default parameters, can be
    seen below:
        
        .. _phase-plane-HMR:
        .. figure :: img/HindmarshRose_01_mode_0_pplane.svg
            :alt: Hindmarsh-Rose phase plane (x, y)
            
            The (:math:`x`, :math:`y`) phase-plane for the Hindmarsh-Rose model.
    
    
    .. #Currently there seems to be a clash betwen traits and autodoc, autodoc
    .. #can't find the methods of the class, the class specific names below get
    .. #us around this...
    .. automethod:: HindmarshRose.__init__
    .. automethod:: HindmarshRose.dfun
    
    """
    _ui_name = "Hindmarsh-Rose"
    ui_configurable_parameters = ['r', 'a', 'b', 'c', 'd', 's', 'x_1']
    
    #Define traited attributes for this model, these represent possible kwargs.
    r = arrays.FloatArray(
        label = ":math:`r`",
        default = numpy.array([0.001]),
        range = basic.Range(lo = 0.0, hi = 1.0, step = 0.001),
        doc = """Adaptation parameter, governs time-scale of the state variable
        :math:`z`.""",
        order = 1)
    
    a = arrays.FloatArray(
        label = ":math:`a`",
        default = numpy.array([1.0]),
        range = basic.Range(lo = 0.0, hi = 1.0, step = 0.01),
        doc = """Dimensionless parameter, governs x-nullcline""",
        order = 2)
    
    b = arrays.FloatArray(
        label = ":math:`b`",
        default = numpy.array([3.0]),
        range = basic.Range(lo = 0.0, hi = 3.0, step = 0.01),
        doc = """Dimensionless parameter, governs x-nullcline""",
        order = 3)
    
    c = arrays.FloatArray(
        label = ":math:`c`",
        default = numpy.array([1.0]),
        range = basic.Range(lo = 0.0, hi = 1.0, step = 0.01),
        doc = """Dimensionless parameter, governs y-nullcline""",
        order = 4)
    
    d = arrays.FloatArray(
        label = ":math:`d`",
        default = numpy.array([5.0]),
        range = basic.Range(lo = 0.0, hi = 5.0, step = 0.01),
        doc =  """Dimensionless parameter, governs y-nullcline""",
        order = 5)
    
    s = arrays.FloatArray(
        label = ":math:`s`",
        default = numpy.array([1.0]),
        range = basic.Range(lo = 0.0, hi = 1.0, step = 0.01),
        doc = """Adaptation parameter, governs feedback""",
        order = 6)
    
    x_1 = arrays.FloatArray(
        label = ":math:`x_{1}`",
        default = numpy.array([-1.6]),
        range = basic.Range(lo = -1.6, hi = 1.0, step = 0.01),
        doc = """Governs leftmost equilibrium point of x""",
        order = 7)
    
    #Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = basic.Dict(
        label = "State Variable ranges [lo, hi]",
        default = {"x": numpy.array([-4.0, 4.0]),
                   "y": numpy.array([-60.0, 20.0]),
                   "z": numpy.array([-2.0, 18.0])},
        doc = """The values for each state-variable should be set to encompass
        the expected dynamic range of that state-variable for the current 
        parameters, it is used as a mechanism for bounding random inital 
        conditions when the simulation isn't started from an explicit history,
        it is also provides the default range of phase-plane plots.""",
        order = 8)
    
    variables_of_interest = arrays.IntegerArray(
        label = "Variables watched by Monitors",
        range = basic.Range(lo = 0, hi = 3, step=1),
        default = numpy.array([0], dtype=numpy.int32),
        doc = """This represents the default state-variables of this Model to be
        monitored. It can be overridden for each Monitor if desired. The 
        corresponding state-variable indices for this model are :math:`x = 0`,
        :math:`y = 1`,and :math:`z = 2`.""",
        order = 9)
    
#    coupling_variables = arrays.IntegerArray(
#        label = "Variables to couple activity through",
#        default = numpy.array([0], dtype=numpy.int32))
    
#    nsig = arrays.FloatArray(
#        label = "Noise dispersion",
#        default = numpy.array([0.0]),
#        range = basic.Range(lo = 0.0, hi = 1.0))
    
    
    def __init__(self, **kwargs):
        """
        Initialize the HindmarshRose model's traited attributes, any provided
        as keywords will overide their traited default.
        
        """
        LOG.info('%s: initing...' % str(self))
        super(HindmarshRose, self).__init__(**kwargs)
        
        self._state_variables = ["x", "y", "z"]
        self._nvar = 3
        
        self.cvar = numpy.array([0], dtype=numpy.int32)
        
        LOG.debug('%s: inited.' % repr(self))
    
    
    def dfun(self, state_variables, coupling, local_coupling=0.0):
        """
        As in the FitzHugh-Nagumo model ([FH_1961]_), :math:`x` and :math:`y`
        signify the membrane potential and recovery variable respectively.
        Unlike FitzHugh-Nagumo model, the recovery variable :math:`y` is
        quadratic, modelling subthreshold inward current. The third
        state-variable, :math:`z` signifies a slow outward current which leads
        to adaptation ([HR_1984]_):
            
            .. math::
                \\dot{x} &= y - a \\, x^3 + b \\, x^2 - z + I \\\\
                \\dot{y} &= c - d \\, x^2 - y \\\\
                \\dot{z} &= r \\, ( s \\, (x - x_1) - z )
        
        where external currents :math:`I` provide the entry point for local and
        long-range connectivity. Default parameters are set as per Figure 6 of
        [HR_1984]_ so that the model shows repetitive bursting when :math:`I=2`.
        
        """
        
        x = state_variables[0, :]
        y = state_variables[1, :]
        z = state_variables[2, :]
        
        c_0 = coupling[0, :]
        
        dx = y - self.a * x**3 + self.b * x**2 - z + c_0 + local_coupling * x
        dy = self.c - self.d * x**2 - y
        dz = self.r * (self.s * (x - self.x_1) - z)
        
        derivative = numpy.array([dx, dy, dz])
        
        return derivative


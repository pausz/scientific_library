# -*- coding: utf-8 -*-

"""

With paramaters properly specified as a function of 'space' (region), for local
dynamics, coupling, and noise, this is equivalent to the model of Robinson etal,
see [RRW1997]_ and [Breaketal2005]_.


Setting gamma and speed to 0 for non cortical regions achieves the limiting of 
phi to the cortical sheet. dphi also needs to be initialised as 0

nu_ei, local inhibition, should also be set to zero for all non-cortical regions

the weights matrix will also require diagonal entries to account for the 
  ... \\alpha \\beta (\\nu_{ee}\\phi_e ... term in Robinson etal.

setting nsig 0 every where except appropriate 'specific' thalamic nodes acheives 
noise only representing sub-thalamic input...

Restricting to specific inputs to a region is simple via spatialised dynamic
paramaters and spatialised coupling functions...

NOTE: can't match Robinson etal exactly, except with a customised connectivity
      matrix, as we have no mechanism for turning off the activity leaving a
      region within the dynamics and Robinso etal don't include explicit long
      range corticocortical connections... 


.. [RRW1997] Robinson, P. A. and Rennie, C. J. and Wright, J. J., *Propagation
    and stability of waves of electrical activity in the cerebral cortex*,
    Phys. Rev. E},  56-1, p826--840, 1997.

.. [Breaketal2005] M. Breakspear, etal, *A Unifying Explanation of Primary
    Generalized Seizures Through Nonlinear Brain Modeling and Bifurcation
    Analysis*, Cerebral Cortex, 2005. 



.. moduleauthor:: Stuart A. Knock <stuart.knock@gmail.com>

"""

import numpy

#The Virtual Brain
try:
    import tvb.core.logger as logger
    LOG = logger.getLogger(parent_module=__name__, config_root='tvb.simulator')
except ImportError:
    import logging
    LOG = logging.getLogger(__name__)

import tvb.simulator.models as models
import tvb.core.traits.basic as basic
import tvb.datatypes.arrays as arrays



class AFR(models.Model):
    """
    
    
    .. automethod:: __init__
    
    """
    #TODO: Get the ranges from one of the papers...
    _ui_name = "Neural field model"
    
    #TODO: Use sigmoidal eqn., replacing theat, sigma, qmax
    
    #Define traited attributes for this model, these represent possible kwargs.
    theta = arrays.FloatArray(
        label = ":math:`\\theta`",
        default = numpy.array([15.0]),
        range = basic.Range(lo = 0.0, hi = 25.0),
        doc = """Mean neuronal threshold for each 'region' (mV).""")
    
    sigma = arrays.FloatArray(
        label = ":math:`\\sigma`",
        default = numpy.array([6.0]),
        range = basic.Range(lo = 0.0, hi = 15.0),
        doc = """Threshold variability for each 'region' (mV).""")
    
    qmax = arrays.FloatArray(
        label = ":math:`Q_{max}`",
        default = numpy.array([0.250]),
        range = basic.Range(lo = 0.0, hi = 5.0),
        doc = """Maximum firing rate (/ms)""")
    
    speed = arrays.FloatArray(
        label = ":math:`v`",
        default = numpy.array([10.00]),
        range = basic.Range(lo = 0.0, hi = 15.0),
        doc = """Conduction velocity within cortex (mm/ms)""")
    
    mean_range = arrays.FloatArray(
        label = ":math:`r_e`",
        default = numpy.array([80.0]),
        range = basic.Range(lo = 0.0, hi = 5.0),
        doc = """Mean range of axons within cortex (mm)""")
    
    alpha = arrays.FloatArray(
        label = ":math:`\\alpha`",
        default = numpy.array([0.060]),
        range = basic.Range(lo = 0.0, hi = 5.0), 
        doc = """Inverse decay time of membrane potential... current values
            a=50; b=4*a; are consistent (/ms)""")
    
    beta = arrays.FloatArray(
        label = ":math:`\\beta`",
        default = numpy.array([0.24]),
        range = basic.Range(lo = 0.0, hi = 5.0),
        doc = """Inverse rise time of membrane
            potential, sensible value  4.0*alpha (/ms).""")
    
    nu_phi = arrays.FloatArray(
        label = ":math:`\\nu_{\\phi}`",
        default = numpy.array([17.0e2]),
        range = basic.Range(lo = 0.0, hi = 42.0e2),
        doc = """field gain/coupling (mV ms)""")
        
    nu_ei = arrays.FloatArray(
        label = ":math:`\\nu_{ei}`",
        default = numpy.array([-18.0e2]),  
        range = basic.Range(lo = -32.0e2, hi = -16.0e2),
        doc = """Inhibitory corticocortical gain/coupling (mV ms)""")
    
    nu_v = arrays.FloatArray(
        label = ":math:`\\nu_v`",
        default = numpy.array([-18.0e2]),  
        range = basic.Range(lo = -42.0e2, hi = 0.0),
        doc = """membrane gain/coupling (mV ms)""")
    
    nu_sn = arrays.FloatArray(
        label = ":math:`\\nu_{sn}`",
        default = numpy.array([10.0e2]),
        range = basic.Range(lo = 0.0, hi = 42.0e2),
        doc = """Nonspecific subthalamic input onto specific thalamic
            nuclei gain/coupling (mV ms)""")
    
    #Proposed informational attribute, used for phase-plane and initial()
    state_variable_range = arrays.FloatArray(
        label = "State Variables range [[lo], [hi]].",
        default = numpy.array([[-6.0, -3.0, -1.0, -3.0],
                               [6.0, 3.0, 1.0, 3.0]]),
        doc = """":math:`\\phi`: Field potential.
            :math:`d\\phi`: Field potential derivative.
            :math:`V`: Membrane potential.
            :math:`dV`: Membrane potential derivative.""")
#    
    variables_of_interest = arrays.IntegerArray(
        label = "Variables watched by Monitors.",
        range = basic.Range(lo = 0, hi = 3, step=1),
        default = numpy.array([0], dtype=numpy.int32),
        doc = """This represents the default state-variables of this Model to be
        monitored. It can be overridden for each Monitor if desired.""")
    
    
    def __init__(self, **kwargs):
        """
        init'n
        
        """
        
        LOG.info("%s: initing..." % str(self))
        
        super(AFR, self).__init__(**kwargs)
        
        self._state_variables = ["phi", "dphi", "V", "dV"]
        self._nvar = 4 #len(self._state_variables)
        self.cvar = numpy.array([0, 2], dtype=numpy.int32)
        
        self.gamma_e = None
        self.axb = None
        self.apb = None
        
        LOG.debug("%s: inited." % repr(self))
    
    
    def dfun(self, state_variables, coupling, local_coupling=0.0):
        """
        
        .. math::
            \\displaystyle \\frac{d \\phi}{dt} &= \\dot{\\phi} \\\\
            \\displaystyle \\frac{d \\dot{\phi}}{dt} &= \\gamma^2 (S(V(t-\\tau)) -
                \\phi) - 2 \\gamma \\dot{\\phi}  + v^2 \\nabla^2 \\phi \\\\
            \\displaystyle \\frac{d V}{dt} &= \\dot{V} \\\\
            \\displaystyle \\frac{d \\dot{V}}{dt} &=  \\alpha \\beta
                (\\nu_{\\phi}\\phi_e(t-\\tau) + \\nu_v V(t-\\tau) - V) -
                ( \\alpha + \\beta) \\dot{V}
        
        
        """
        phi = state_variables[0, :]
        dphi = state_variables[1, :]
        V = state_variables[2, :]
        dV = state_variables[3, :]
        
        #[State_variables, nodes]
        c_0 = coupling[0, :]
        c_1 = coupling[1, :]
        
        Fphi = dphi
        Fdphi = (self.gamma**2 * (c_1 - phi) - 2.0 * self.gamma * dphi +
                 self.speed**2 * (local_coupling * phi)) #local_coupling==>LapOp
        FV = dV
        FdV = self.axb * (self.nu_phi * c_0 + self.nu_ei * V + self.nu_v * c_1 - V) - self.apb * dV
        
        derivative = numpy.array([Fphi, Fdphi, FV, FdV])
        
        return derivative
    
    
    def update_derived_parameters(self):
        """
        Description of what we're precalculating...
        
        Include equations here...
        
        """
        
        self.gamma = self.speed / self.mean_range
        self.axb = self.alpha * self.beta
        self.apb = self.alpha + self.beta



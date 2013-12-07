# -*- coding: utf-8 -*-

"""
Implements Equations A1-A8 of [Breaketal2005]_, which represents the
corticothalamic model of [RRW1997]_. The spatial term is reintroduced on A2.                                          

.. [RRW1997] Robinson, P. A. and Rennie, C. J. and Wright, J. J., *Propagation
    and stability of waves of electrical activity in the cerebral cortex*,
    Phys. Rev. E},  56-1, p826--840, 1997.

.. [Breaketal2005] M. Breakspear, etal, *A Unifying Explanation of Primary
    Generalized Seizures Through Nonlinear Brain Modeling and Bifurcation
    Analysis.*, Cerebral Cortex, 2005. 


NOTE: Explicit inclusion of corticothalamic delays


.. moduleauthor:: Stuart A. Knock <stuart.knock@gmail.com>

"""

import numpy

#The Virtual Brain
from tvb.simulator.common import get_logger
LOG = get_logger(__name__)

import tvb.simulator.models as models
import tvb.basic.traits.types_basic as basic
import tvb.datatypes.arrays as arrays


class Sigmoid(object):
    """
    """
    
    def __init__(self, qmax, theta, sigma):
        """
        
        
        """
        self.qmax = qmax
        self.theta = theta
        self.sigma = sigma
        self.npionsqrt3 = -numpy.pi / numpy.sqrt(3.0)
    
    def __call__(self, x):
        """
        """
        val = self.qmax / (1.0 + numpy.exp(self.npionsqrt3 * (self.theta - x) / self.sigma))
        return val




class BRRW(models.Model):
    """
    
    .. automethod:: __init__
    
    """
    #TODO: Get the ranges from one of the papers...
    _ui_name = "Neural field model (Robinson etal)"
    
    #Define traited attributes for this model, these represent possible kwargs.
    theta_e = arrays.FloatArray(
        label = ":math:`\\theta_e`",
        default = 15.0,
        range = basic.Range(lo = 0.0, hi = 5.0),
        doc = """Mean neuronal threshold for Excitatory cortical population
            (mV).""")
         
    theta_s = arrays.FloatArray(
        label = ":math:`theta_s`",
        default =  15.0,
        range = basic.Range(lo = 0.0, hi = 5.0),
        doc = """Mean neuronal threshold for specific thalamic population
            (mV).""")
        
    theta_r = arrays.FloatArray(
        label = ":math:`\\theta_r`",
        default = 15.0,
        range = basic.Range(lo = 0.0, hi = 5.0),
        doc = """Mean neuronal threshold for reticular thalamic population
            (mV).""")
        
    sigma_e = arrays.FloatArray(
        label = ":math:`\\sigma_e`",
        default = 6.0,
        range = basic.Range(lo = 0.0, hi = 5.0),
        doc = """Threshold variability for Excitatory cortical population
            (mV).""")
        
    sigma_s = arrays.FloatArray(
        label = ":math:`\\sigma_s`",
        default = 6.0,   
        range = basic.Range(lo = 0.0, hi = 5.0),    
        doc = """Threshold variability for specific thalamic population
            (mV).""")
        
    sigma_r = arrays.FloatArray(
        label = ":math:`\\sigma_r`",
        default = 6.0,       
        range = basic.Range(lo = 0.0, hi = 5.0),
        doc = """Threshold variability for reticular thalamic population
            (mV).""")
        
    qmax = arrays.FloatArray(
        label = ":math:`Q_{max}`",
        default = 0.250,    
        range = basic.Range(lo = 0.0, hi = 5.0),
        doc = """Maximum firing rate (/ms)""")
        
    vel = arrays.FloatArray(
        label = ":math:`v`",
        default = 10.00,   
        range = basic.Range(lo = 0.0, hi = 5.0),
        doc = """Conduction velocity (mm/ms)""")
        
    r_e = arrays.FloatArray(
        label = ":math:`r_e`",
        default = 80.0,     
        range = basic.Range(lo = 0.0, hi = 5.0),  
        doc = """Mean range of axons (mm)""")
    
    alfa = arrays.FloatArray(
        label = ":math:`\\alpha`",
        default = 0.060,   
        range = basic.Range(lo = 0.0, hi = 5.0), 
        doc = """Inverse decay time of membrane potential... current values
            a=50; b=4*a; are consistent (/ms)""")
        
    btta = arrays.FloatArray(
        label = ":math:`\\beta`",
        default =   0.24, 
        range = basic.Range(lo = 0.0, hi = 5.0),
        doc = """Inverse rise time of membrane
            potential, sensible value  4.0*alfa (/ms).""")
        
    delay_ct = arrays.FloatArray(
        label = "Corticothalamic delay",
        default =  40.0,       
        range = basic.Range(lo = 0.0, hi = 5.0),
        doc = """Corticothalamic delay (ms)""")
        
    delay_tc = arrays.FloatArray(
        label = "Thalamocortical delay",
        default =  40.0,      
        range = basic.Range(lo = 0.0, hi = 5.0),
        doc = """Thalamocortical delay  (ms)""")
        
    nu_ee = arrays.FloatArray(
        label = ":math:`\\nu_{ee}`",
        default =  17.0e2,  
        range = basic.Range(lo = 0.0, hi = 5.0),
        doc = """Excitatory corticocortical gain/coupling (mV ms)""")
        
    nu_ei = arrays.FloatArray(
        label = ":math:`\\nu_{ei}`",
        default = -18.0e2,  
        range = basic.Range(lo = 0.0, hi = 5.0),
        doc = """Inhibitory corticocortical gain/coupling (mV ms)""")
        
    nu_es = arrays.FloatArray(
        label = ":math:`\\nu_{es}`",
        default = 12.0e2,  
        range = basic.Range(lo = 0.0, hi = 5.0),
        doc = """Specific thalamic nuclei to cortical gain/coupling
            (mV ms)""")
        
    nu_se = arrays.FloatArray(
        label = ":math:`\\nu_{se}`",
        default = 10.0e2,  
        range = basic.Range(lo = 0.0, hi = 5.0),
        doc = """Cortical to specific thalamic nuclei gain/coupling... turn
            seizure on and off (mV ms)""")
        
    nu_sr = arrays.FloatArray(
        label = ":math:`\\nu_{sr}`",
        default = -10.0e2,  
        range = basic.Range(lo = 0.0, hi = 5.0),
        doc = """Thalamic reticular nucleus to specific thalamic nucleus
            gain/coupling (mV ms)""")
        
    nu_sn = arrays.FloatArray(
        label = ":math:`\\nu_{sn}`",
        default = 10.0e2,  
        range = basic.Range(lo = 0.0, hi = 5.0),
        doc = """Nonspecific subthalamic input onto specific thalamic
            nuclei gain/coupling (mV ms)""")
        
    nu_re = arrays.FloatArray(
        label = ":math:`\\nu_{re}`",
        default = 4.0e2,  
        range = basic.Range(lo = 0.0, hi = 5.0),
        doc = """Excitatory cortical to thalamic reticular nucleus
            gain/coupling (mV ms)""")
        
    nu_rs = arrays.FloatArray(
        label = ":math:`\\nu_{rs}`",
        default = 2.0e2, 
        range = basic.Range(lo = 0.0, hi = 5.0), 
        doc = """Specific to reticular thalamic nuclei gain/coupling
            (mV ms)""")
    
    variables_of_interest = basic.Enumerate(
        label="Variables watched by Monitors",
        options=["phi_e", "dphi_e", "V_e", "dV_e", "V_s", "dV_s", "V_r", "dV_r"],
        default=["phi_e"],
        select_multiple=True,
        doc = """This represents the default state-variables of this Model to be
        monitored. It can be overridden for each Monitor if desired.""")
            
    #Proposed informational attribute, used for phase-plane and initial()
    state_variable_range = basic.Dict(
        label="State Variable ranges [lo, hi]",
        default={"phi_e": numpy.array([-6.0, 6.0]),
                 "dphi_e": numpy.array([-3.0, 3.0]),
                 "V_e": numpy.array([-1.0, 1.0]),
                 "dV_e": numpy.array([-3.0, 3.0]),
                 "V_s": numpy.array([-1.0, 1.0]),
                 "dV_s": numpy.array([-3.0, 3.0]),
                 "V_r": numpy.array([-1.0, 1.0]),
                 "dV_r": numpy.array([-3.0, 3.0])},
        doc = """":math:`\\phi_e`: Field potential, excitatory population.
            :math:`d\\phi_e`: Field potential derivative, excitatory population.
            :math:`V_e`: Membrane potential, excitatory population.
            :math:`dV_e`:Membrane potential derivative, excitatory population.
            :math:`V_s`: Membrane potential, 'specific' population.
            :math:`dV_s`: Membrane potential derivative, 'specific' population.
            :math:`V_r`: Membrane potential, 'reticular' population.
            :math:`dV_r`: Membrane potential derivative, 'reticular' population.""")
    
    
    
    def __init__(self, **kwargs):
        """
        May need to put kwargs back if we can't get them from trait...
        
        """
        
        LOG.info("%s: initing..." % str(self))
        
        super(BRRW, self).__init__(**kwargs)
        
        self._nvar = 8 #len(self._state_variables)
        self.cvar = numpy.array([0, 4], dtype=numpy.int32)
        
        
        self.axb = None
        self.apb = None
        
        self.sigmoidal_e = None
        self.sigmoidal_s = None
        self.sigmoidal_r = None
        
        LOG.debug("%s: inited." % repr(self))
    
    
    # def initial(self, dt, history_shape): #**kwargs
    #     """
    #     Set initial conditions to:
            
    #         .. math::
    #             V(0) &\\in \\left[-3.0, 3.0 \\right] \\\\
    #             W(0) &\\in \\left[-6.0, 6.0 \\right]
        
    #     """
    #     #TODO: Should make inital() parameterisable, with default to sensible
    #     #state variable ranges (defined as model traits), which could then also 
    #     #be used as default ranges in model tests via phase-plane... 
    #     initial_conditions = self.random_stream.uniform(size=history_shape)
    #     initial_conditions[:, 0, :] = ((initial_conditions[:, 0, :] * 
    #                                    (self.state_variable_range[1, 0] - 
    #                                     self.state_variable_range[0, 0])) +
    #                                    self.state_variable_range[0, 0])
    #     initial_conditions[:, 1, :] = ((initial_conditions[:, 1, :] * 
    #                                    (self.state_variable_range[1, 1] - 
    #                                     self.state_variable_range[0, 1])) +
    #                                    self.state_variable_range[0, 1])
    #     initial_conditions[:, 2, :] = ((initial_conditions[:, 2, :] * 
    #                                    (self.state_variable_range[1, 2] - 
    #                                     self.state_variable_range[0, 2])) +
    #                                    self.state_variable_range[0, 2])
    #     initial_conditions[:, 3, :] = ((initial_conditions[:, 3, :] * 
    #                                    (self.state_variable_range[1, 3] - 
    #                                     self.state_variable_range[0, 3])) +
    #                                    self.state_variable_range[0, 3])
    #     initial_conditions[:, 4, :] = ((initial_conditions[:, 4, :] * 
    #                                    (self.state_variable_range[1, 4] - 
    #                                     self.state_variable_range[0, 4])) +
    #                                    self.state_variable_range[0, 4])
    #     initial_conditions[:, 5, :] = ((initial_conditions[:, 5, :] * 
    #                                    (self.state_variable_range[1, 5] - 
    #                                     self.state_variable_range[0, 5])) +
    #                                    self.state_variable_range[0, 5])
    #     initial_conditions[:, 6, :] = ((initial_conditions[:, 6, :] * 
    #                                    (self.state_variable_range[1, 6] - 
    #                                     self.state_variable_range[0, 6])) +
    #                                    self.state_variable_range[0, 6])
    #     initial_conditions[:, 7, :] = ((initial_conditions[:, 7, :] * 
    #                                    (self.state_variable_range[1, 7] - 
    #                                     self.state_variable_range[0, 7])) +
    #                                    self.state_variable_range[0, 7])
        
    #     return initial_conditions
    
    
    def dfun(self, state_variables, coupling, local_coupling=0.0):
        """
        
        .. math::
            \\displaystyle \\frac{d \\phi_e}{dt} &= \\dot{\\phi_e} \\\\
            \\displaystyle \\frac{d \\dot{\phi_e}}{dt} &= \\gamma_e^2 (S(V_e) -
                \\phi_e) - 2 \\gamma_e \\dot{\\phi_e}  + v_e^2 \\nabla^2 \\phi_e \\\\
            \\displaystyle \\frac{d V_e}{dt} &= \\dot{V_e} \\\\
            \\displaystyle \\frac{d \\dot{V_e}}{dt} &=  \\alpha \\beta
                (\\nu_{ee}\\phi_e + \\nu_{ei} S(V_i) + \\nu_{es} S(V_s(t-\\tau))
                - V_e) - ( \\alpha + \\beta) \\dot{V_e} \\\\
            \\displaystyle \\frac{d V_s}{dt} &= \\dot{V_s} \\\\
            \\displaystyle \\frac{d \\dot{V_s}}{dt} &=  \\alpha \\beta
                (\\nu_{sn}\\phi_n + \\nu_{sr} S(V_r) + \\nu_{se}\\phi_e(t-\\tau)
                - V_s)-( \\alpha + \\beta) \\dot{V_s} \\\\
            \\displaystyle \\frac{d V_r}{dt} &= \\dot{V_r} \\\\
            \\displaystyle \\frac{d \\dot{V_r}}{dt} &= \\alpha \\beta
                (\\nu_{re}\\phi_e(t-\\tau) + \\nu_{rs} S(V_s) - V_r)-( \\alpha +
                \\beta) \\dot{V_r}
        
      

        """
        phi_e = state_variables[0, :]
        dphi_e = state_variables[1, :]
        V_e = state_variables[2, :]
        dV_e = state_variables[3, :]
        V_s = state_variables[4, :]
        dV_s = state_variables[5, :]
        V_r = state_variables[6, :]
        dV_r = state_variables[7, :]
        
        #[State_variables, nodes]
        c_0 = coupling[0, :]
        c_1 = coupling[1, :]
        
        #[State_variables, nodes, nodes] (sparse)
        #lc_0 = local_coupling[0, :]
        #lc_1 = local_coupling[1, :]
        
        sig_ve = self.sigmoidal_e(V_e) #,P.Qmax,P.Theta_e,P.sigma_e);
        
        #TODO: phi_n: replaceable with state variable specific nsig support:  self.axb *  self.nu_sn * self.phi_n[k,:]
        
        #TODO: Need to specify nu_* and V_* in terms of reggion...
        
        #TODO: Make two models, one as is with cortical region only restriction
        #      and one more like AFR (see TODO above)
        
        Fphi_e = dphi_e
        Fdphi_e = self.gamma_e**2 * (sig_ve - phi_e) - 2.0 * self.gamma_e * dphi_e + self.vel**2 * (local_coupling * phi_e)                 #local_coupling==>LapOp
        FV_e = dV_e
        FdV_e = self.axb * (self.nu_ee * phi_e  + self.nu_ei * sig_ve + self.nu_es * self.sigmoidal_s(c_1) - V_e) - self.apb * dV_e         #,self.qmax,self.theta_s,self.sigma_s
        FV_s = dV_s
        FdV_s = self.axb * (self.nu_se * c_0 + self.nu_sr * self.sigmoidal_r(V_r) - V_s) - self.apb * dV_s   #,self.qmax,self.theta_r,self.sigma_r
        FV_r = dV_r
        FdV_r = self.axb * (self.nu_re * c_0 + self.nu_rs * self.sigmoidal_s(V_s) - V_r) - self.apb * dV_r                                  #,self.qmax,self.theta_s,self.sigma_s

        
        derivative = numpy.array([Fphi_e, Fdphi_e, FV_e, FdV_e, FV_s, FdV_s, FV_r, FdV_r])
        
        return derivative
        
        
    def update_derived_parameters(self):
        """
        Description of what we're precalculating...
        
        Include equations here...
        
        """
        
        self.gamma_e = self.vel / self.r_e
     
        self.axb = self.alfa * self.btta
        self.apb = self.alfa + self.btta
        ###self.dtcsf = options.Integration.dt * self.csf
        
        
        self.sigmoidal_e = Sigmoid(self.qmax, self.theta_e, self.sigma_e)
        self.sigmoidal_s = Sigmoid(self.qmax, self.theta_s, self.sigma_s)
        self.sigmoidal_r = Sigmoid(self.qmax, self.theta_r, self.sigma_r)
        
    

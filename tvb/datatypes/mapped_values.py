# -*- coding: utf-8 -*-
#
#
# (c)  Baycrest Centre for Geriatric Care ("Baycrest"), 2012, all rights reserved.
#
# No redistribution, clinical use or commercial re-sale is permitted.
# Usage-license is only granted for personal or academic usage.
# You may change sources for your private or academic use.
# If you want to contribute to the project, you need to sign a contributor's license. 
# Please contact info@thevirtualbrain.org for further details.
# Neither the name of Baycrest nor the names of any TVB contributors may be used to endorse or 
# promote products or services derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY BAYCREST ''AS IS'' AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, 
# BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
# ARE DISCLAIMED. IN NO EVENT SHALL BAYCREST BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS 
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY 
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
#
#
"""
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

from tvb.basic.traits.types_mapped import MappedType
from tvb.basic.traits.types_basic import JSONType, String, Dict
from tvb.datatypes.time_series import TimeSeries


# Accepted Value Types to be stored.
ACCEPTED_TYPES = ['float', 'int']


class ValueWrapper(MappedType):
    """
    Class to wrap a singular value storage in DB.
    """
    
    data_value = JSONType()
    data_type = String(default='unknown')  
    data_name = String() 
    
    @property
    def display_name(self):
        """ Simple String to be used for display in UI."""
        return "Value Wrapper - " + self.data_name +" : "+ str(self.data_value) + " ("+ str(self.data_type)+ ")"
            
    
class DatatypeMeasure(MappedType):
    """
    Class to hold the metric for a previous stored DataType.
    E.g. Measure (single value) for any TimeSeries resulted in a group of Simulations
    """
    ### Actual measure (dictionary Algorithm: single Value)
    metrics = Dict
    ### DataType for which the measure was computed.
    analyzed_datatype = TimeSeries
    
    
    @property
    def display_name(self):
        """
        To be implemented in each sub-class which is about to be displayed in UI, 
        and return the text to appear.
        """
        name = "-"
        if self.metrics is not None:
            value = "\n"
            for entry in self.metrics:
                value = value + entry + ' : ' + str(self.metrics[entry]) + '\n'
            name = value
        return name
    
    
    
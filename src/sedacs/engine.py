"""engine
This module will be used to store information about the code
we will be interfacing to.

"""

__all__ = ["Engine"]


## Engine type
# @brief This will contain the information that sedacs needs
# in order to use an external quantum chemistry code.
#
class Engine:
    """A prototype for the engine type."""

    def __init__(self, idIn):
        ## Name of the engine
        self.name = "ProxyA"
        ## Interface type
        self.interface = "None"
        ## Engine path files. Used to interchange data
        self.path = "/tmp/engine"
        ## Engine ID. A number to identify which MPI rank is the engine executed from.
        self.id = idIn
        ## Engine execution file absolute path
        self.run = "/home/engine/engine.py"
        ## Engine status. A logical variable to check the status of the engine.
        self.up = False


### Engine input reader. This will use the parser module.
## @brief This will be used to read the
## input variables used in the code.
##
# def sdc_engine_input(fileName,engine):
#    """Simple engine input parser
#    """
#        ## Keys and values read from the input file
#        keyVals = sdc_input.get_all_vals(fileName)
#        ## Engine data
#        engine_dict = sdc_input.get_a_dict("Engine=",{"Name":"MyEngine","InterfaceType":"Files",
#            "Path":"/tmp/","Executable":"/tmp/run"},keyVals,verb)
#        engine.name = engine_dict["Name"]
#        engine.path = engine_dict["Path"]
#        engine.run = engine_dict["Executable"]

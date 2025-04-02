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
        ## Method or specific density matrix solver
        self.method = "Diag" 
        ## Accelerator (library or specific computer architecture) 
        self.accel = "No"





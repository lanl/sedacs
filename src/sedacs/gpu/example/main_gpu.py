#!/usr/bin/env python3                                                                                                  
""" Main gpu driver                                                                                  
"""  
from sedacs.system import *                                                                                                    
from proxy_a import *                                                                                                       
from sedacs.parser import *

sdc = Input("input.in",True)

ham = sdc_proxya()

#!/usr/bin/env python3                                                                                                  
""" Main gpu driver                                                                                  
"""  
from sedacs.system import *                                                                                                    
from proxy_a import *                                                                                                       
from sedacs.parser import *

sdc = sdc_input("input.in",True)

ham = sdc_proxya()

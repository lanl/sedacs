#!/usr/bin/env python3                                                                                                  
""" Main gpu driver                                                                                  
"""  
from sdc_system import *                                                                                                    
from proxy_a import *                                                                                                       
from sdc_parser import *

sdc = sdc_input("input.in",True)

ham = sdc_proxya()

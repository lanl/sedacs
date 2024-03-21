""" Standarized messages.
 
 This will provide some standard messages used throughout the code
"""

import sys

class textcolor:
    green = '\033[92m'
    yellow = '\033[93m'
    red = '\033[91m'
    normal = '\033[0m'


def sdc_error_at(name):
    print("\n !!!ERROR at",name,"\n")
    exit(0)

def sdc_warning_at(name):
    print("\n !!!WARNING at",name,"\n")

def sdc_test_fail(name):
    print("  Test for ",name,"... " + textcolor.red + "Failed" + textcolor.normal)

def sdc_test_pass(name):
    print("  Test for ",name,"... " + textcolor.green + "Passed" + textcolor.normal)

def sdc_fail_at(name):
    print("\n !!!ERROR at",name,"\n")
    exit(1)

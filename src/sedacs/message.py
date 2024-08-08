"""Standarized messages.

This will provide some standard messages used throughout the code
"""

import sys

__all__ = ["status_at", "error_at", "warning_at", "sdc_test_fail", "sdc_test_pass"]

import warnings

class TextColor:
    green = "\033[92m"
    yellow = "\033[93m"
    red = "\033[91m"
    normal = "\033[0m"

def status_at(name,msg):
    print(TextColor.green, "\n >>> STATUS at",name,TextColor.normal," ",msg, "...\n")

def error_at(name, message=None):
    msg = "\n" + TextColor.red  + "### ERROR at:" + TextColor.normal + " " + name + " "
    if message is not None:
        msg += TextColor.normal + message + "\n"
    raise RuntimeError(msg)

def warning_at(name, msg):
    print(TextColor.yellow, "\n !!! WARNING at", name,TextColor.normal," ",msg, "...\n")

def sdc_test_fail(name):
    print("  Test for ", name, "... " + TextColor.red + "Failed" + TextColor.normal)

def sdc_test_pass(name):
    print("  Test for ", name, "... " + TextColor.green + "Passed" + TextColor.normal)


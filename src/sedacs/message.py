"""Standarized messages.

This will provide some standard messages used throughout the code
"""

import sys

__all__ = ["sdc_status_at", "sdc_error_at", "sdc_warning_at", "sdc_fail_at", "sdc_test_fail", "sdc_test_pass"]


class TextColor:
    green = "\033[92m"
    yellow = "\033[93m"
    red = "\033[91m"
    normal = "\033[0m"


def sdc_status_at(name):
    print("\n >>> At ", name, "...\n")


def sdc_error_at(name, message=None):
    print("\n !!!ERROR at", name, "\n")
    if message is not None:
        print("    ", message, "\n")
    sys.exit(0)


def sdc_warning_at(name):
    print("\n !!!WARNING at", name, "\n")


def sdc_fail_at(name):
    print("\n !!!ERROR at", name, "\n")
    sys.exit(1)


def sdc_test_fail(name):
    print("  Test for ", name, "... " + TextColor.red + "Failed" + TextColor.normal)


def sdc_test_pass(name):
    print("  Test for ", name, "... " + TextColor.green + "Passed" + TextColor.normal)

"""Standarized messages.

This will provide some standard messages used throughout the code
"""

import sys

__all__ = ["status_at", "error_at", "warning_at", "sdc_test_fail", "sdc_test_pass"]

import warnings

class TextColor:
    """
    Text color class.
    """
    green = "\033[92m"
    yellow = "\033[93m"
    red = "\033[91m"
    normal = "\033[0m"

def status_at(name: str,
              msg: str) -> None:
    """
    Print a status message.

    Parameters
    ----------
    name : str
        The name of the function.
    msg : str
        The message to print.

    Returns
    -------
    None
    """
    print(TextColor.green, "\n >>> STATUS at",name,TextColor.normal," ",msg, "...\n")

def error_at(name, message=None):

    """
    Raises a runtime error with the specified output.

    Parameters
    ----------
    name : str
        Error specification.
    message : str
        Additional information to print.

    Returns
    -------
    None    
    """

    msg = "\n" + TextColor.red  + "### ERROR at:" + TextColor.normal + " " + name + " "
    if message is not None:
        msg += TextColor.normal + message + "\n"
    raise RuntimeError(msg)

def warning_at(name: str,
               msg: str) -> None:
    """
    Print a warning message.

    Parameters
    ----------
    name : str
        Error specification.
    msg : str
        Additional information to print.

    Returns
    -------
    None
    """

    print(TextColor.yellow, "\n !!! WARNING at", name,TextColor.normal," ",msg, "...\n")

def sdc_test_fail(name: str) -> None:
    """
    Print a failed test message.

    Parameters
    ----------
    name : str
        The name of the test.   

    Returns
    -------
    None
    """

    print("  Test for ", name, "... " + TextColor.red + "Failed" + TextColor.normal)

def sdc_test_pass(name: str) -> None:
    """
    Print a passed test message.

    Parameters
    ----------
    name : str
        The name of the test.   
    
    Returns
    -------
    None
    """ 

    print("  Test for ", name, "... " + TextColor.green + "Passed" + TextColor.normal)


"""simple test runner
This is used to run individual tests
"""

from sedacs.system import *
from sedacs.periodic_table import *
from sedacs.graph_partition import *


## For coloring text
class textcolors:
    green = "\033[92m"
    yellow = "\033[93m"
    red = "\033[91m"
    normal = "\033[0m"


## Function to run the tests
# @brief General function to run a test
# @param testName Name of the routine to be tested
#
def run_my_test(testName):
    testFunctionName = "test_" + testName
    passed = False
    exit1 = True
    function = globals()[testFunctionName]
    passed = function(exit1)
    return passed, exit1


# List of all the name of the routins to be tested
# make a list using system calls based on the grep results
if len(sys.argv) <= 1:
    print("\nNo test was passed ...")
    print("\nUssage:")
    print("\n          ./sdc_test_runner.py <test_name>\n")
    exit(0)

testName = sys.argv[1]

print("\nRunning tests ...", testName)

# Run the tests
passed, exit1 = run_my_test(testName)

# Print message
if passed:
    sdc_test_pass(testName)
else:
    sdc_test_fail(testName)
    if exit1:
        exit(1)

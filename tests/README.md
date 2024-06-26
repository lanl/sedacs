## Tests for SEDACS

### Running the tests

To run the tests we have a bash script that must be executed as follows:

```shell
./build_and_run_tests.sh <python/ctest> 
```
The two options python and ctest can be used. This script call a python 
scrit `sdc_test_runer.py` which can run a test a a time. 

If only one test needs to be run, we can then do the following:

```shell 
python3 ./sdc_test_runer.py <test_name>
```

### Adding a test 

A test for any function can be added simply by prefixing the word test followed by the function name. An example follows: 

```python 

def test_my_function(exit1):

	#Prepare test paramters
	#and reference 
	passed = True
	
	#Conditions for passing
	try:
		#Call the function
		result = my_function(...) 
		if(result != reference):
			passed = False
	except:
		passed = False

	#Option to fail execution
	if(exit1 and (not passed)): exit(1)	
	
	return passed 

```



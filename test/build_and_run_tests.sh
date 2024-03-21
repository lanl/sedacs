#Select either python or ctest
if [ "$1" = "" ]
then 
  echo "Ussage:"
  echo " ./build_and_run_tests.sh <options>"
  echo "Options are ctest and python"
  exit
fi

#Capturing all the tests from the module folder
test_list=`grep test_ ../mods/*.py |grep -v sdc_test | grep def |awk {'print $2'} | sed -n 's/(.*$//p'` 


if [ "$1" = "ctest" ]
then
  #The following will build a CMakeList.txt file
  echo "project(SEDACS)" > CMakeLists.txt
  echo "cmake_minimum_required(VERSION 3.16)" >> CMakeLists.txt
  echo "enable_testing()" >> CMakeLists.txt
  echo "find_package(Python REQUIRED)" >> CMakeLists.txt
  echo 'set(bin ${CMAKE_CURRENT_SOURCE_DIR}/build/)' >> CMakeLists.txt
  echo 'set(CMAKE_BUILD_DIRECTORY ${bin})' >> CMakeLists.txt
  echo 'set(mods ${CMAKE_CURRENT_SOURCE_DIR}/../mods/)' >> CMakeLists.txt
fi

for testName in $test_list 
do
  testName="${testName:5}"
  if [ "$1" = "ctest" ]
  then
    echo 'add_test('$testName' ${Python_EXECUTABLE} sdc_test_runner.py '$testName')' >> CMakeLists.txt
  else
    python sdc_test_runner.py $testName
  fi
done

if [  "$1" = "ctest" ] 
then 
  cmake ./ ; make test
fi


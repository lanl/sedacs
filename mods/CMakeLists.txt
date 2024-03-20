project(SEDACS)
cmake_minimum_required(VERSION 3.16)
enable_testing()
find_package(Python REQUIRED)

#Test system module
add_test(write_xyz_coordinates ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/sdc_system.py test write_xyz_coordinates)
add_test(read_xyz_file ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/sdc_system.py test read_xyz_file)
add_test(read_pdb_file ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/sdc_system.py test read_pdb_file)
add_test(write_pdb_file ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/sdc_system.py test write_pdb_file)
add_test(parameters_to_vectors ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/sdc_system.py test parameters_to_vectors)
add_test(get_volBox ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/sdc_system.py test get_volBox)

#Test ptable
add_test(ptable ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/sdc_ptable.py test ptable)

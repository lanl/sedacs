#!/bin/bash

#Make sure to source the environmental variable 
#sour vars

sdc_input=$'Threshold= 1.0E-5 \n CoordsFile= \n Rcut= 5.0 '

export OMP_NUM_THREADS=1

run="python sdc_main_nl_test.py"

tag="Time for build_nlist" 

fileout="times_nl.dat"
rm $fileout 

for numranks in 1 2 4 
do
  echo " "${numranks}  >> $fileout
  echo "#"${numranks}  >> $fileout
  for coordsSize in 5133  10000 20000 30000 65000 
  do
    echo "coords, numranks:" "$coordsSize" "$numranks"
    coordsFile="coords_"$coordsSize".pdb"
    echo "$sdc_input" > tmp
    echo "$sdc_input" 
    sed 's/CoordsFile=.*/CoordsFile= '$coordsFile'/g' tmp > input.in
    PYTHONPATH="../mods:../proxya/python_proxy" mpirun -np $numranks python sdc_main_nl_test.py --use-torch | tee  out$coordsSize$numranks
#    PYTHONPATH="../mods:../proxya/python_proxy" mpirun -np $numranks python sdc_main_nl_test.py | tee  out$coordsSize$numranks
    time=`grep -e $tag out$coordsSize$numranks | head -1 | awk '{print $4}'`
    echo $coordsSize $time >> $fileout
  done
done


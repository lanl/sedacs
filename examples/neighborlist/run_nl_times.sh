#!/bin/bash

#Make sure to source the environmental variable 
#sour vars


export OMP_NUM_THREADS=1

run="python main.py"

tag="build_nlist" 

fileout="times_nl.dat"
rm $fileout 

for numranks in 1 2 4 
do
  echo " "${numranks}  >> $fileout
  echo "#"${numranks}  >> $fileout
  for coordsSize in 5133  10000 20000 30000 65000 
  do
    echo "coords, numranks:" "$coordsSize" "$numranks"
    coordsFile="$PWD/../../data/driver/coords_"$coordsSize".pdb"
    sdc_input=$'Threshold= 1.0E-5 \n CoordsFile= '"$coordsFile"$' \n Rcut= 5.0 '
    echo "$sdc_input" > input.in 
    echo "$sdc_input" 
    mpirun -np $numranks python3 main.py | tee  out$coordsSize$numranks
    time=`grep -e $tag out$coordsSize$numranks | head -1 | awk '{print $4}'`
    echo $coordsSize $time >> $fileout
  done
done


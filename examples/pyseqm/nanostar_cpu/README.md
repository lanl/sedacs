### `PySEQM`
PySEQM works with finite systems. Therefore, systems are placed inside boxes with large vacuum gaps.

In ```main.py```, change ```proxya_path``` and ```pyseqm_path```.
In ```input.in```, change ```Path``` and ```Executable``` paths.

Set OMP_NUM_THREADS to a desired number
```shell
cd JOB_DIRECTORY
source activate sedacs
export OMP_NUM_THREADS=4
```
Then run the calculation on two ranks. 
```shell
mpirun -bind-to none -n 2 python -u main.py > out.out 2>&1
```
Nanostar will be partitioned into 4 parts, as specified by ```NumParts``` keyword. Each rank will process 2 parts.
Note that NumParts in the input file must be devisible by a number of ranks ```-n #```.
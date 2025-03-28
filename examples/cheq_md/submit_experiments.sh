#!/bin/bash
model_folder="./model_data"
N_vals=("6540" "10008" "25050" "52320")
use_ewald="1"
precision="FP64"
use_shadow="1"
rtol="1e-1"
dt="0.2"
cutoff="10.0"
ewald_method="Ewald"

# dublicates in x dimension, if we want to create even larger systems
# it dublicates the geo and the saved snapshot.
for dx in "1"; do
prefix="test"

# Loop through the arrays and submit a job for each combination
#for i in "${!N_vals[@]}"; do
for i in 0; do
    N=${N_vals[$i]}
    echo $use_shadow
    echo $use_ewald

    # Create a unique job script for each job
    job_file="jobbb_${i}_${prefix}_${rtol}_${use_shadow}_${ewald_method}_${precision}_${dx}.slurm"
    output_file="jobbb_${i}_${prefix}_${rtol}_${use_shadow}_${ewald_method}_${precision}_${dx}_out.txt"

    # Write the SLURM job script with the new parameters
    cat > $job_file <<EOL
#!/bin/bash --login
#SBATCH --time=08:55:00             # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --nodes=1         # number of different nodes - could be an exact number or a range of nodes (same as -N)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8        # number of CPUs (or cores) per task (same as -c)
#SBATCH --tasks-per-node=1
#SBATCH --mem=32G            # memory required
#SBATCH --job-name MD_$prefix      # you can give your job a name for easier identification (same as -J)
#SBATCH --partition volta-x86 #shared-gpu-ampere
#SBATCH --output=$output_file     # Standard output and error will be written to this file
#SBATCH --error=$output_file      # Combine both stdout and stderr into the same file


source activate sedacs
python3 run_MD.py --model_folder $model_folder \\
                  --N $N \\
                 --use_shadow $use_shadow --rtol $rtol \\
                 --use_ewald $use_ewald \\
                 --prefix $prefix \\
                 --sim_length 10000 \\
                 --restart 1 \\
                 --is_nvt 0 \\
                 --ewald_method $ewald_method \\
                 --dt $dt \\
                 --cutoff $cutoff \\
                 --precision $precision \\
                 --ewald_err 5e-4 \\
                 --use_jacobi_precond 1 \\
                 --buffer 1.0 \\
                 --dx ${dx} \\
EOL

    # Submit the job
    sbatch $job_file
    echo "Submitted job $job_file"
done
done



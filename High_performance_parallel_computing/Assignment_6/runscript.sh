#!/usr/bin/env bash

job_script_base="job_script"

for ntasks in {1..512..64}
do
    nodes=$((ntasks / 64 + 1))
    job_script="${job_script_base}_${ntasks}.sh"
    output_file="output_${ntasks}_tasks.txt"
    
    cat <<EOF > "$job_script"
#!/usr/bin/env bash
#SBATCH --job-name=TaskFarm_${ntasks}
#SBATCH --partition=modi_HPPC
#SBATCH --exclusive

mpiexec apptainer exec \
   ~/modi_images/ucphhpc/hpc-notebook:latest \
   ./fwc_parallel_2D --iter 1000 --model models/large.hdf5
EOF
    sbatch --ntasks="$ntasks" --nodes="$nodes" --output="$output_file" "$job_script"
    sleep 0.3
    rm "$job_script"
done

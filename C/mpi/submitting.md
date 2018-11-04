# Submitting jobs to supercomputers

Practically you won't be issuing mpirun commands on a couple of GCP machines. You will probably be running your code on a cluster. Exactly how you sumbit jobs will vary but here's an example:

## Slurm on Edison (nersc)

See [nersc docs](http://www.nersc.gov/users/computational-systems/edison/getting-started/your-first-program/).

```
#!/bin/bash -l
#SBATCH -p debug
#SBATCH -N 2
#SBATCH -t 30:00
#SBATCH -J my_job

srun -n 48 ./a.out
```

There are a number of SBATCH flags, see
[Slurm docs][https://slurm.schedmd.com/sbatch.html]

Your code will wait in queue `p`. When it is scheduled `./a.out` will run on `n` cores spread out over `N` nodes (so you need to know your machine's architecture). If it runs for time `t` it will be killed (if it runs for less than `t` you will only be billed for how long it actually runs). Unclear whether this time is in walltime or processor time or what. `J` is the job name.

Submit this job with `sbatch submit_scipt.sh`.

Check the queue with `squeue -u <username>`.

Stdout will appear in a `slurm-<job_id>.out` file.

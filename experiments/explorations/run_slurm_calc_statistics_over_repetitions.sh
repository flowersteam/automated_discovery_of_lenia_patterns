#!/bin/bash

echo "Start experiments via slurm ..."
python -c "import exputils
exputils.start_slurm_experiments(directory='./experiments/', 
				 start_scripts='run_calc_statistics_over_repetitions.slurm', 
				 is_parallel=True,
				 verbose=True,
				 post_start_wait_time=0.5)"

echo "Finished"


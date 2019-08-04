#!/bin/bash

echo "Start experiments via slurm ..."
python -c "import exputils
exputils.start_experiments(directory='./experiments/', 
				 start_scripts='run_calc_statistics_per_repetition.slurm', 
				 is_parallel=False, 
				 is_chdir=True,
				 verbose=True,
				 post_start_wait_time=0.5)"

echo "Finished"


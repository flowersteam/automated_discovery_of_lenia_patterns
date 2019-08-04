#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=inria
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --input=none
#SBATCH --input=none

STATUSFILE=run_calc_statistics_over_repetitions.slurm.status

# Uncomment if needed on a cluster
#echo "Activate autodisc conda environment ..."
#source ~/anaconda3/bin/activate autodisc


echo "Run ..."
STATE='Running'

date "+%Y/%m/%d %H:%M:%S" >> $STATUSFILE
echo $STATE >>  $STATUSFILE

python calc_statistics_over_repetitions.py
RETURN_CODE=$?

echo "Write status file ..."
if [ $RETURN_CODE == 0 ] 
then
	STATE='Finished'
else
	STATE='Error'
fi

date "+%Y/%m/%d %H:%M:%S" >> $STATUSFILE
echo $STATE >> $STATUSFILE

echo "Finished."


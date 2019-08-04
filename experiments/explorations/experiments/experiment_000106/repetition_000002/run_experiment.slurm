#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=inria
#SBATCH --mem-per-cpu=5G
#SBATCH --time=24:00:00
#SBATCH --input=none

STATUSFILE=run_experiment.slurm.status

# Uncomment if needed on a cluster
#echo "Activate autodisc conda environment ..."
#source ~/anaconda3/bin/activate autodisc


echo "Run the experiment ..."
STATE='Running'

date "+%Y/%m/%d %H:%M:%S" >> $STATUSFILE
echo $STATE >>  $STATUSFILE

python run_experiment.py
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



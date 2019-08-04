#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --partition=inria
#SBATCH --time=72:00:00
#SBATCH --input=none
#SBATCH -o logs/console.out
#SBATCH -e logs/error.out

STATUSFILE=run_experiment.slurm.status

# Uncomment if needed on a cluster
#echo "Activate autodisc conda environment ..."
#source ~/anaconda3/bin/activate autodisc


echo "Run the experiment ..."
STATE='Running'
date "+%Y/%m/%d %H:%M:%S" >> $STATUSFILE
echo $STATE >>  $STATUSFILE

python run_experiment.py > logs/experiment.out
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


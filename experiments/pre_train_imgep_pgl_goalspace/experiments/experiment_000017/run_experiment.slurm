#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=72:00:00
#SBATCH --input=none
#SBATCH --mail-user=mayalen.etcheverry@inria.fr
#SBATCH -o logs/console.out
#SBATCH -e logs/error.out

echo "Activate autodisc conda environment ..."
source /gpfs/home/metchever010/anaconda3/bin/activate autodisc

echo "Run the experiment ..."
python run_training.py > logs/training.out
RETURN_CODE=$?

if [ $RETURN_CODE == 0 ] 
then
	echo "Run the testing: calculation of post statistics ..."
	python run_testing.py > logs/testing.out

	RETURN_CODE=$?
fi

echo "Write status file ..."
if [ $RETURN_CODE == 0 ] 
then
	STATE='Finished'
else
	STATE='Error'
fi

date "+%Y/%m/%d %H:%M:%S" >> run_experiment.slurm.status
echo $STATE >>  run_experiment.slurm.status

echo "Finished."


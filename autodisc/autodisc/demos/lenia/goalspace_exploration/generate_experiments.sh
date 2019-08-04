#!/bin/bash 

echo "Activate autodisc conda environment ..."
source /home/creinke/anaconda3/bin/activate autodisc

echo "Generate experiments ..."
python -c "import exputils
exputils.generate_experiment_files('experiment_configurations.ods', directory='./experiments/')"

echo "Finished."

$SHELL

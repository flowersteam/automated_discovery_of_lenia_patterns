#!/bin/bash 

echo "Generate experiments ..."
python -c "import exputils
exputils.generate_experiment_files('experiment_configurations.ods', directory='./experiments/')"

echo "Finished."

$SHELL

#!/bin/bash

STATUSFILE=run_calc_statistics_over_repetitions.sh.status

echo "Activate autodisc conda environment ..."
source ~/anaconda3/bin/activate autodisc

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


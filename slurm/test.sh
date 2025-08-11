#!/bin/sh
# https://stackoverflow.com/questions/27708656/pass-command-line-arguments-via-sbatch

export HYDRA_FULL_ERROR=1
export TQDM_LOG=1
export TQDM_LOG_INTERVAL=100

if [ ${1} = 'boundaries' ]; then
  ser=proposal
else
  ser=default
fi


old="$IFS"
IFS='_'
str="$*"

val="splitlearning_${str}"

echo "${ser}"

IFS=$old

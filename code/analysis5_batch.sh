#!/bin/bash

## Yale Grace batch script
##
## This script runs a python command
##


#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=sgd_1.35
#SBATCH --array=1-20
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=1
#SBATCH --constraint=doubleprecision
#SBATCH --mem=16000
#SBATCH --time=0-09:00:00

### Interactive command: srun --partition=gpu --gres=gpu:1 -A anticevic --time=1-00:00:00 --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8000 --pty bash -i

echo $SLURM_ARRAY_TASK_ID
# Orig
#python analysis5_model_traindat1dat2_richVsLazyLearning.py --weight_init 1.0 --bias_init 0 --nhidden 500 --relu --normalize --cuda --outfilename analysis5_run${SLURM_ARRAY_TASK_ID}

##### Parameter sweep
#parameter_sweep="0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0 2.2"
#for param in $parameter_sweep
#do
#    #python analysis5_model_traindat1dat2_richVsLazyLearning.py --weight_init $param --bias_init 0 --nhidden 1000 --relu --normalize --cuda --outfilename analysis5_run${SLURM_ARRAY_TASK_ID} # with relu
#    python analysis5_model_traindat1dat2_richVsLazyLearning.py --weight_init $param --bias_init 0 --nhidden 1000 --normalize --cuda --outfilename analysis5_run${SLURM_ARRAY_TASK_ID} # linear
#done


## Manuscript version
python analysis5_model_traindat1dat2_richVsLazyLearning.py --weight_init 2.4 --bias_init 0 --nhidden 500 --normalize --cuda --outfilename analysis5_run${SLURM_ARRAY_TASK_ID}


## Manuscript Supplement(UNTIED model version + LINEAR (SFIG))
#python analysis5_model_traindat1dat2_richVsLazyLearning.py --weight_init 2.0 --bias_init 0 --nhidden 250 --nlayers 5 --normalize --untied --cuda --outfilename analysis5_run${SLURM_ARRAY_TASK_ID}


#### REVISION -- try with different learning optimizer/ learning rates
# SGD
#python analysis5_model_traindat1dat2_richVsLazyLearning.py --weight_init 1.35 --bias_init 0 --nhidden 500 --nlayers 5 --normalize --cuda --outfilename analysis5_run${SLURM_ARRAY_TASK_ID} --optim 'sgd' --learning_rate 0.01

sacct --units=G --format=MaxRSS,MaxDiskRead,MaxDiskWrite,Elapsed,NodeList -j $SLURM_JOBID


#!/bin/bash

#SBATCH -p amd-ep2,amd-ep2-short,intel-sc3
#SBATCH -q normal
#SBATCH -c 16 
#SBATCH --mem 32G
#SBATCH -o %j_%a.log


SCRIPT_DIR=$1
TYPE=$2
DIR=$3
ID=${4:-$SLURM_ARRAY_TASK_ID}


#DATA_PATH="/storage/gaoxiaofeiLab/yaoxingyun/temp_dir/"
DATA_PATH="$DIR/tmp_dir/"
dir_local="$DATA_PATH"$$               #tmp directory private to specific excute compute node
mkdir -p $dir_local              #$$ refer to  unique pid of spccific job

source ~/.bashrc
conda activate lucky

export LC_ALL=C

TMP_DIR="/dev/shm/$$"
mkdir $TMP_DIR -p

cd $TMP_DIR

cp -sn $DIR/train.tab.*[0-9].out.tab .
cp -v $DIR/all.$TYPE.sample.info.$ID ./all.sample.info

ls train.tab.*[0-9].out.tab  | xargs -n 1 -P ${SLURM_CPUS_ON_NODE:-32} -I {} python $SCRIPT_DIR/dim_reduction_single_step.py {} > /dev/null 2>> error.log
cat error.log
cat train.tab.*[0-9].out.tab.bed > $DIR/all.$TYPE.tab.$ID

cd $DIR
$SCRIPT_DIR/bed_select all.$TYPE.tab.$ID all.$TYPE.tab.$ID.out 1000
echo $TMP_DIR
rm -rf $TMP_DIR
echo "$ID finished"

rm -rf $dir_local

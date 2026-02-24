#!/bin/bash

#SBATCH -p amd-ep2,amd-ep2-short,intel-sc3
#SBATCH -q huge
#SBATCH -c 16
#SBATCH --mem 32G
#SBATCH -o gam_%A_%a.log
#SBATCH --requeue              # 允许任务重新排队
# 定义检查函数

check_data_access() {
    if ! cd "/data/" 2>/dev/null; then
        echo "错误: 无法进入/data/目录"
        return 1
    fi

    # 返回原目录（可选）
    cd - >/dev/null 2>&1
    return 0
}

# 执行检查
if ! check_data_access; then
    CURRENT_NODE=$(hostname -s)
    echo "/data/ 挂载检查失败，将重新排队并排除节点 $CURRENT_NODE"
    
    # 将当前节点添加到排除列表
    scontrol requeue $SLURM_JOB_ID
    exit 1
fi

SCRIPT_DIR=$1
WORKING_DIR=$2
FILE_NAME="train.tab.$SLURM_ARRAY_TASK_ID"
DATA_PATH="/dev/shm/temp_dir/"

dir_local="$DATA_PATH"$$               #tmp directory private to specific excute compute node
mkdir -p $dir_local              #$$ refer to  unique pid of spccific job

module load singularity
singularity exec -c -B /storage -B /home -B /data -W "$dir_local" /home/gaoxiaofeiLab/yaoxingyun/singularity/rbc-workspace_latest.sif \
  $SCRIPT_DIR/do_gam.sh $SCRIPT_DIR $FILE_NAME $WORKING_DIR

source ~/.bashrc
conda activate lucky

ID=$SLURM_ARRAY_TASK_ID

cd "$WORKING_DIR" || exit 1
php $SCRIPT_DIR/convert_output_to_tab.php train.tab.$ID.out

file1="$WORKING_DIR/train.tab.$ID"
file2="$WORKING_DIR/train.tab.$ID.out.tab"
# if file2 not exist, requeue
[ ! -f "$file2" ] && scontrol requeue $SLURM_JOB_ID
[ $(wc -l < "$file1") -ne $(wc -l < "$file2") ] && scontrol requeue $SLURM_JOB_ID

rm -rf $dir_local


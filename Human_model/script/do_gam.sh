#!/bin/bash

SCRIPT_DIR=$1
FILE_NAME=$2
WORKING_DIR=$3

cd /tmp
cp $WORKING_DIR/info.csv .
cp $WORKING_DIR/$FILE_NAME .
php $SCRIPT_DIR/convert_tab_to_input.php $FILE_NAME
ls ${FILE_NAME}.* | xargs -n 1 -P $SLURM_CPUS_ON_NODE -I %1 Rscript $SCRIPT_DIR/gam_normal.r %1 %1.out
$SCRIPT_DIR/merge_gam_out.sh $FILE_NAME out

# 添加重试逻辑
retry_count=0
max_retries=4
while [ $retry_count -le $max_retries ]; do
    mv $FILE_NAME.out $WORKING_DIR && break
    retry_count=$((retry_count+1))
    if [ $retry_count -le $max_retries ]; then
        echo "mv failed, retrying in 15 seconds... (attempt $retry_count of $max_retries)"
        sleep 15
    else
        echo "mv failed after $max_retries attempts, giving up."
        exit 1
    fi
done

rm *.out
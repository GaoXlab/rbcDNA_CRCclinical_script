#!/bin/bash

SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
FEATURE_SELECTION_OUTPUT_DIR=$(cd $SCRIPT_DIR/../results/2_FeatureSelection; pwd)
FEATURE_REDUCTION_OUTPUT_DIR=$(cd $SCRIPT_DIR/../results/3_FeatureReduction; pwd)
FEATURE_CLASSIFICATION_DIR=$(cd $SCRIPT_DIR/../results/4_Classification; pwd)

MODEL_DATA_DIR=$(cd $SCRIPT_DIR/../modelData; pwd)
OUTPUT_PREFIX=$1
# define a function to send message to feishu
message_to_feishu() {
    local message="$1"
    message_xy "[$OUTPUT_PREFIX] $message"
}
module load R/4.2.1
WORKING_DIR=`pwd`/$OUTPUT_PREFIX
mkdir -p $WORKING_DIR
BASE_DIR=$(pwd)

# make 10k - 1000k tab
cd $WORKING_DIR

sbatch-conda lucky --wait -c 4 --mem 12G -o /dev/null -e error.log --open-mode=append python $SCRIPT_DIR/step2_build_info.py  "$OUTPUT_PREFIX" "$WORKING_DIR" "$SCRIPT_DIR" "$MODEL_DATA_DIR"
exit_code=$?
if [ $exit_code -ne 0 ]; then
    message_xy "step2_zheer_new_only_base.py failed with exit code $exit_code"
    exit $exit_code
fi
message_to_feishu "Build 10-1000k tab"
sbatch-conda lucky --wait -a 1-7 -c 16 --mem 32G -q huge -J build_tab -o /dev/null -e error.log --open-mode=append  python $SCRIPT_DIR/step2_build_tab.py "$OUTPUT_PREFIX" "$WORKING_DIR" "$SCRIPT_DIR" "$MODEL_DATA_DIR" --multi 16
exit_code=$?
if [ $exit_code -ne 0 ]; then
    message_xy "step2_build_tab.py failed with exit code $exit_code"
    exit $exit_code
fi

message_to_feishu "apply GAM"
sbatch --wait -a 1-100 $SCRIPT_DIR/gam.sh $SCRIPT_DIR `pwd`
message_to_feishu "GAM finished"
# calc 10m feature scores and select top 1000 for 40 random repeat
message_to_feishu "Feature selection"
sbatch -a 1-50 --wait -q huge $SCRIPT_DIR/fs_gam.sh $SCRIPT_DIR $OUTPUT_PREFIX `pwd`
message_to_feishu "Feature selection finished"
# merge all 40 top 1000 feature scores
message_to_feishu "Merge feature scores"
sbatch-conda lucky --wait -c 16 --mem 64G python $SCRIPT_DIR/merge_p80.py $OUTPUT_PREFIX
message_to_feishu "Merge feature scores finished"
# wait for 10s for storage to sync
message_to_feishu "Feature reduction"
$SCRIPT_DIR/bed_select all.$OUTPUT_PREFIX.bed all.$OUTPUT_PREFIX.bed.out 1000
message_to_feishu "Feature reduction finished"
## clean up workspace and backup all random ids

rm train.tab.*
rm all.*.tab*[0-9]

message_to_feishu "Start building new mode"
sbatch-conda lucky --wait -c 4 --mem 8G $SCRIPT_DIR/new_mode.sh "$OUTPUT_PREFIX" all.$OUTPUT_PREFIX.bed.out

cd $BASE_DIR || exit 1

sbatch-conda lucky --wait -c 48 --mem 96G "$SCRIPT_DIR"/build_feature_data.sh "$MODEL_DATA_DIR"/"$OUTPUT_PREFIX".full.ids.txt "$OUTPUT_PREFIX"
message_to_feishu "build finished"

sbatch-conda lucky --wait -c 4 --mem 8G python "$SCRIPT_DIR"/check_mode.py "$MODEL_DATA_DIR"/"$OUTPUT_PREFIX"

sbatch-conda lucky --wait -c 4 --mem 16G "$SCRIPT_DIR"/make_all_tab.sh "$OUTPUT_PREFIX" "$OUTPUT_PREFIX"/"all.$OUTPUT_PREFIX.raw.tab"

message_to_feishu "Start gam"
sbatch --wait -c 16 --mem 64G -o $OUTPUT_PREFIX/gam.log -p amd-ep2,amd-ep2-short,intel-sc3 xy_cmd Rscript "$SCRIPT_DIR"/step2_gam.R "$OUTPUT_PREFIX"
message_to_feishu "gam finished"

message_to_feishu "All finished"

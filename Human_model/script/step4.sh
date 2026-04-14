#!/bin/bash

# You should run this script in the parent dir of 'Human_Model'

mkdir Figures -p

WORKING_DIR="$(pwd)"

# Build sample_info.RData and prediction_final.Data
Rscript figure_scripts/Build_RData.R "$(pwd)"

# Build train_gam_100k.RData
cd Human_Model || exit 1

script/build_feature_data.sh modelData/zheer.p100.ids.txt trim_gcc_r100k_0start
mkdir trim_gcc_r100k_0start -p

## Using all control samples in trn
cp modelData/{zheer_zr10_1234.neg.ids.txt,trim_gcc_r100k_0start.neg.ids.txt}
sbatch-conda lucky --wait -c 4 --mem 16G script/make_all_tab.sh trim_gcc_r100k_0start trim_gcc_r100k_0start/all.trim_gcc_r100k_0start.raw.tab
sbatch --wait -c 16 --mem 64G -o trim_gcc_r100k_0start/gam.log -p amd-ep2,amd-ep2-short,intel-sc3 xy_cmd Rscript script/step2_build_gam_and_save.R trim_gcc_r100k_0start

cd "$WORKING_DIR" || exit 1

# Build zr268.bed and raw_1_10.bed
Rscript figure_scripts/Build_Features_Data.R "$(pwd)"

# Build zr268.group.bed raw_1~10.group.bed

## Download anno data from official web site
figure_scripts/FeatureAnno/download_anno.sh "$(pwd)"
Python figure_scripts/FeatureAnno/chrom_full_stack_batch.py figure_scripts/FeatureAnno/refData figure_scripts/FeatureAnno figure_scripts/FeatureAnno/*.bed

# Build TotalSample_MT.noalt.log
cd Human_Model ||  exit 1

script/qc_reads_check.sh "./bams/"
mv TotalSample_MT.noalt.log ../Figures/

cd "$WORKING_DIR" || exit 1
# Build r_g_10controls_smooth.RData and trn_nonAN_smooth_anno.RData

# Build selected_model_zheer_with_internal_test.csv
cd Human_Model ||  exit 1
python script/step4_build_p20_auc.py zheer "$(pwd)"
cd "$WORKING_DIR" || exit 1

# Start Run Fig*.R
for fig in 2 3 4 5 S2 S3 S4 S5 S6 S7 S8 S9; do
  /usr/local/bin/Rscript figure_scripts/Figure"$fig".R "$(pwd)"
done
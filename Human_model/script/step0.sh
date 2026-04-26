#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config.sh

WORKING_DIR="$(pwd)"

"$SCRIPT_DIR"/build_feature_data.sh "$MODEL_DATA_DIR"/rbcDNA_regions.ids.txt trim_q30_gcc_10k_cpm

cd rbcDNA_regions || exit 1

bash ${SCRIPT_DIR}/make_tab_fast.sh ${MODEL_DATA_DIR}/rbcDNA_regions.ids.txt trim_q30_gcc_10k_cpm train.tab

sbatch-conda lucky --wait -a 1-7 -c 16 --mem 32G -q huge -J build_tab -o /dev/null -e error.log --open-mode=append  python $SCRIPT_DIR/step2_build_tab.py rbcDNA_regions "$WORKING_DIR"/rbcDNA_regions "$SCRIPT_DIR" "$MODEL_DATA_DIR" --multi 16

cd "$WORKING_DIR" || exit 1


seq 1 100 | xargs -I {} -P 16 python ${SCRIPT_DIR}/step0_build_rbcdna_regions.py --tab_id {}

cat ./rbcDNA_regions/region_fc_part.bed.* > ./rbcDNA_regions/region_fc_all.bed

tail -n +2 rbcDNA_regions/rbcDNAenriched_original.bed | bedtools intersect -a ./rbcDNA_regions/region_fc_all.bed -b - -u > ./rbcDNA_regions/region_fc_all.sig.overlap.bed

CUTOFF=$(python "${SCRIPT_DIR}"/step0_calc_cutoff.py ./rbcDNA_regions/region_fc_all.sig.overlap.bed)
echo "Dynamic Cutoff: $CUTOFF"

cat ./rbcDNA_regions/region_fc_all.bed | awk -F "\t" -v cutoff="$CUTOFF" '$9<0.001 && $7>cutoff {print $1"\t"$2"\t"$3"\t"$7}' > ./rbcDNA_regions/region_fc_all.adjp0.001_log2fc_dynamic_r3.bed
./script/bed_select ./rbcDNA_regions/region_fc_all.adjp0.001_log2fc_dynamic_r3.bed ./rbcDNA_regions/region_fc_all.adjp0.001_log2fc_dynamic_r3.out.bed 1000
echo -n "r3 total region size: "
awk '{sum += ($3 - $2)} END {print sum}' ./rbcDNA_regions/region_fc_all.adjp0.001_log2fc_dynamic_r3.out.bed


cat ./rbcDNA_regions/region_fc_all.bed | awk -F "\t" -v cutoff="$CUTOFF" '$9<0.001 && $7<(-cutoff) {print $1"\t"$2"\t"$3"\t"$7}' > ./rbcDNA_regions/region_fc_all.adjp0.001_log2fc_dynamic_g3.bed
./script/bed_select ./rbcDNA_regions/region_fc_all.adjp0.001_log2fc_dynamic_g3.bed ./rbcDNA_regions/region_fc_all.adjp0.001_log2fc_dynamic_g3.out.bed 1000
echo -n "g3 total region size: "
awk '{sum += ($3 - $2)} END {print sum}' ./rbcDNA_regions/region_fc_all.adjp0.001_log2fc_dynamic_g3.out.bed

$SCRIPT_DIR/new_mode.sh r_enriched ./rbcDNA_regions/region_fc_all.adjp0.001_log2fc_dynamic_r3.out.bed
$SCRIPT_DIR/new_mode.sh g_enriched ./rbcDNA_regions/region_fc_all.adjp0.001_log2fc_dynamic_g3.out.bed


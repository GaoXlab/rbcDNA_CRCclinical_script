# This file shows how to use the pipeline module to reproduce the results of the paper.

# 1. directory structure
```text
├── fq
├── bams
│   ├── gc_corrected
│   └── pipeline_trimmomatic
├── rbcDNA_regions
├── modelData
│   ├── empty
│   │   ├── cleaned
│   │   └── origin
│   └── trim_q30_gcc_10k_cpm
│   │   ├── cleaned
│   │   └── origin
│   └── r_enriched
│   │   ├── cleaned
│   │   └── origin
│   └── r_depleted
│       ├── cleaned
│       └── origin
├── results
│   ├── 2_FeatureSelection
│   ├── 3_FeatureReduction
│   └── 4_Classification
└── script 
```
GC-corrected BAM files should be placed in bams/gc_corrected/, and the module data should be placed in modelData/. The results will be saved in results/.


# 2. run the pipeline
```bash
# 0. build rbcDNA-enriched/-depleted regions.
./script/step0.sh
# 1. 10-kb CPM data construction and rbcDNA-enriched/-depleted value generation for the development cohort.
./script/step1.sh zheer

# zheer pipeline
# 2. Feature preprocessing, GAM-normalization, and feature selection.
## Hardware requirement: At least 48 cores and 96GB RAM
./script/batch_build.sh zheer
./script/batch_build_search.sh zheer

# 3. Feature reduction, model development, and integration.
TYPE="zheer"
./script/step3.sh "$TYPE" 
python ./script/step3_test.py "$TYPE" `pwd` internal_test 
python ./script/step3_test.py "$TYPE" `pwd` trn 

# 4. Independent validation.
TYPE="zheer"
# 4.1. Feature data construction for each independent validation set.
./script/extend_feature_data.sh "$TYPE" ind_sd
./script/extend_feature_data.sh "$TYPE" ind_wz
./script/extend_feature_data.sh "$TYPE" clin

# 4.2. Model evaluation and performance assessment.
python ./script/step3_test.py "$TYPE" `pwd` ind_sd
python ./script/step3_test.py "$TYPE" `pwd` ind_wz
python ./script/step3_test.py "$TYPE" `pwd` clin
``` 

Feature selection results will be saved in results/2_FeatureSelection/, feature reduction results in results/3_FeatureReduction/, and classification results in results/4_Classification/.


suppressPackageStartupMessages({
  library(dplyr)
  library(readxl)
  library(readr)
})
excel_file <- "./Figures/Supplementary_Tables.xlsx"

table1 <- read_excel(excel_file, sheet = "Supplementary Table 1")
table3 <- read_excel(excel_file, sheet = "Supplementary Table 3")


trn_info  <- table1 %>% filter(Cohort == "Discovery")
test_info <- table1 %>% filter(Cohort == "Internal test")
wz_info   <- table1 %>% filter(Cohort == "WENZHOU")
sd_info   <- table1 %>% filter(Cohort == "SHANDONG")
zr_info   <- table3

save(trn_info, test_info, wz_info, sd_info, zr_info, file = "./Figures/sampleinfo.RData")

pred_trn   <- read_csv('Human_model/results/4_Classification/zheer_trn_predictions.csv', show_col_types = FALSE)
pred_test  <- read_csv('Human_model/results/4_Classification/zheer_internal_test_predictions.csv', show_col_types = FALSE)
pred_sd    <- read_csv('Human_model/results/4_Classification/zheer_ind_sd_predictions.csv', show_col_types = FALSE)
pred_wz    <- read_csv('Human_model/results/4_Classification/zheer_ind_wz_predictions.csv', show_col_types = FALSE)
pred_clinc <- read_csv('Human_model/results/4_Classification/zheer_clin_predictions.csv', show_col_types = FALSE)

align_prediction <- function(pred_df, info_df) {
  pred_df %>%
    rename(any_of(c(Sample = "seqID"))) %>%
    rename(any_of(c(merged_score = "final_prob"))) %>%
    select(Sample, merged_score) %>%
    inner_join(info_df %>% select(Sample, Group), by = "Sample") %>%
    mutate(Target = ifelse(Group %in% c("AA", "CRC"), 1, 0)) %>%
    # 【关键修改】：剔除 Group，只输出纯粹的三列
    select(Sample, Target, merged_score)
}

trn      <- align_prediction(pred_trn, trn_info)
test     <- align_prediction(pred_test, test_info)
sd       <- align_prediction(pred_sd, sd_info)
wz       <- align_prediction(pred_wz, wz_info)
zr_clinc <- align_prediction(pred_clinc, zr_info)

save(trn, test, sd, wz, zr_clinc, file = "./Figures/prediction.RData")
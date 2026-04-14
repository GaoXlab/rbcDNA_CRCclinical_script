args <- commandArgs(trailingOnly = TRUE)
working_dir <- args[1]
setwd(working_dir)

args_all <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args_all[grep("--file=", args_all)])
script_dir <- dirname(normalizePath(script_path))

out_dir <- file.path(working_dir, "Figures")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

suppressPackageStartupMessages({
  library(openxlsx)
  library(dplyr)
  library(pROC)
  library(reportROC)
  library(stringr)
})

source(file.path(script_dir, 'plot_function.r'), chdir = TRUE)

load('./Figures/prediction.RData')
load('./Figures/sampleinfo.RData')
cutoff_spe90 <- Cutoff(0.90, test)

zr_fit <- merge(zr_clinc, zr_info, by = 'Sample')

colnames(zr_fit) <- gsub('-| ', '', colnames(zr_fit))

zr_fit <- zr_fit %>%
  mutate(
    Target = ifelse(grepl('AA|CRC', Group), 1, 0),
    rbcDNA_results = ifelse(merged_score >= cutoff_spe90, 1, 0)
  )

zr_fit_rbc <- zr_fit %>% mutate(Target = ifelse(Group == 'AA', 1, Target))

eval_subsets <- list(
  "CRC_total" = zr_fit_rbc %>% filter(Group != 'AA'),
  "CRC_stage_I_III" = zr_fit_rbc %>% filter(Group == 'Control' | Subgroup %in% c('I', 'II', 'III')),
  "AA_total" = zr_fit_rbc %>% filter(Group != 'CRC'),
  "Advanced_neoplasia" = zr_fit_rbc,
  "Negative_colonoscopy" = zr_fit_rbc %>% filter(!Subgroup %in% c('Nonneoplastic findings', 'Non-advanced adenoma (NAA)', '(other CA)')),
  "Nonneoplastic_findings_and_Negative_colonoscopy" = zr_fit_rbc %>% filter(!Subgroup %in% c('(other CA)', 'Non-advanced adenoma (NAA)'))
)

res_rbc <- lapply(names(eval_subsets), function(nm) {
  evaluate_all_sets(eval_subsets[[nm]], cutoff_spe90, nm)
})
names(res_rbc) <- names(eval_subsets)

rbc_sens <- do.call(rbind, lapply(res_rbc[c("CRC_total", "CRC_stage_I_III", "AA_total")], `[[`, "sensitivity")) %>%
  rename(Sensitivity = SEN, Sens.low = SEN.low, Sens.up = SEN.up)

rbc_spec <- do.call(rbind, lapply(res_rbc[c("Advanced_neoplasia", "Nonneoplastic_findings_and_Negative_colonoscopy", "Negative_colonoscopy")], `[[`, "specificity")) %>%
  rename(Specificity = SPE, Spec.low = SPE.low, Spec.up = SPE.up)

fit_subsets <- lapply(eval_subsets, function(df) {
  df %>% rename(rbcDNA_score = merged_score, merged_score = FIT_results)
})

res_fit <- lapply(names(fit_subsets), function(nm) {
  evaluate_all_sets(fit_subsets[[nm]], 0.5, nm)
})
names(res_fit) <- names(fit_subsets)

fit_sens <- do.call(rbind, lapply(res_fit[c("CRC_total", "CRC_stage_I_III", "AA_total")], `[[`, "sensitivity")) %>%
  rename(Sensitivity = SEN, Sens.low = SEN.low, Sens.up = SEN.up)

fit_spec <- do.call(rbind, lapply(res_fit[c("Advanced_neoplasia", "Nonneoplastic_findings_and_Negative_colonoscopy", "Negative_colonoscopy")], `[[`, "specificity")) %>%
  rename(Specificity = SPE, Spec.low = SPE.low, Spec.up = SPE.up)

out_file <- file.path(out_dir, 'figureS6_table.xlsx')
wb <- createWorkbook()

addWorksheet(wb, "rbcDNA_Sensitivity")
writeData(wb, "rbcDNA_Sensitivity", rbc_sens)

addWorksheet(wb, "rbcDNA_Specificity")
writeData(wb, "rbcDNA_Specificity", rbc_spec)

addWorksheet(wb, "qFIT_Sensitivity")
writeData(wb, "qFIT_Sensitivity", fit_sens)

addWorksheet(wb, "qFIT_Specificity")
writeData(wb, "qFIT_Specificity", fit_spec)

saveWorkbook(wb, out_file, overwrite = TRUE)
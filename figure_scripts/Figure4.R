args <- commandArgs(trailingOnly = TRUE)
working_dir <- args[1]
setwd(working_dir)

args_all <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args_all[grep("--file=", args_all)])
script_dir <- dirname(normalizePath(script_path))

out_dir <- file.path(working_dir, "Figures")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

suppressPackageStartupMessages({
  library(pROC)
  library(stringr)
  library(ggplot2)
  library(cowplot)
  library(dplyr)
  library(openxlsx)
})
source(file.path(script_dir, 'plot_function.r'), chdir = TRUE)
source(file.path(script_dir, 'plotAUC_theme.r'), chdir = TRUE)

load('./Figures/prediction.RData')
load('./Figures/sampleinfo.RData')
cutoff_spe90 <- Cutoff(0.90, test)

zr_fit <- merge(zr_clinc, zr_info, by = 'Sample')
colnames(zr_fit) <- gsub('-| ', '', colnames(zr_fit))

zr_fit <- zr_fit %>%
  mutate(
    Target = ifelse(grepl('AA|CRC', Group), 1, 0),
    Group = ifelse(Group == 'Control', 'Non-AN control', Group),
    rbcDNA_results = ifelse(merged_score >= cutoff_spe90, 1, 0),
    predLabel = ifelse(merged_score < cutoff_spe90, 0, 1)
  )

# Figure 4b & 4c: ROC Curves vs qFIT
pdf(file.path(out_dir, 'figure4_b.pdf'), width = 4, height = 4)
color1 <- "#006838"
color2 <- "#D2F3EA7F"
plot_auc_95CI_1_withFIT(zr_fit, color1, color2)
dev.off()

HDCRC <- zr_fit %>% filter(Group != 'AA')
HDAA <- zr_fit %>% filter(Group != 'CRC')

pdf(file.path(out_dir, 'figure4_c.pdf'), width = 4, height = 4)
plot_auc_95CI_2_withFIT(HDCRC, HDAA)
dev.off()

# Figure 4d, 4e, 4f: Sensitivity and Specificity Bar Charts (rbcDNA vs qFIT)

eval_subsets <- list(
  "CRC_total" = zr_fit %>% filter(Group != 'AA'),
  "crc_1" = zr_fit %>% filter(Group == 'Non-AN control' | Subgroup == 'I'),
  "crc_2" = zr_fit %>% filter(Group == 'Non-AN control' | Subgroup == 'II'),
  "crc_3" = zr_fit %>% filter(Group == 'Non-AN control' | Subgroup == 'III'),
  "crc_4" = zr_fit %>% filter(Group == 'Non-AN control' | Subgroup == 'IV'),
  "AA_total" = zr_fit %>% filter(Group != 'CRC'),
  "aa_ss10mm" = zr_fit %>% filter(Group == 'Non-AN control' | Subgroup == '(2e)Serrated adenoma, ≥10 mm'),
  "aa_10mm" = zr_fit %>% filter(Group == 'Non-AN control' | Subgroup == '(2d)Tubular adenoma, ≥10 mm'),
  "aa_vil" = zr_fit %>% filter(Group == 'Non-AN control' | Subgroup == '(2c)Villous'),
  "aa_hgd" = zr_fit %>% filter(Group == 'Non-AN control' | Subgroup == '(2b)High-grade dysplasia'),
  "Ctrl_total" = zr_fit,
  "Negative_colonoscopy" = zr_fit %>% filter(!Subgroup %in% c('Nonneoplastic findings', 'Non-advanced adenoma (NAA)', '(other CA)')),
  "Nonneoplastic_findings" = zr_fit %>% filter(!Subgroup %in% c('Negative colonoscopy', 'Non-advanced adenoma (NAA)', '(other CA)')),
  "NAA" = zr_fit %>% filter(!Subgroup %in% c('Nonneoplastic findings', 'Negative colonoscopy', '(other CA)')),
  "otherCA" = zr_fit %>% filter(!Subgroup %in% c('Nonneoplastic findings', 'Negative colonoscopy', 'Non-advanced adenoma (NAA)'))
)

rbc_res <- lapply(names(eval_subsets), function(nm) {
  evaluate_all_sets(eval_subsets[[nm]], cutoff_spe90, nm)
})
names(rbc_res) <- names(eval_subsets)

fit_subsets <- lapply(eval_subsets, function(df) {
  df %>% rename(rbcDNA_score = merged_score, merged_score = FIT_results)
})

fit_res <- lapply(names(fit_subsets), function(nm) {
  evaluate_all_sets(fit_subsets[[nm]], 0.5, nm)
})
names(fit_res) <- names(fit_subsets)

extract_metrics <- function(res_list, metric_type, test_name) {
  if (metric_type == "sens") {
    target_names <- c("Ctrl_total", "CRC_total", "crc_1", "crc_2", "crc_3", "crc_4", "AA_total", "aa_ss10mm", "aa_10mm", "aa_vil", "aa_hgd")
    df <- do.call(rbind, lapply(res_list[target_names], `[[`, "sensitivity"))
    df <- df %>% rename(perf_metric = SEN, perf_metric.low = SEN.low, perf_metric.up = SEN.up, Label2 = classify)
  } else {
    target_names <- c("Ctrl_total", "Negative_colonoscopy", "Nonneoplastic_findings", "NAA", "otherCA")
    df <- do.call(rbind, lapply(res_list[target_names], `[[`, "specificity"))
    df <- df %>% rename(perf_metric = SPE, perf_metric.low = SPE.low, perf_metric.up = SPE.up, Label2 = classify)
  }
  df$test <- test_name
  return(df)
}

rbc_sens <- extract_metrics(rbc_res, "sens", "rbcDNA")
rbc_spec <- extract_metrics(rbc_res, "spec", "rbcDNA")
fit_sens <- extract_metrics(fit_res, "sens", "fit")
fit_spec <- extract_metrics(fit_res, "spec", "fit")

all_spec <- rbind(fit_spec, rbc_spec) %>% filter(Label2 != 'Nonneoplastic_findings + Negative_colonoscopy')
all_spec$Label2 <- factor(all_spec$Label2, levels = c("Ctrl_total", "Negative_colonoscopy", "Nonneoplastic_findings", "NAA", 'otherCA'))
all_spec$test <- factor(all_spec$test, levels = c('rbcDNA', 'fit'))

all_sens <- rbind(fit_sens, rbc_sens)
all_sens$test <- factor(all_sens$test, levels = c('rbcDNA', 'fit'))
all_sens$Label2 <- factor(all_sens$Label2, levels = c('AA_total', 'aa_ss10mm', 'aa_10mm', 'aa_vil', 'aa_hgd', 'CRC_total', 'crc_1', 'crc_2', 'crc_3', 'crc_4'))

g1_spec <- ggplot(all_spec, aes(x = Label2, y = perf_metric, fill = test)) +
  geom_vline(xintercept = 1.5, color = 'grey40', linetype = 'dotted', linewidth = 0.5) +
  geom_bar(stat = "identity", color = "black", position = position_dodge(), alpha = 0.6) +
  geom_errorbar(aes(ymin = perf_metric.low, ymax = perf_metric.up), width = .2, position = position_dodge(.9)) +
  geom_text(aes(label = perc), vjust = 1.2, color = "black", position = position_dodge(.9), size = 2.5) +
  ylim(0, 100) +
  scale_fill_manual(values = c("#3B6642", "grey")) +
  ylab('Specificity (%)') +
  xlab(NULL) +
  scale_x_discrete(labels = c("Ctrl_total" = "Total",
                              "Negative_colonoscopy" = "Negative\ncolonoscopy",
                              "Nonneoplastic_findings" = "Nonneoplastic\nfindings",
                              "NAA" = "Non-\nadvanced\nadenomas",
                              "otherCA" = "other\ncancer")) +
  ggtitle("Subtype of control group") +
  theme_bar +
  theme(
    legend.position = 'none',
    plot.title = element_text(hjust = 0.5, face = "bold", size = 12),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.8),
    axis.line = element_blank()
  )

g1_aa_sens <- ggplot(all_sens %>% filter(grepl('AA_|aa_', Label2)), aes(x = Label2, y = perf_metric, fill = test)) +
  geom_vline(xintercept = c(1.5), color = 'grey40', linetype = 'dotted', linewidth = 0.5) +
  geom_bar(stat = "identity", color = "black", position = position_dodge(), alpha = 0.6) +
  geom_errorbar(aes(ymin = perf_metric.low, ymax = perf_metric.up), width = .2, position = position_dodge(.9)) +
  geom_text(aes(label = perc), vjust = 1.2, color = "black", position = position_dodge(.9), size = 2.5) +
  ylim(0, 100) +
  scale_fill_manual(values = c("#3B6642", "grey")) +
  ylab('Subtype of advanced adenomas') +
  xlab(NULL) +
  scale_x_discrete(labels = c("AA_total" = "Total",
                              "aa_ss10mm" = "Sessile serrated\nlesions\n≥10 mm",
                              "aa_10mm" = "Tubular\nadenomas\n≥10 mm",
                              "aa_vil" = "Villous",
                              "aa_hgd" = "High-grade\ndysplasia")) +
  ggtitle("Subtype of advanced adenomas") +
  theme_bar +
  theme(
    legend.position = 'none',
    plot.title = element_text(hjust = 0.5, face = "bold", size = 12),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.8),
    axis.line = element_blank()
  )

g1_crc_sens <- ggplot(all_sens %>% filter(grepl('CRC_|crc_', Label2)), aes(x = Label2, y = perf_metric, fill = test)) +
  geom_vline(xintercept = c(1.5), color = 'grey40', linetype = 'dotted', linewidth = 0.5) +
  geom_bar(stat = "identity", color = "black", position = position_dodge(), alpha = 0.6) +
  geom_errorbar(aes(ymin = perf_metric.low, ymax = perf_metric.up), width = .2, position = position_dodge(.9)) +
  geom_text(aes(label = perc), vjust = 1.2, color = "black", position = position_dodge(.9), size = 2.5) +
  ylim(0, 100) +
  scale_fill_manual(values = c("#3B6642", "grey")) +
  ylab('Colorectal cancer stage') +
  xlab(NULL) +
  scale_x_discrete(labels = c("CRC_total" = "Total",
                              "crc_1" = "I",
                              "crc_2" = "II",
                              "crc_3" = "III",
                              "crc_4" = "IV")) +
  ggtitle("Colorectal Cancer (CRC)") +
  theme_bar +
  theme(
    legend.position = 'none',
    plot.title = element_text(hjust = 0.5, face = "bold", size = 12),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.8),
    axis.line = element_blank()
  )

shared_legend <- cowplot::get_legend(
  g1_spec +
    scale_fill_manual(values = c("#3B6642", "grey"), labels = c("rbcDNA", "qFIT")) +
    theme(
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.justification = "center",
      legend.title = element_blank(),
      legend.text = element_text(size = 11, color = "black"),
      legend.key.size = unit(0.5, "cm")
    )
)

g_combined <- plot_grid(
  g1_spec, g1_aa_sens, g1_crc_sens,
  ncol = 3,
  align = "hv",
  axis = "tblr"
)

final_plot <- plot_grid(
  g_combined, shared_legend,
  ncol = 1,
  rel_heights = c(1, 0.1)
)

ggsave(file.path(out_dir, 'figure4_def.pdf'), final_plot, width = 12.5, height = 4.2)

# Figure 4: AUCs (rbcDNA vs qFIT) pvalues (McNemar's Test)
run_delong <- function(df, label) {
  roc1 <- pROC::roc(df$Target, df$merged_score, ci = TRUE)
  roc2 <- pROC::roc(df$Target, df$qFIT_value, ci = TRUE)

  tst <- roc.test(roc1, roc2, method = "delong", paired = FALSE)

  data.frame(
    Comparison = label,
    N = nrow(df),
    AUC_rbcDNA = as.numeric(pROC::auc(roc1)),
    AUC_qFIT = as.numeric(pROC::auc(roc2)),
    Delta_AUC = as.numeric(pROC::auc(roc1) - pROC::auc(roc2)),
    Z = unname(tst$statistic),
    p_value = unname(tst$p.value),
    stringsAsFactors = FALSE
  )
}

res_all <- run_delong(zr_fit, "All")
res_noAA <- run_delong(subset(zr_fit, Group != "AA"), "Exclude AA")
res_noCRC <- run_delong(subset(zr_fit, Group != "CRC"), "Exclude CRC")

out1 <- rbind(res_all, res_noAA, res_noCRC)

# Figure 4: Sensitivity and Specificity (rbcDNA vs qFIT) pvalues (McNemar's Test)
subgroups <- c(
  "Negative colonoscopy",
  "Nonneoplastic findings",
  "Non-advanced adenoma (NAA)",
  "(2e)Serrated adenoma, ≥10 mm",
  "(2d)Tubular adenoma, ≥10 mm",
  "(2c)Villous",
  "(2b)High-grade dysplasia",
  "I", "II", "III", "IV"
)

combo_defs <- list(
  "Neg + Nonneo" = c("Negative colonoscopy", "Nonneoplastic findings"),
  "CRC I+II" = c("I", "II"),
  "CRC I–III" = c("I", "II", "III"),
  "CRC II–IV" = c("II", "III", "IV"),
  "CRC III+IV" = c("III", "IV")
)

single_defs <- setNames(lapply(subgroups, function(x) x), subgroups)
group_defs <- c(single_defs, combo_defs)

mcnemar_one <- function(df, groups,
                        subgroup_col = "Subgroup",
                        rbc_col = "rbcDNA_results",
                        fit_col = "FIT_results") {

  sub <- df[df[[subgroup_col]] %in% groups, c(rbc_col, fit_col)]
  sub <- sub[complete.cases(sub), , drop = FALSE]
  n <- nrow(sub)
  if (n == 0) return(list(n = 0, p = NA_real_, tab = matrix(NA, 2, 2)))

  tab <- table(rbc = sub[[rbc_col]], qfit = sub[[fit_col]])

  if (!all(dim(tab) == c(2, 2))) {
    return(list(n = n, p = NA_real_, tab = tab))
  }

  p <- as.numeric(mcnemar.test(tab)$p.value)
  list(n = n, p = p, tab = tab)
}

out2 <- do.call(rbind, lapply(names(group_defs), function(name) {
  res <- mcnemar_one(zr_fit, group_defs[[name]])

  a <- b <- c_ <- d <- NA_real_
  if (is.matrix(res$tab) && all(dim(res$tab) == c(2, 2))) {
    vals <- as.numeric(res$tab)
    a <- vals[1]; b <- vals[2]; c_ <- vals[3]; d <- vals[4]
  }

  data.frame(
    Group = name,
    Subgroups_included = paste(group_defs[[name]], collapse = " | "),
    N = res$n,
    p_value = res$p,
    a = a, b = b, c = c_, d = d,
    stringsAsFactors = FALSE
  )
}))

write.xlsx(list('AUCs_comparison' = out1, 'SensSpec_comparison' = out2),
           file.path(out_dir, 'figure4_stats.xlsx'), rowNames = FALSE)
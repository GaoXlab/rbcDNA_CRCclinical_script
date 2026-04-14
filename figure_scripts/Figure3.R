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
  library(ggplotify)
  library(patchwork)
  library(openxlsx)
})

source(file.path(script_dir, 'plot_function.r'), chdir = TRUE)
source(file.path(script_dir, 'plotAUC_theme.r'), chdir = TRUE)

load('./Figures/prediction.RData')
load('./Figures/sampleinfo.RData')

test <- merge(test, test_info, by = 'Sample')
colnames(test) <- gsub('-| ', '', colnames(test))
cutoff_spe90 <- Cutoff(0.90, test)

# Figure 3b: ROC Curve for Test Cohort
pdf(file.path(out_dir, 'figure3_b.pdf'), width = 4, height = 4)
plot_auc_95CI_1(test, "#E64B35B2", rgb(255, 0, 0, 20, maxColorValue = 255))
dev.off()

# Figure 3c: Boxplot of Predictive Scores by Disease Group
test <- test %>%
  mutate(
    Group2 = factor(ifelse(Group != 'Non-AN control', 'Advanced neoplasia', 'Non-AN'), levels = c('Non-AN', 'Advanced neoplasia')),
    Subgroup = case_when(
      Subgroup == '(2c)Villous, focal HGD' ~ '(2b)High-grade dysplasia',
      grepl('≥10 mm', Subgroup) ~ 'adenoma, ≥10 mm',
      TRUE ~ Subgroup
    )
  )

p_boxplot <- ggplot(data = test, aes(x = Group2, y = merged_score, fill = Group2)) +
  geom_boxplot(outlier.shape = NA, lwd = 0.3, alpha = 0.7, width = 0.5) +
  theme_classic() +
  geom_jitter(width = 0.2, size = 0.05) +
  geom_hline(yintercept = cutoff_spe90, color = 'red4', linetype = 'dashed', linewidth = 0.5) +
  annotate("text", x = Inf, y = cutoff_spe90, label = "Cutoff", color = "black", hjust = -0.1, vjust = 0.4, size = 3.5) +
  scale_fill_manual(values = c(ggsci::pal_material("blue-grey")(10)[5], "#9F1A1AFF")) +
  theme(
    legend.position = 'none',
    legend.title = element_blank(),
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1, color = "black"),
    axis.text.y = element_text(color = "black"),
    plot.margin = margin(t = 5.5, r = 40, b = 5.5, l = 5.5)
  ) +
  coord_cartesian(ylim = c(0, 1), clip = "off") +
  ylab('rbcDNA predictive score') +
  xlab(NULL)

# ggsave(file.path(out_dir, 'figure3_c.pdf'), p_boxplot, width = 2.1, height = 3.5)

# Figure 3d: Sensitivity and Specificity Bar Charts Across Subgroups
cols_ind <- c(
  ggsci::pal_material("blue-grey")(10)[5], ggsci::pal_material("blue-grey")(10)[2:4],
  '#DF8F44FF', ggsci::pal_material("orange")(10)[2:4],
  "#9F1A1AFF", ggsci::pal_material("red")(10)[2:5]
)

CRC_test <- test

eval_sets <- list(
  "CRC_total"              = CRC_test %>% filter(Group != 'AA'),
  "crc_1"                  = CRC_test %>% filter(Group == 'Non-AN control' | Subgroup == 'I'),
  "crc_2"                  = CRC_test %>% filter(Group == 'Non-AN control' | Subgroup == 'II'),
  "crc_3"                  = CRC_test %>% filter(Group == 'Non-AN control' | Subgroup == 'III'),
  "crc_4"                  = CRC_test %>% filter(Group == 'Non-AN control' | Subgroup == 'IV'),
  "AA_total"               = CRC_test %>% filter(Group != 'CRC'),
  "aa_10mm"                = CRC_test %>% filter(Group == 'Non-AN control' | Subgroup == 'adenoma, ≥10 mm'),
  "aa_vil"                 = CRC_test %>% filter(Group == 'Non-AN control' | Subgroup == '(2c)Villous'),
  "aa_hgd"                 = CRC_test %>% filter(Group == 'Non-AN control' | Subgroup == '(2b)High-grade dysplasia'),
  "Ctrl_total"             = CRC_test,
  "Negative_colonoscopy"   = CRC_test %>% filter(!Subgroup %in% c('Nonneoplastic findings', 'Non-advanced adenoma (NAA)')),
  "Nonneoplastic_findings" = CRC_test %>% filter(!Subgroup %in% c('Negative colonoscopy', 'Non-advanced adenoma (NAA)')),
  "NAA"                    = CRC_test %>% filter(!Subgroup %in% c('Nonneoplastic findings', 'Negative colonoscopy'))
)

res_list <- lapply(names(eval_sets), function(nm) {
  evaluate_all_sets(eval_sets[[nm]], cutoff_spe90, nm)
})
names(res_list) <- names(eval_sets)

sens_names <- c("CRC_total", "crc_1", "crc_2", "crc_3", "crc_4", "AA_total", "aa_10mm", "aa_vil", "aa_hgd")
sens <- do.call(rbind, lapply(res_list[sens_names], `[[`, "sensitivity"))

spec_names <- c("Ctrl_total", "Negative_colonoscopy", "Nonneoplastic_findings", "NAA")
spec <- do.call(rbind, lapply(res_list[spec_names], `[[`, "specificity"))

sens <- sens %>%
  rename(perf_metric = SEN, perf_metric.low = SEN.low, perf_metric.up = SEN.up, Label2 = classify) %>%
  mutate(label = 'sens',
         Label2 = factor(Label2, levels = c('AA_total', 'aa_10mm', 'aa_vil', 'aa_hgd', 'CRC_total', 'crc_1', 'crc_2', 'crc_3', 'crc_4')))

spec <- spec %>%
  rename(perf_metric = SPE, perf_metric.low = SPE.low, perf_metric.up = SPE.up, Label2 = classify) %>%
  mutate(label = 'spec',
         Label2 = factor(Label2, levels = c("Ctrl_total", "Negative_colonoscopy", "Nonneoplastic_findings", "NAA")))

g1_spec <- ggplot(spec, aes(x = Label2, y = perf_metric, fill = Label2)) +
  geom_vline(xintercept = 1.5, color = 'grey40', linetype = 'dotted', linewidth = 0.5) +
  geom_bar(stat = "identity", color = "black", position = position_dodge(), alpha = 0.6) +
  geom_errorbar(aes(ymin = perf_metric.low, ymax = perf_metric.up), width = .2, position = position_dodge(.9)) +
  geom_text(aes(label = perc), vjust = 3, color = "black", position = position_dodge(.9), size = 2.5) +
  ylim(0, 100) +
  scale_fill_manual(values = cols_ind[1:4]) +
  scale_x_discrete(labels = c(
    "Ctrl_total" = "Non-AN control\n(Total)",
    "Negative_colonoscopy" = "Negative colonoscopy",
    "Nonneoplastic_findings" = "Nonneoplastic findings",
    "NAA" = "Nonadvanced adenomas"
  )) +
  theme_bar +
  theme(
    legend.position = 'none',
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1, color = "black"),
    axis.text.y = element_text(color = "black"),
    axis.title.y = element_text(size = 10)
  ) +
  ylab('Specificity (%)') +
  xlab(NULL)

g1_sens <- ggplot(sens, aes(x = Label2, y = perf_metric, fill = Label2)) +
  geom_vline(xintercept = c(1.5, 4.5, 5.5), color = 'grey40', linetype = 'dotted', linewidth = 0.5) +
  geom_bar(stat = "identity", color = "black", position = position_dodge(), alpha = 0.6) +
  geom_errorbar(aes(ymin = perf_metric.low, ymax = perf_metric.up), width = .2, position = position_dodge(.9)) +
  geom_text(aes(label = perc), vjust = 3, color = "black", position = position_dodge(.9), size = 2.5) +
  ylim(0, 100) +
  scale_fill_manual(values = cols_ind[5:length(cols_ind)]) +
  scale_x_discrete(labels = c(
    "AA_total" = "AA (Total)",
    "aa_10mm" = "Adenomas, ≥10 mm",
    "aa_vil" = "Villous",
    "aa_hgd" = "High-grade dysplasia",
    "CRC_total" = "CRC (Total)",
    "crc_1" = "I",
    "crc_2" = "II",
    "crc_3" = "III",
    "crc_4" = "IV"
  )) +
  theme_bar +
  theme(
    legend.position = 'none',
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1, color = "black"),
    axis.text.y = element_text(color = "black"),
    axis.title.y = element_text(size = 10)
  ) +
  ylab('Sensitivity at 90% specificity (%)') +
  xlab(NULL)

# ggsave(file.path(out_dir, 'figure3_d.pdf'), plot_grid(g1_spec, g1_sens, ncol = 2, rel_widths = c(1, 2)), width = 7, height = 3.8, device = cairo_pdf)

p3c <- p_boxplot + theme(plot.margin = margin(t = 10, r = 10, b = 10, l = 10))
p3d_left <- g1_spec + theme(plot.margin = margin(t = 10, r = 10, b = 10, l = 10))
p3d_right <- g1_sens + theme(plot.margin = margin(t = 10, r = 10, b = 10, l = 10))

g_cd <- plot_grid(
  p3c,
  p3d_left,
  p3d_right,
  nrow = 1,
  align = "h",
  axis = "bt",
  rel_widths = c(1, 1.5, 3)
)

ggsave(file.path(out_dir, 'figure3_cd_combined.pdf'),
       g_cd,
       width = 10,
       height = 4,
       device = cairo_pdf)
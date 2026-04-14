# =====================================================================
# Script: FigureS7.R
# Description: Generates Figure S7 subplots (a, b) visualizing the
# concordance and accuracy between rbcDNA and qFIT across subgroups.
# =====================================================================

args <- commandArgs(trailingOnly = TRUE)
working_dir <- args[1]
setwd(working_dir)

args_all <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args_all[grep("--file=", args_all)])
script_dir <- dirname(normalizePath(script_path))

out_dir <- file.path(working_dir, "Figures")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

suppressPackageStartupMessages({
  library(tableone)
  library(pROC)
  library(stringr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(cowplot)
})

source(file.path(script_dir, 'plot_function.r'), chdir = TRUE)
source(file.path(script_dir, 'plotAUC_theme.r'), chdir = TRUE)

load('./Figures/prediction_final.clean.zheer.20260405.RData')
load('./Figures/sampleinfo_final.clean.zheer.20260405.RData')

cutoff_spe90 <- Cutoff(0.90, test)

# =====================================================================
# Clean Data & Determine Concordance Labels
# =====================================================================
zr_fit <- merge(zr_clinc, zr_info, by = 'Sample')
colnames(zr_fit) <- gsub('-| ', '', colnames(zr_fit))

zr_fit_clean <- zr_fit %>%
  mutate(
    Target = ifelse(grepl('AA|CRC', Group), 1, 0),
    rbcDNA_results = ifelse(merged_score >= cutoff_spe90, 1, 0),
    rbcDNA_acc = ifelse(Target == rbcDNA_results, 'right', 'wrong'),
    FIT_acc = ifelse(Target == FIT_results, 'right', 'wrong'),
    final_label = case_when(
      rbcDNA_acc == 'right' & FIT_acc == 'right' ~ 'rbcDNA+qFIT',
      rbcDNA_acc == 'wrong' & FIT_acc == 'right' ~ 'qFIT only',
      rbcDNA_acc == 'right' & FIT_acc == 'wrong' ~ 'rbcDNA only',
      TRUE ~ 'Not detected'
    ),
    final_label = factor(final_label, levels = rev(c('rbcDNA+qFIT', 'rbcDNA only', 'qFIT only', 'Not detected')))
  )

# =====================================================================
# Figure S7a: Overall Accuracy Breakdown with Integrated Table
# =====================================================================
library(gridExtra)
library(cowplot)

tmp1_melt <- zr_fit_clean %>%
  mutate(
    Cohort_clean = case_when(
      Target == 0 ~ "Non-AN",
      Target == 1 ~ "AA / CRC",
      TRUE ~ NA_character_
    ),
    Cohort_clean = factor(Cohort_clean, levels = c("Non-AN", "AA / CRC"))
  ) %>%
  group_by(Cohort_clean, final_label) %>%
  tally() %>%
  group_by(Cohort_clean) %>%
  mutate(value = n / sum(n) * 100, variable = Cohort_clean) %>%
  ungroup()

p3 <- ggplot(tmp1_melt, aes(x = variable, y = value, fill = final_label)) +
  geom_bar(stat = "identity", width = 0.8) +
  scale_fill_manual(values = rev(c("#8491B4", "#91D1C2", "#F39B7F", "#C7C7C7"))) +
  # scale_y_continuous(expand = c(0, 0)) +
  coord_flip() +
  labs(x = NULL, y = 'Accuracy (%)') +
  theme_bw(base_family = "ArialMT") +
  theme(
      legend.position = 'top',
      legend.key.size = unit(0.3, "cm"),
      legend.title = element_blank(),
      legend.text = element_text(size = 5),
      legend.margin = margin(b = -5),
      axis.text = element_text(color = "black", size = 5),
      axis.title.x = element_text(color = "black", size = 5, margin = margin(t = 10)),
      panel.grid.major.y = element_blank(),
      panel.border = element_rect(color = "black", linewidth = 0.4),
      plot.margin = margin(t = 5, r = 5, b = 25, l = 5)
  )

tbl_data <- data.frame(
  " " = c("Non-AN", "", "AA / CRC", ""),
  Misdiagnosed = c("qFIT positive", "rbcDNA positive", "qFIT negative", "rbcDNA negative"),
  n = c(18, 30, 181, 90),
  "Diagnosis\nmethod" = c("rbcDNA", "qFIT", "rbcDNA", "qFIT"),
  "Predicted\nright" = c(15, 27, 111, 20),
  "Accuracy\n(%)" = c("83.3%", "90.0%", "61.3%", "22.2%"),
  check.names = FALSE
)

tt <- ttheme_minimal(
  core = list(
    bg_params = list(fill = NA, col = NA),
    fg_params = list(fontfamily = "ArialMT", fontsize = 5, hjust = 0.5, x = 0.5),
    padding = unit(c(1.5, 4), "mm")
  ),
  colhead = list(
    bg_params = list(fill = NA, col = NA),
    fg_params = list(fontfamily = "ArialMT", fontsize = 5, fontface = "bold"),
    padding = unit(c(1.5, 4), "mm"),
    border = list(bottom = grid::gpar(lwd = 1, col = "black"))   # 添加下划线

  )
)

tbl_grob <- tableGrob(tbl_data, rows = NULL, theme = tt)

# 添加表头顶线（粗黑线）
tbl_grob <- gtable::gtable_add_grob(tbl_grob,
                                    grobs = grid::segmentsGrob(
                                      y0 = unit(1, "npc"), y1 = unit(1, "npc"),
                                      gp = grid::gpar(lwd = 2, col = "black")
                                    ),
                                    t = 1, l = 1, r = ncol(tbl_grob))

# 添加表头下划线（黑色细线，在表头行底部）
tbl_grob <- gtable::gtable_add_grob(tbl_grob,
                                    grobs = grid::segmentsGrob(
                                      y0 = unit(0, "npc"), y1 = unit(0, "npc"),
                                      gp = grid::gpar(lwd = 1, col = "black")
                                    ),
                                    t = 1, l = 1, r = ncol(tbl_grob))

# 添加第3行下方的灰色分隔线（即第二组数据后的分隔线）
tbl_grob <- gtable::gtable_add_grob(tbl_grob,
                                    grobs = grid::segmentsGrob(
                                      y0 = unit(0, "npc"), y1 = unit(0, "npc"),
                                      gp = grid::gpar(lwd = 0.5, col = "grey50")
                                    ),
                                    t = 3, l = 1, r = ncol(tbl_grob))

# 添加表格底部的粗线（可选，视需要保留）
tbl_grob <- gtable::gtable_add_grob(tbl_grob,
                                    grobs = grid::segmentsGrob(
                                      y0 = unit(0, "npc"), y1 = unit(0, "npc"),
                                      gp = grid::gpar(lwd = 2, col = "black")
                                    ),
                                    t = nrow(tbl_grob), l = 1, r = ncol(tbl_grob))

# g_s7a <- plot_grid(p3, tbl_grob, ncol = 1, rel_heights = c(1, 0.75))
g_s7a <- ggdraw() +
  draw_plot(p3, x = 0, y = 0.42, width = 1, height = 0.58) +
  draw_plot(tbl_grob, x =0.08, y = 0, width = 0.92, height = 0.58)
ggsave(file.path(out_dir, 'figureS7_a.pdf'), g_s7a, width = 7.5, height = 5.2, device = cairo_pdf)

# =====================================================================
# Figure S7b: True Positive / Negative Rates by Clinical Subgroup
# =====================================================================
# Map detailed clinical subgroups safely
zr_fit_clean_stage <- zr_fit_clean %>%
  mutate(
    Sub_group = case_when(
      Subgroup %in% c('III', 'IV') ~ 'III/IV',
      Subgroup %in% c('I', 'II') ~ 'I/II',
      Subgroup %in% c('(2d)Tubular adenoma, ≥10 mm', '(2e)Serrated adenoma, ≥10 mm') ~ 'aa_size',
      Subgroup %in% c('(2c)Villous') ~ 'vil',
      Subgroup %in% c('(2b)High-grade dysplasia') ~ 'hgd',
      TRUE ~ NA_character_
    )
  )

target1_summary <- zr_fit_clean_stage %>%
  filter(Target == 1, !is.na(Sub_group)) %>%
  group_by(Sub_group) %>%
  summarise(
    Total_Freq = n(),
    `qFIT only` = sum(FIT_acc == 'right') / Total_Freq * 100,
    `rbcDNA only` = sum(rbcDNA_acc == 'right') / Total_Freq * 100,
    Total_detected = sum(final_label != 'Not detected') / Total_Freq * 100,
    .groups = 'drop'
  )

aaca_total <- zr_fit_clean_stage %>%
  filter(Target == 1) %>%
  summarise(
    Sub_group = "AACA",
    Total_Freq = n(),
    `qFIT only` = sum(FIT_acc == 'right') / Total_Freq * 100,
    `rbcDNA only` = sum(rbcDNA_acc == 'right') / Total_Freq * 100,
    Total_detected = sum(final_label != 'Not detected') / Total_Freq * 100
  )

legend_colors <- c("Detected by qFIT" = "#F39B7F", "Detected by rbcDNA" = "#91D1C2", "Combined qFIT and rbcDNA" = "#8491B4")

tmp123 <- bind_rows(aaca_total, target1_summary) %>%
  select(Sub_group, `qFIT only`, `rbcDNA only`, Total_detected) %>%
  pivot_longer(cols = -Sub_group, names_to = "variable", values_to = "value") %>%
  mutate(
    variable = case_when(
      variable == 'qFIT only' ~ 'Detected by qFIT',
      variable == 'rbcDNA only' ~ 'Detected by rbcDNA',
      variable == 'Total_detected' ~ 'Combined qFIT and rbcDNA'
    ),
    variable = factor(variable, levels = c('Detected by qFIT', 'Detected by rbcDNA', 'Combined qFIT and rbcDNA')),
    Sub_group = case_when(
      Sub_group == 'AACA' ~ "Advanced\nneoplasia",
      Sub_group == 'aa_size' ~ "Adenomas,\n≥10 mm",
      Sub_group == 'vil' ~ "Villous",
      Sub_group == 'hgd' ~ "High-grade\ndysplasia",
      Sub_group == 'I/II' ~ "CRC\n(I/II)",
      Sub_group == 'III/IV' ~ "CRC\n(III/IV)"
    ),
    Sub_group = factor(Sub_group, levels = c("Advanced\nneoplasia", "Adenomas,\n≥10 mm", "Villous", "High-grade\ndysplasia", "CRC\n(I/II)", "CRC\n(III/IV)"))
  ) %>% filter(!is.na(Sub_group))

shared_theme <- theme_classic(base_family = "ArialMT") + theme(
  legend.position = "none",
  axis.text.x = element_text(color = "black", size = 5, lineheight = 0.9, margin = margin(t = 5)),
  axis.text.y = element_text(color = "black", size = 5),
  axis.title.y = element_text(color = "black", size = 5, margin = margin(r = 8)),
  axis.title.x = element_blank(),
  axis.line = element_line(color = "black", linewidth = 0.2),
  axis.ticks = element_line(color = "black", linewidth = 0.2),
  plot.margin = margin(t = 10, r = 5, b = 5, l = 5)
)

p2 <- ggplot(data = tmp123, aes(x = Sub_group, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.8) +
  scale_fill_manual(values = legend_colors) +
  geom_vline(xintercept = 1.5, linetype = 'dotted', color = 'grey60') +
  scale_y_continuous(limits = c(0, 105), breaks = seq(0, 100, 25), expand = c(0, 0)) +
  ylab('True positive rate (%)') +
  geom_text(aes(label = sprintf("%.1f", value), y = value + 2), position = position_dodge(width = 0.8), size = 1.76, family = "ArialMT", color = "black") +
  shared_theme

tmp_hd <- zr_fit_clean_stage %>%
  filter(Target == 0) %>%
  summarise(
    Total_Freq = n(),
    `qFIT only` = sum(FIT_acc == 'right') / Total_Freq * 100,
    `rbcDNA only` = sum(rbcDNA_acc == 'right') / Total_Freq * 100,
    `rbcDNA+qFIT` = sum(final_label == 'rbcDNA+qFIT') / Total_Freq * 100
  ) %>%
  mutate(label = 'Non-AN') %>%
  select(label, `qFIT only`, `rbcDNA only`, `rbcDNA+qFIT`) %>%
  pivot_longer(cols = -label, names_to = "final_label", values_to = "HD") %>%
  mutate(
    final_label = case_when(
      final_label == 'qFIT only' ~ 'Detected by qFIT',
      final_label == 'rbcDNA only' ~ 'Detected by rbcDNA',
      final_label == 'rbcDNA+qFIT' ~ 'Combined qFIT and rbcDNA'
    ),
    final_label = factor(final_label, levels = c('Detected by qFIT', 'Detected by rbcDNA', 'Combined qFIT and rbcDNA'))
  )

p1 <- ggplot(data = tmp_hd, aes(x = label, y = HD, fill = final_label)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.8) +
  scale_fill_manual(values = legend_colors) +
  scale_y_continuous(limits = c(0, 105), breaks = seq(0, 100, 25), expand = c(0, 0)) +
  ylab('True negative rate (%)') +
  geom_text(aes(label = sprintf("%.1f", HD), y = HD + 2), position = position_dodge(width = 0.8), size = 1.76, family = "ArialMT", color = "black") +
  shared_theme

dummy_p <- ggplot(data = tmp_hd, aes(x = label, y = HD, fill = final_label)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = legend_colors) +
  theme_void(base_family = "ArialMT") +
  theme(
    legend.position = "top",
    legend.title = element_blank(),
    legend.text = element_text(size = 5, color = "black", margin = margin(r = 15)),
    legend.key.size = unit(0.2, "cm")
  )
shared_legend <- get_legend(dummy_p)

g_plots <- plot_grid(p1, p2, ncol = 2, rel_widths = c(1.8, 5.5), align = "h", axis = "b")
g_s7b <- plot_grid(shared_legend, g_plots, ncol = 1, rel_heights = c(0.1, 1))

ggsave(file.path(out_dir, 'figureS7_b.pdf'), g_s7b, width = 11, height = 4.5, device = cairo_pdf)

g_final <- plot_grid(g_s7a, g_s7b, ncol = 2, labels = c("A", "B"), label_size = 10, rel_widths = c(1, 1.6), label_fontface = "plain")

ggsave(file.path(out_dir, 'figureS7.pdf'), g_final, width = 8, height = 2.8, device = cairo_pdf)
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
  library(ggplot2)
  library(ggpubr)
  library(cowplot)
})

source(file.path(script_dir, 'plot_function.r'), chdir = TRUE)
source(file.path(script_dir, 'plotAUC_theme.r'), chdir = TRUE)

load('./Figures/prediction.RData')
load('./Figures/sampleinfo.RData')

zr_info <- merge(zr_clinc, zr_info, by = 'Sample')
colnames(zr_info) <- gsub('-| ', '', colnames(zr_info))

zr_info <- zr_info %>%
  mutate(
    Group_clean = case_when(
      Group == 'Control' | is.na(Group) ~ 'Non-AN',
      Group == 'AA' ~ 'AA',
      Group == 'CRC' ~ 'CRC',
      TRUE ~ 'Non-AN'
    ),
    Group = factor(Group_clean, levels = c('Non-AN', 'AA', 'CRC')),
    Target = ifelse(grepl('AA|CRC', Group), 1, 0)
  ) %>%
  rename_with(~"NLR", starts_with("nlr", ignore.case = TRUE)) %>%
  rename_with(~"PLR", starts_with("plr", ignore.case = TRUE)) %>%
  rename_with(~"NPS", starts_with("nps", ignore.case = TRUE)) %>%
  rename_with(~"LMR", starts_with("lmr", ignore.case = TRUE)) %>%
  rename_with(~"RBC", starts_with("RBC", ignore.case = FALSE)) %>%
  rename_with(~"HGB", starts_with("HGB", ignore.case = FALSE)) %>%
  rename_with(~"WBC", starts_with("WBC", ignore.case = FALSE)) %>%
  rename_with(~"PLT", starts_with("PLT", ignore.case = FALSE))

col_s9 <- c("Non-AN" = "#78909C", "AA" = "#DF8F44FF", "CRC" = "#9F1A1AFF")

plot_lifestyle <- function(df, x_var, x_lab) {
  df <- df %>% mutate(!!sym(x_var) := factor(.data[[x_var]], levels = c('Never', 'Current', 'Former', 'No record')))
  ggplot(df, aes(x = .data[[x_var]], y = merged_score)) +
    geom_boxplot(outlier.colour = NA) +
    theme_classic(base_family = "Arial") +
    theme(
      legend.position = 'none',
      strip.background = element_blank(),
      strip.text = element_text(size = 7, color = "black"),
      axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1, color = "black", size = 7),
      axis.text.y = element_text(color = "black", size = 7),
      axis.title = element_text(color = "black", size = 7),
      panel.border = element_blank(),
      axis.line = element_line(color = "black", linewidth = 0.2),
      axis.ticks = element_line(color = "black", linewidth = 0.2),
      plot.margin = margin(t = 5, r = 5, b = 5, l = 5)
    ) +
    scale_y_continuous(limits = c(0, 1.15), breaks = seq(0, 1, 0.25), expand = expansion(mult = c(0.05, 0.05))) +
    facet_grid(. ~ Group) +
    stat_compare_means(
      aes(label = paste0(after_stat(method), ", ", after_stat(p.signif))),
      method = "kruskal.test",
      label.x.npc = 0.5,
      hjust = 0.5,
      label.y = 1.05,
      size = 1.76,
      family = "ArialMT"
    ) +
    labs(y = 'rbcDNA predictive score', x = x_lab)
}

p1_smoking <- plot_lifestyle(zr_info, "Smokingstatus", 'History of smoking')
p2_alcohol <- plot_lifestyle(zr_info, "Alcoholconsumptionstatus", 'History of alcohol consumption')

g1 <- plot_grid(p1_smoking, p2_alcohol, ncol = 2, rel_widths = c(1, 1))
ggsave(file.path(out_dir, 'figureS9_a.pdf'), g1, width = 12.8, height = 5.4, device = cairo_pdf)

plot_comparison_box <- function(df, y_var, y_lab) {
  ggplot(data = df, aes(x = Group, y = .data[[y_var]], color = Group)) +
    geom_boxplot(outlier.shape = NA, width = 0.6) +
    geom_jitter(width = 0.2, size = 0.2, alpha = 0.6) +
    scale_color_manual(values = col_s9) +
    theme_classic(base_family = "Arial") +
    theme(
      legend.position = 'none',
      axis.title.x = element_blank(),
      axis.text.x = element_text(color = "black", size = 7),
      axis.text.y = element_text(color = "black", size = 7),
      axis.title.y = element_text(color = "black", size = 7),
      axis.line = element_line(color = "black", linewidth = 0.2),
      axis.ticks = element_line(color = "black", linewidth = 0.2),
      plot.margin = margin(t = 15, r = 5, b = 5, l = 5)
    ) +
    scale_y_continuous(expand = expansion(mult = c(0.05, 0.20))) +
    stat_compare_means(
      comparisons = list(c('Non-AN', 'AA'), c('AA', 'CRC'), c('Non-AN', 'CRC')),
      method = "wilcox.test",
      label = 'p.signif',
      step.increase = 0.12,
      vjust = 0.5,
      size = 1.76,
      family = "Arial"
    ) +
    labs(y = y_lab)
}

p_NLR <- plot_comparison_box(zr_info, "NLR", "NLR")
p_PLR <- plot_comparison_box(zr_info, "PLR", "PLR")
p_NPS <- plot_comparison_box(zr_info, "NPS", "NPS")
p_LMR <- plot_comparison_box(zr_info, "LMR", "LMR")

p_RBC <- plot_comparison_box(zr_info, "RBC", expression(paste("RBC (\u00D7", 10^{12}, "/L)")))
p_HGB <- plot_comparison_box(zr_info, "HGB", "HGB (g/L)")
p_WBC <- plot_comparison_box(zr_info, "WBC", expression(paste("WBC (\u00D7", 10^9, "/L)")))
p_PLT <- plot_comparison_box(zr_info, "PLT", expression(paste("PLT (\u00D7", 10^9, "/L)")))

g2 <- plot_grid(p_NLR, p_PLR, p_NPS, p_LMR, ncol = 4, align = "h", axis = "tb")
g3 <- plot_grid(p_RBC, p_HGB, p_WBC, p_PLT, ncol = 4, align = "h", axis = "tb")

ggsave(file.path(out_dir, 'figureS9_bc.pdf'), plot_grid(g2, g3, ncol = 1, rel_heights = c(1, 1)), width = 12.8, height = 8.3, device = cairo_pdf)

g_final <- plot_grid(g1, g2, g3, ncol = 1, rel_heights = c(1.2, 1, 1), labels = c("A", "B", "C"), label_size = 10, label_fontfamily = "Arial", label_fontface = "plain")
ggsave(file.path(out_dir, 'figureS9_all.pdf'), g_final, width = 8, height = 8.5, device = cairo_pdf)
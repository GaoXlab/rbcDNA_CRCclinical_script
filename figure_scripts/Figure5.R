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
  library(stringr)
  library(patchwork)
})

source(file.path(script_dir, 'plot_function.r'), chdir = TRUE)

load('./Figures/prediction.RData')
load('./Figures/sampleinfo.RData')

cutoff_spe90 <- Cutoff(0.90, test)

group_cols <- c("Control" = ggsci::pal_material("blue-grey")(10)[5],
                "AA" = "#DF8F44FF",
                "CRC" = "#9F1A1AFF")

CRC_color <- c("#F45B5BFF", "#E63131FF", "#C51D1DFF", "#9F1A1AFF", "#821D1DFF", "#450A0AFF")

zr_info <- merge(zr_info, zr_clinc[, c('Sample', 'merged_score')], by = 'Sample')

colnames(zr_info) <- gsub('-| ', '', colnames(zr_info))

zr_info <- zr_info %>%
  rename_with(~"rbcDNA_concentration", matches("rbcDNA_concentration", ignore.case = TRUE)) %>%
  rename_with(~"Age", matches("^Age", ignore.case = TRUE)) %>%
  rename_with(~"RBC", matches("^RBC\\(", ignore.case = TRUE)) %>%
  rename_with(~"HGB", matches("^HGB\\(", ignore.case = TRUE)) %>%
  rename_with(~"WBC", matches("^WBC\\(", ignore.case = TRUE)) %>%
  rename_with(~"PLT", matches("^PLT\\(", ignore.case = TRUE)) %>%
  mutate(Group = factor(Group, levels = c('Control', 'AA', 'CRC')))

# Figure 5a: rbcDNA Concentration vs Predictive Score
p1_rbccon <- ggplot(zr_info, aes(x = rbcDNA_concentration, y = merged_score, color = Group, fill = Group)) +
  geom_point(size = 1.2, alpha = 0.5, shape = 16) +
  geom_smooth(method = lm, linetype = "longdash", se = FALSE, fullrange = TRUE, linewidth = 0.4) +
  ggpubr::stat_cor(aes(color = Group), method = "pearson", label.x = 8.8, label.y = c(0.96, 0.88, 0.8), size = 7 / .pt, show.legend = FALSE) +
  scale_color_manual(values = group_cols, labels = c("Control" = "Non-AN", "AA" = "AA", "CRC" = "CRC")) +
  scale_fill_manual(values = group_cols, labels = c("Control" = "Non-AN", "AA" = "AA", "CRC" = "CRC")) +
  scale_y_continuous(breaks = c(0, 0.25, 0.50, 0.75, 1.00)) +
  coord_cartesian(ylim = c(-0.02, 1.05)) +
  labs(x = "rbcDNA concentration (ng/mL)", y = "rbcDNA predictive score") +
  theme_classic() +
  theme(
    legend.position = "top",
    legend.title = element_blank(),
    legend.text = element_text(size = 7, color = "black"),
    legend.box.margin = margin(b = -10),
    axis.text = element_text(color = "black", size = 7),
    axis.title = element_text(color = "black", size = 7),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.4),
    axis.line = element_blank()
  )

# ggsave(file.path(out_dir, 'figure5_a.pdf'), p1_rbccon, width = 3.8, height = 4.0, device = cairo_pdf)

# Figure 5b, c: Demographics (Age and Sex)
zr_info <- zr_info %>%
  mutate(
    age_group = case_when(
      Age < 50 ~ '<50',
      Age >= 50 & Age < 65 ~ '50-65',
      Age >= 65 ~ '≥65'
    ),
    age_group = factor(age_group, levels = c('<50', '50-65', '≥65')),
    Sex = factor(Sex, levels = c('Female', 'Male'))
  )

facet_labels <- as_labeller(c("Control" = "Non-AN", "AA" = "AA", "CRC" = "CRC"))

p1_score_age <- ggplot(data = zr_info, aes(x = age_group, y = merged_score, color = age_group)) +
  geom_boxplot(outlier.colour = NA, width = 0.6) +
  theme_classic(base_line_size = 0.4) +
  scale_color_manual(values = c(ggsci::pal_material("blue-grey")(10)[c(5, 7, 9)])) +
  scale_y_continuous(breaks = seq(0, 1, by = 0.25)) +
  facet_grid(. ~ Group, labeller = facet_labels) +
  stat_compare_means(aes(label = paste0(after_stat(method), ",\n", after_stat(p.signif))),
                     method = "kruskal.test", label.x.npc = 'center', label.y = 1.02, size = 5 / .pt, hjust = 0.5) +
  labs(x = 'Age (years)', y = 'rbcDNA predictive score') +
  coord_cartesian(ylim = c(0, 1.1), clip = "off") +
  theme(
    legend.position = 'none',
    strip.background = element_blank(),
    strip.text = element_text(size = 7, color = "black", margin = margin(b = 8)),
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1, color = "black", size = 7),
    axis.text.y = element_text(color = "black", size = 7),
    axis.title = element_text(color = "black", size = 7),
    axis.line = element_line(color = "black", linewidth = 0.2, lineend = "square"),
    axis.ticks = element_line(color = "black", linewidth = 0.2)
  )

p1_score_gender <- ggplot(data = zr_info, aes(x = Sex, y = merged_score, color = Sex)) +
  geom_boxplot(outlier.colour = NA, width = 0.6) +
  theme_classic(base_line_size = 0.4) +
  scale_color_manual(values = c(ggsci::pal_material("blue-grey")(10)[c(7, 9)])) +
  scale_y_continuous(breaks = seq(0, 1, by = 0.25)) +
  facet_grid(. ~ Group, labeller = facet_labels) +
  stat_compare_means(aes(label = paste0(after_stat(method), ",\n", after_stat(p.signif))),
                     method = "wilcox.test", label.x.npc = 'center', label.y = 1.02, size = 5 / .pt, hjust = 0.5) +
  labs(x = 'Sex', y = 'rbcDNA predictive score') +
  coord_cartesian(ylim = c(0, 1.1), clip = "off") +
  theme(
    legend.position = 'none',
    strip.background = element_blank(),
    strip.text = element_text(size = 7, color = "black", margin = margin(b = 8)),
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1, color = "black", size = 7),
    axis.text.y = element_text(color = "black", size = 7),
    axis.title = element_text(color = "black", size = 7),
    axis.line = element_line(color = "black", linewidth = 0.2, lineend = "square"),
    axis.ticks = element_line(color = "black", linewidth = 0.2)
  )

# ggsave(file.path(out_dir, 'figure5_bc.pdf'),
#        plot_grid(p1_score_age, p1_score_gender, ncol = 2, rel_widths = c(1.3, 1), align = "h", axis = "b"),
#        width = 8.5, height = 4.5, device = cairo_pdf)

# Figure 5d: Non-AN Controls Subgroup Breakdown
zr_clinfo_hd <- zr_info %>%
  filter(Group == 'Control' & Subgroup != '(other CA)') %>%
  mutate(
    Control_subgroup_new = case_when(
      Control_subgroup == 'Negative colonoscopy' ~ 'No lesions',
      Control_subgroup == 'Nonneoplastic findings' ~ 'Nonneoplastic findings',
      Control_subgroup == 'Non-advanced adenoma (NAA)' ~ 'Non-advanced adenomas',
      Control_subgroup == 'inflammation' ~ 'Colitis / Proctitis',
      Control_subgroup == '(other disease)' ~ 'Other colorectal disease',
      TRUE ~ Control_subgroup
    ),
    Control_subgroup_new = factor(Control_subgroup_new, levels = c(
      'No lesions', 'Nonneoplastic findings', 'Non-advanced adenomas',
      'Colitis / Proctitis', 'Other colorectal disease'
    ))
  )

p_hd <- ggplot(zr_clinfo_hd, aes(x = Control_subgroup_new, y = merged_score)) +
  geom_boxplot(fill = ggsci::pal_material("blue-grey")(10)[5], width = 0.5, outlier.colour = NA, alpha = 0.8) +
  geom_jitter(width = 0.2, size = 0.5, color = 'grey50', shape = 16) +
  theme_classic() +
  scale_y_continuous(breaks = c(0, 0.25, 0.50, 0.75, 1.00)) +
  coord_cartesian(ylim = c(-0.02, 1.05)) +
  geom_hline(yintercept = cutoff_spe90, color = "red4", linetype = "dashed", linewidth = 0.4) +
  ylab('rbcDNA predictive score') +
  xlab(NULL) +
  stat_compare_means(aes(label = paste0(after_stat(method), ", ", after_stat(p.signif))),
                     method = "kruskal.test", label.x = 0.55, label.y = 1.02, size = 5 / .pt, hjust = 0) +
  theme(
    legend.position = 'none',
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1, color = "black", size = 7),
    axis.text.y = element_text(color = "black", size = 7),
    axis.title.y = element_text(color = "black", size = 7),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.4),
    axis.line = element_blank()
  )

# ggsave(file.path(out_dir, 'figure5_d.pdf'), p_hd, width = 3.2, height = 4.5, device = cairo_pdf)

g_abcd <- plot_grid(
  p1_rbccon,
  p1_score_age,
  p1_score_gender,
  p_hd,
  nrow = 1,
  align = "h",
  axis = "tb",
  rel_widths = c(1, 1, 0.72, 0.72)
)

# ggsave(file.path(out_dir, 'figure5_abcd_combined.pdf'), g_abcd, width = 12.8, height = 4.8, device = cairo_pdf)

unified_boxplot_theme <- theme_classic(base_family = "ArialMT") +
  theme(
    legend.position = 'none',
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1, color = "black", size = 7),
    axis.text.y = element_text(color = "black", size = 7),
    axis.title.x = element_text(color = "black", size = 7),
    axis.title.y = element_text(color = "black", size = 7),
    plot.title = element_text(hjust = 0.5, size = 7, color = "black"),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.2),
    axis.line = element_line(color = "black", linewidth = 0.2, lineend = "square"),
    axis.ticks = element_line(color = "black", linewidth = 0.2),
    plot.margin = margin(t = 10, r = 2, b = 5, l = 2)
  )

hide_y_axis <- theme(axis.text.y = element_blank(), axis.ticks.y = element_blank(), axis.title.y = element_blank())

# Figure 5e: CRC Clinical Stage & TNM
zr_clinfo_crc_e <- zr_info %>%
  filter(Group == 'CRC') %>%
  mutate(
    Subgroup_new = factor(ifelse(Subgroup == '/', 'Unknown', as.character(Subgroup)), levels = c('I', 'II', 'III', 'IV', 'Unknown')),
    T_lab = factor(case_when(T %in% c('1', '2') ~ 'T1-T2', T %in% c('3', '4') ~ 'T3-T4', TRUE ~ NA_character_), levels = c('T1-T2', 'T3-T4')),
    N_lab_new = factor(case_when(N == '0' ~ 'N0', N %in% c('1', '2') ~ 'N>1', TRUE ~ NA_character_), levels = c('N0', 'N>1')),
    M_new = factor(case_when(M == '0' ~ 'M0', M == '1' ~ 'M1', TRUE ~ NA_character_), levels = c('M0', 'M1'))
  )

plot_e <- function(df, x_var, fill_colors, x_label, method, show_y = FALSE) {
  p <- ggplot(df %>% filter(!is.na(.data[[x_var]])), aes(x = .data[[x_var]], y = merged_score)) +
    geom_boxplot(aes(fill = .data[[x_var]]), width = 0.5, outlier.colour = NA, alpha = 0.8) +
    geom_jitter(width = 0.2, size = 0.5, color = 'grey50', shape = 16) +
    scale_fill_manual(values = fill_colors) + scale_y_continuous(breaks = seq(0, 1, by = 0.25)) +
    coord_cartesian(ylim = c(-0.02, 1.15), clip = "off") +
    geom_hline(yintercept = cutoff_spe90, color = "red4", linetype = "dashed", linewidth = 0.4) +
    stat_compare_means(aes(label = paste0(after_stat(method), ", ", after_stat(p.signif))),
                       method = method,
                       label.x = 0.55,
                       label.y = 1.15,
                       hjust = 0,
                       vjust = 1,
                       size = 5 / .pt) +
    labs(x = x_label, y = 'rbcDNA predictive score') +
    ggtitle(" ") +
    unified_boxplot_theme

  if (!show_y) p <- p + hide_y_axis
  return(p)
}

p_stage <- plot_e(zr_clinfo_crc_e, "Subgroup_new", c("#FCAE91", "#FB6A4A", "#DE2D26", "#A50F15", "#67000D"), "Stage", "kruskal.test", TRUE)
p_t <- plot_e(zr_clinfo_crc_e, "T_lab", c("#FB6A4A", "#CB181D"), "T", "wilcox.test", FALSE)
p_n <- plot_e(zr_clinfo_crc_e, "N_lab_new", c("#FB6A4A", "#CB181D"), "N", "wilcox.test", FALSE)
p_m <- plot_e(zr_clinfo_crc_e, "M_new", c("#FB6A4A", "#CB181D"), "M", "wilcox.test", FALSE)

# Figure 5f: Lesion Size
zr_size_aa <- zr_info %>% filter(Group == 'AA', !is.na(Sizegroup), Sizegroup != '/', Sizegroup != 'Missing') %>%
  mutate(Sizegroup_clean = factor(case_when(
    Sizegroup == '<1cm' ~ '<1', Sizegroup == '1-2cm' ~ '1-2', Sizegroup == '2-3cm' ~ '2-3', Sizegroup == '>=3cm' ~ '≥3', TRUE ~ NA_character_
  ), levels = c('<1', '1-2', '2-3', '≥3'))) %>% filter(!is.na(Sizegroup_clean))

zr_size_crc <- zr_info %>% filter(Group == 'CRC', !is.na(Sizegroup), Sizegroup != '/', Sizegroup != 'Missing') %>%
  mutate(Sizegroup_clean = factor(case_when(
    Sizegroup == '<3cm' ~ '<3', Sizegroup == '3-5cm' ~ '3-5', Sizegroup == '>=5cm' ~ '≥5', TRUE ~ NA_character_
  ), levels = c('<3', '3-5', '≥5'))) %>% filter(!is.na(Sizegroup_clean))

plot_fg <- function(df, x_var, fill_colors, x_label, title_text, method = "kruskal.test", show_y = FALSE) {
  p <- ggplot(df %>% filter(!is.na(.data[[x_var]])), aes(x = .data[[x_var]], y = merged_score)) +
    geom_boxplot(aes(fill = .data[[x_var]]), width = 0.5, outlier.colour = NA, alpha = 0.8) +
    geom_jitter(width = 0.2, size = 0.5, color = 'grey50', shape = 16) +
    scale_fill_manual(values = fill_colors) + scale_y_continuous(breaks = seq(0, 1, by = 0.25)) +
    coord_cartesian(ylim = c(-0.02, 1.15), clip = "off") +
    geom_hline(yintercept = cutoff_spe90, color = "red4", linetype = "dashed", linewidth = 0.4) +
    stat_compare_means(aes(label = paste0(after_stat(method), ", ", after_stat(p.signif))),
                       method = method,
                       label.x = 0.55,
                       label.y = 1.15,
                       hjust = 0,
                       vjust = 1,
                       size = 5 / .pt) +
    labs(x = x_label, y = 'rbcDNA predictive score') + ggtitle(title_text) +
    unified_boxplot_theme

  if (!show_y) {
    p <- p + hide_y_axis
  }
  return(p)
}

col_size <- c("<1"="#FDD0A2", "1-2"="#FDAE6B", "2-3"="#FD8D3C", "≥3"="#E6550D", "<3"="#FCBBA1", "3-5"="#EF3B2C", "≥5"="#A50F15")

p_size_aa <- plot_fg(zr_size_aa, "Sizegroup_clean", col_size, "Lesion size (cm)", "AA", show_y = TRUE)
p_size_crc <- plot_fg(zr_size_crc, "Sizegroup_clean", col_size, "Lesion size (cm)", "CRC", show_y = FALSE)

# Figure 5g: Tumor Location
clean_loc <- function(loc) case_when(grepl("Proximal", loc) ~ "Proximal\ncolon", grepl("Distal", loc) ~ "Distal\ncolon", grepl("Rectum", loc) ~ "Rectum", TRUE ~ NA_character_)

zr_loc_aa <- zr_info %>% filter(Group == 'AA', !is.na(Tumorlocation), Tumorlocation != '/') %>%
  mutate(Tumorlocation_clean = factor(clean_loc(Tumorlocation), levels = c('Proximal\ncolon', 'Distal\ncolon', 'Rectum'))) %>% filter(!is.na(Tumorlocation_clean))

zr_loc_crc <- zr_info %>% filter(Group == 'CRC', !is.na(Tumorlocation), Tumorlocation != '/') %>%
  mutate(Tumorlocation_clean = factor(clean_loc(Tumorlocation), levels = c('Proximal\ncolon', 'Distal\ncolon', 'Rectum'))) %>% filter(!is.na(Tumorlocation_clean))

col_loc_aa <- c("Proximal\ncolon"="#FDD0A2", "Distal\ncolon"="#FDAE6B", "Rectum"="#FD8D3C")
col_loc_crc <- c("Proximal\ncolon"="#FCBBA1", "Distal\ncolon"="#EF3B2C", "Rectum"="#A50F15")

p_loc_aa <- plot_fg(zr_loc_aa, "Tumorlocation_clean", col_loc_aa, "Tumor location", "AA", show_y = TRUE)
p_loc_crc <- plot_fg(zr_loc_crc, "Tumorlocation_clean", col_loc_crc, "Tumor location", "CRC", show_y = FALSE)

g_efg_combined <- p_stage + p_t + p_n + p_m + p_size_aa + p_size_crc + p_loc_aa + p_loc_crc +
  plot_layout(nrow = 1, widths = c(2, 0.8, 0.8, 0.8, 1.6, 1.2, 1.4, 1.4))

# ggsave(file.path(out_dir, 'figure5_efg_combined.pdf'), g_efg_combined, width = 12.8, height = 4.5, device = cairo_pdf)

# Figure 5h, 5i: CBC Indices Scatter Plots
plot_scatter <- function(df, x_var, x_lab_expr, show_y_title = FALSE) {
  ggplot(df, aes(x = .data[[x_var]], y = merged_score, color = Group, fill = Group)) +
    geom_point(size = 1.2, alpha = 0.4, shape = 16) +
    geom_smooth(method = lm, linetype = "longdash", se = FALSE, fullrange = TRUE, linewidth = 0.4) +
    ggpubr::stat_cor(aes(color = Group), method = 'spearman',
                     label.x.npc = 0.30,
                     hjust = 0,
                     label.y = c(0.98, 0.88, 0.78),
                     size = 7 / .pt, show.legend = FALSE, family = "Arial") +
    scale_color_manual(values = group_cols, labels = c("Control" = "Non-AN", "AA" = "AA", "CRC" = "CRC")) +
    scale_fill_manual(values = group_cols, labels = c("Control" = "Non-AN", "AA" = "AA", "CRC" = "CRC")) +
    scale_y_continuous(breaks = c(0, 0.25, 0.50, 0.75, 1.00)) +
    coord_cartesian(ylim = c(-0.02, 1.05)) +
    labs(x = x_lab_expr, y = if(show_y_title) "rbcDNA predictive score" else NULL) +
    theme_classic(base_family = "Arial") +
    theme(
      legend.position = "none",
      axis.text = element_text(color = "black", size = 7),
      axis.title.x = element_text(color = "black", size = 7, margin = margin(t = 8)),
      axis.title.y = element_text(color = "black", size = 7, margin = margin(r = 8)),
      panel.border = element_rect(color = "black", fill = NA, linewidth = 0.4),
      axis.ticks = element_line(color = "black", linewidth = 0.4),
      axis.line = element_blank(),
      plot.margin = margin(t = 10, r = 5, b = 10, l = 5)
    )
}

p_nlr <- plot_scatter(zr_info, "NLR", "NLR", show_y_title = TRUE)
p_plr <- plot_scatter(zr_info, "PLR", "PLR")
p_nps <- plot_scatter(zr_info, "NPS", "NPS")
p_lmr <- plot_scatter(zr_info, "LMR", "LMR")

p_rbc <- plot_scatter(zr_info, "RBC", expression(paste("RBC (\u00D7", 10^12, "/L)")), show_y_title = TRUE)
p_hgb <- plot_scatter(zr_info, "HGB", "HGB (g/L)")
p_wbc <- plot_scatter(zr_info, "WBC", expression(paste("WBC (\u00D7", 10^9, "/L)")))
p_plt <- plot_scatter(zr_info, "PLT", expression(paste("PLT (\u00D7", 10^9, "/L)")))

dummy_p <- ggplot(zr_info, aes(x = RBC, y = merged_score, color = Group)) +
  geom_point(size = 1.5, alpha = 0.6, shape = 16) +
  geom_smooth(method = lm, linetype = "longdash", se = FALSE, linewidth = 0.4) +
  scale_color_manual(values = group_cols, labels = c("Control" = "Non-AN", "AA" = "AA", "CRC" = "CRC")) +
  theme_classic(base_family = "Arial") +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 7, color = "black", margin = margin(r = 1, unit = "cm")),
    legend.key.width = unit(1, "cm"),
    legend.background = element_blank(),
    legend.box.margin = margin(t = -5, b = 5)
  )
shared_legend <- get_legend(dummy_p)

row_h <- p_nlr + p_plr + p_nps + p_lmr + plot_layout(nrow = 1)
row_i <- p_rbc + p_hgb + p_wbc + p_plt + plot_layout(nrow = 1)

combined_scatter <- row_h / row_i

final_hi <- plot_grid(combined_scatter, shared_legend, ncol = 1, rel_heights = c(1, 0.08))

# ggsave(file.path(out_dir, 'figure5_hi_combined.pdf'), final_hi, width = 12.8, height = 7.2, device = cairo_pdf)

final_all <- plot_grid(
  g_abcd,
  g_efg_combined,
  final_hi,
  ncol = 1,
  rel_heights = c(1, 0.9, 1.5)
)
ggsave(file.path(out_dir, 'Figure5_all.pdf'), final_all, width = 8, height = 10.6, device = cairo_pdf)
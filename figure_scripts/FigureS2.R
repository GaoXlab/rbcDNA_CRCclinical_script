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
  library(readr)
  library(ggpubr)
  library(cowplot)
  library(stringr)
  library(pROC)
  library(patchwork)
})

source(file.path(script_dir, 'plot_function.r'), chdir = TRUE)
source(file.path(script_dir, 'plotAUC_theme.r'), chdir = TRUE)

load('./Figures/sampleinfo.RData')

# Figure S2a, S2b: Cohort Demographics and Blood Counts
trn_test <- bind_rows(
  trn_info %>% mutate(dataset = 'Discovery'),
  test_info %>% mutate(dataset = 'Test')
) %>%
  mutate(
    Group_clean = case_when(
      Group == 'Control' | is.na(Group) ~ 'Non-AN',
      Group == 'AA' ~ 'AA',
      Group == 'CRC' ~ 'CRC',
      TRUE ~ 'Non-AN'
    ),
    Group_clean = factor(Group_clean, levels = c('Non-AN', 'AA', 'CRC')),
    dataset = factor(dataset, levels = c('Discovery', 'Test'))
  ) %>%
  rename_with(~"Age", starts_with("Age")) %>%
  rename_with(~"RBC", starts_with("RBC")) %>%
  rename_with(~"WBC", starts_with("WBC")) %>%
  rename_with(~"PLT", starts_with("PLT")) %>%
  rename_with(~"HGB", starts_with("HGB"))

col_s2 <- c("Non-AN" = "#78909C", "AA" = "#DF8F44FF", "CRC" = "#9F1A1AFF")

plot_clinical_boxplot <- function(df, y_var, y_label) {
  ggplot(data = df, aes(x = Group_clean, y = .data[[y_var]], color = Group_clean)) +
    geom_boxplot(fill = NA, outlier.colour = NA) +
    geom_jitter(width = 0.3, size = 0.05, alpha = 0.7, shape = 16) +
    scale_color_manual(values = col_s2) +
    facet_grid(. ~ dataset) +
    scale_y_continuous(expand = expansion(mult = c(0.05, 0.05))) +
    stat_compare_means(aes(label = paste0(after_stat(method), "\n", after_stat(p.signif))),
                       method = "kruskal.test", label.x.npc = 0.5, label.y.npc = 1,
                       size = 5 / .pt, hjust = 0.5, vjust = 1, family = "ArialMT") +
    labs(x = NULL, y = y_label) +
    theme_classic(base_family = "Arial") +
    theme(
      legend.position = 'none',
      strip.background = element_blank(),
      strip.text = element_text(size = 7, color = "black", margin = margin(b = 5)),
      axis.text.x = element_text(angle = 45, hjust = 1, color = "black", size = 7),
      axis.text.y = element_text(color = "black", size = 7),
      axis.title.y = element_text(color = "black", size = 7, margin = margin(r = 2)),
      axis.title.x = element_text(color = "black", size = 7),
      panel.border = element_blank(),
      axis.line = element_line(color = "black", linewidth = 0.2),
      axis.ticks = element_line(color = "black", linewidth = 0.2),
      plot.margin = margin(t = 10, r = 5, b = 5, l = 5)
    )
}

p_age <- plot_clinical_boxplot(trn_test, "Age", 'Age (years)')
p_rbc <- plot_clinical_boxplot(trn_test, "RBC", expression(paste("RBC (\u00D7", 10^12, "/L)")))
p_wbc <- plot_clinical_boxplot(trn_test, "WBC", expression(paste("WBC (\u00D7", 10^9, "/L)")))
p_plt <- plot_clinical_boxplot(trn_test, "PLT", expression(paste("PLT (\u00D7", 10^9, "/L)")))

g_ab <- p_age + p_rbc + p_wbc + p_plt + plot_layout(nrow = 1, widths = c(1, 1, 1, 1))
# ggsave(file.path(out_dir, 'figureS2_ab.pdf'), g_ab, width = 14, height = 4.5, device = cairo_pdf)

# Figure S2c: Mitochondrial (MT) Read Proportion Analysis
read_mt_log <- function(filepath) {
  df <- read.table(filepath, sep = '\t', header = TRUE)
  colnames(df)[1] <- 'seqID'
  df$seqID <- gsub('.nodup.q30.bam', '', sapply(strsplit(as.character(df$seqID), "/"), tail, n = 1))
  return(na.omit(df))
}

all_mt_data <- read_mt_log('./Figures/TotalSample_MT.noalt.log')

gDNA_ids <- c('GLGHD1069', 'GLGHD0046', 'GLGHD0053', 'GLGHD0014', 'GLGHD0015',
              'GLGHD0058', 'GLGHD0822', 'GLGHD1049', 'GLGHD1111', 'GLGHD0068')
gDNA_reads <- all_mt_data %>%
  filter(seqID %in% gDNA_ids) %>%
  distinct() %>%
  mutate(
    perc_mt = as.numeric(mt) / as.numeric(total_reads),
    dataset = 'Control',
    datGroup = 'Control'
  ) %>%
  select(seqID, dataset, perc_mt, datGroup)

summary_reads <- bind_rows(
  trn_info %>% select(Sample) %>% mutate(dataset = 'trn'),
  test_info %>% select(Sample) %>% mutate(dataset = 'internaltest'),
  sd_info %>% select(Sample) %>% mutate(dataset = 'sd'),
  wz_info %>% select(Sample) %>% mutate(dataset = 'wz'),
  zr_info %>% select(Sample) %>% mutate(dataset = 'zr')
)

mt_cfdna_info <- all_mt_data %>%
  select(seqID, mt, total_reads) %>%
  mutate(perc_mt = as.numeric(mt) / as.numeric(total_reads))

summary_reads <- summary_reads %>%
  left_join(mt_cfdna_info, by = c("Sample" = "seqID")) %>%
  mutate(
    datGroup = case_when(
      dataset %in% c('trn', 'internaltest', 'test') ~ 'Development',
      dataset %in% c('wz', 'sd') ~ 'Validation',
      dataset == 'zr' ~ 'Clinical',
      TRUE ~ ''
    )
  ) %>%
  select(seqID = Sample, dataset, perc_mt, datGroup)

plot_df <- bind_rows(gDNA_reads, summary_reads) %>%
  mutate(
    dataset_clean = case_when(
      dataset == 'Control' ~ 'Leukocyte DNA',
      dataset == 'trn' ~ 'Discovery',
      dataset == 'internaltest' ~ 'Internal test',
      dataset == 'wz' ~ 'WENZHOU',
      dataset == 'sd' ~ 'SHANDONG',
      dataset == 'zr' ~ 'Clinical',
      TRUE ~ dataset
    ),
    dataset_clean = factor(dataset_clean, levels = c('Leukocyte DNA', 'Discovery', 'Internal test', 'WENZHOU', 'SHANDONG', 'Clinical'))
  ) %>% filter(!is.na(perc_mt))

n_labels <- plot_df %>%
  group_by(dataset_clean) %>%
  summarize(n = n(), y_pos = max(perc_mt * 100, na.rm = TRUE) + 0.015)

p_mt <- ggplot(data = plot_df, aes(x = dataset_clean, y = perc_mt * 100, fill = dataset_clean)) +
  geom_hline(yintercept = 0.04,linetype = "dashed", color='grey') +
  geom_boxplot(width = 0.8, outlier.shape = NA, alpha = 0.5) +
  geom_jitter(width = 0.3, size = 0.3, alpha = 0.4, shape = 16) +
  geom_text(data = n_labels, aes(x = dataset_clean, y = y_pos, label = paste0("n=", n)),
            size = 5 / .pt, family = "Arial", inherit.aes = FALSE) +
  scale_y_continuous(breaks = seq(0, 0.15, 0.05), expand = expansion(mult = c(0.0, 0.0))) +
  labs(y = 'Proportion of rbcDNA or leukocyte DNA\nmapped to MT regions (%)', x = NULL) +
  scale_fill_manual(values = c("Leukocyte DNA" = "#BDBDBD", "Discovery" = "#AE3235",
                               "Internal test" = "#BC3C29CC", "WENZHOU" = '#293E90',
                               "SHANDONG" = '#478AC9', "Clinical" = '#3C6743')) +
  theme_classic(base_family = "Arial") +
  coord_cartesian(ylim = c(0, 0.16), clip = "off") +
  annotate("text", x = 2.5, y = 0.165, label = "Development", size = 7 / .pt, family = "Arial") +
  annotate("text", x = 5, y = 0.165, label = "Validation", size = 7 / .pt, family = "Arial") +
  theme(
    legend.position = 'none',
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1, color = "black", size = 5),
    axis.text.y = element_text(color = "black", size = 7),
    axis.title.y = element_text(color = "black", size = 7, margin = margin(r = 10)),
    axis.line = element_line(color = "black", linewidth = 0.2),
    axis.ticks = element_line(color = "black", linewidth = 0.2),
    plot.margin = margin(t = 20, r = 5, b = 35, l = 5)
  )

# ggsave(file.path(out_dir, 'figureS2_c.pdf'), p_mt, width = 5.5, height = 4.5, device = cairo_pdf)

# Figure S2d: Chromosome Arm Coverage Smoothing
load('./Figures/r_g_10controls_smooth.RData')
load('./Figures/trn_nonAN_smooth_anno.RData')

arm_levels <- c("1p","1q","2p","2q","3p","3q","4p","4q","5p","5q","6p","6q","7p","7q","8p","8q", "9p","9q","10p","10q","11p","11q","12p","12q","13q","14q","15q","16p","16q","17p","17q","18p", "18q","19p","19q","20p","20q","21q","22q")

colnames(df_HD_1000k_anno)[grep('chr', colnames(df_HD_1000k_anno))] <- 'chromosome'

HD10_gDNA_1000kb <- merge(HD10_gDNA_1000kb, df_HD_1000k_anno[, c('feature', 'arm')], by = 'feature') %>%
  mutate(SampleType = 'Control_gDNA', arm = factor(arm, levels = arm_levels))

HD10_rbcDNA_1000kb <- merge(HD10_rbcDNA_1000kb, df_HD_1000k_anno[, c('feature', 'arm')], by = 'feature') %>%
  mutate(SampleType = 'Control_rbcDNA', arm = factor(arm, levels = arm_levels))

df_HD_1000k_anno <- df_HD_1000k_anno %>%
  mutate(SampleType = 'trn_HD', arm = factor(arm, levels = arm_levels), broadPeak_y = -4)

used_cols <- intersect(colnames(HD10_rbcDNA_1000kb), colnames(HD10_gDNA_1000kb))
coverage_data <- bind_rows(
  HD10_rbcDNA_1000kb[, used_cols],
  HD10_gDNA_1000kb[, used_cols],
  df_HD_1000k_anno[, used_cols]
) %>%
  mutate(SampleType = factor(SampleType, levels = c('Control_rbcDNA', 'Control_gDNA', 'trn_HD')))

track_theme <- theme_classic(base_family = "Arial") + theme(
  panel.grid = element_blank(),
  panel.spacing = unit(0.02, "inches"),
  panel.background = element_rect(fill = "#F5F5F5", color = NA),
  panel.border = element_blank(),
  axis.line = element_line(color = "black", linewidth = 0.2),
  strip.background = element_blank(),
  strip.text = element_blank(),
  axis.text.x = element_blank(),
  axis.ticks.x = element_blank(),
  axis.text.y = element_text(size = 7, color = "black"),
  axis.title.y = element_text(size = 7, color = "black"),
  plot.title = element_text(size = 7, hjust = 0.01, margin = margin(b = 2), face = "plain"),
  plot.margin = margin(t = 5, r = 5, b = 2, l = 5)
)

build_track <- function(df, stype, col_fill, col_line, title_str, show_y_title=FALSE) {
  ggplot(df %>% filter(SampleType == stype)) +
    geom_hline(yintercept = 1, linewidth = 0.4, linetype = 'dashed', color = 'grey50') +
    geom_ribbon(aes(x = start, ymin = min, ymax = max), fill = col_fill, alpha = 0.6) +
    geom_line(aes(x = start, y = median, group = 1), color = col_line, linewidth = 0.3) +
    scale_y_continuous(limits = c(0.4, 2.1), breaks = c(0.5, 1.0, 1.5, 2.0)) +
    facet_grid(. ~ arm, space = "free_x", scales = "free_x") +
    labs(title = title_str, y = if(show_y_title) "Median read counts" else NULL, x = NULL) +
    track_theme +
    theme(
      plot.title.position = "panel",
      plot.title = element_text(
        size = 7,
        hjust = 0,
        vjust = 1,
        margin = margin(t = 2, l = 5, b = -12),
        family = "Arial"
      )
    )
}

heatmap_theme <- theme_classic(base_family = "Arial") + theme(
  axis.line = element_line(color = "black", linewidth = 0.2),
  axis.text = element_blank(),
  axis.ticks = element_blank(),
  panel.grid = element_blank(),
  panel.spacing = unit(0.02, "inches"),
  panel.border = element_blank(),
  panel.background = element_rect(fill = "#F5F5F5", color = NA),
  strip.background = element_blank(),
  strip.placement = "outside",
  strip.text.x = element_text(size = 5, angle = 45, color = "black", margin = margin(t = 3, b = 3)),
  axis.title.y = element_text(size = 5, color = "black", angle = 0, hjust = 1, vjust = 0.5, margin = margin(r=5)),
  axis.title.x = element_text(size = 7, color = "black", margin = margin(t = 5)),
  legend.position = "none",
  plot.margin = margin(t = 2, r = 5, b = 2, l = 5)
)

p_trk1 <- build_track(coverage_data, 'Control_rbcDNA', '#E64B35FF', '#E64B35FF', 'Control samples, rbcDNA (n = 10)')
p_trk2 <- build_track(coverage_data, 'Control_gDNA', '#3E60AA', '#3E60AA', 'Control samples, leukocyte DNA (n = 10)', show_y_title = TRUE)
p_trk3 <- build_track(coverage_data, 'trn_HD', '#BC3C29CC', '#BC3C29CC', 'Non-AN (Discovery cohort, n = 449)')
p_anno1 <- ggplot(df_HD_1000k_anno) +
  geom_tile(aes(x = start, y = broadPeak_y, fill = broadPeak), colour = NA) +
  scale_fill_gradient2(low = "white", mid = "white", high = "#F39B7FFF", midpoint = 0) +
  facet_grid(. ~ arm, space = "free_x", scales = "free_x") +
  labs(y = "rbcDNA-\nenriched\nregions", x = NULL) +
  heatmap_theme + theme(strip.text.x = element_blank())
p_anno2 <- ggplot(df_HD_1000k_anno) +
  geom_tile(aes(x = start, y = broadPeak_y, fill = broadPeak), colour = NA) +
  scale_fill_gradient2(low = "#3C5488FF", mid = "white", high = "white", midpoint = 0) +
  facet_grid(. ~ arm, space = "free_x", scales = "free_x", switch = "x") +
  labs(y = "rbcDNA-\ndepleted\nregions", x = "Chromosome") +
  heatmap_theme

panel_combined <- p_trk1 / p_trk2 / p_trk3 / p_anno1 / p_anno2 +
  plot_layout(heights = c(2, 2, 2, 0.4, 0.5))

# ggsave(file.path(out_dir, 'figureS2_d.pdf'), panel_combined, width = 8, height = 7, device = cairo_pdf)

theme_common <- theme(
  text = element_text(size = 7, family = "Arial"),
  axis.title = element_text(size = 7),
  axis.text = element_text(size = 7),
  legend.text = element_text(size = 7),
  strip.text = element_text(size = 7),
  plot.title = element_text(size = 7)
)

p_1 <- p_age + labs(tag = "A") + theme(plot.tag = element_text(size = 10))
p_2 <- p_rbc + labs(tag = "B") + theme(plot.tag = element_text(size = 10))
p_3 <- p_wbc
p_4 <- p_plt
p_5 <- p_mt + labs(tag = "C") + theme(plot.tag = element_text(size = 10))
p_6 <- panel_combined + labs(tag = "D") + theme(plot.tag = element_text(size = 10))

row2 <- wrap_elements(full = p_5) +
        wrap_elements(full = p_6) +
        plot_layout(widths = c(1, 3))

p_final <- (p_1 + p_2 + p_3 + p_4 + plot_layout(nrow = 1)) /
           row2 +
           plot_layout(heights = c(1, 1.5))

ggsave(file.path(out_dir, 'figureS2_combined.pdf'), p_final, width = 8, height = 7.2, device = cairo_pdf)
args <- commandArgs(trailingOnly = TRUE)
working_dir <- args[1]
setwd(working_dir)

args_all <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args_all[grep("--file=", args_all)])
script_dir <- dirname(normalizePath(script_path))

out_dir <- file.path(working_dir, "Figures")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

suppressPackageStartupMessages({
  library(stringr)
  library(tibble)
  library(dplyr)
  library(ggplot2)
  library(cowplot)
})

source(file.path(script_dir, 'plot_function.r'), chdir = TRUE)
source(file.path(script_dir, 'plotAUC_theme.r'), chdir = TRUE)

cfg <- tribble(
  ~file, ~tag, ~ycol,
  "Human_model/search_trncv_v2_zheer_zr11_1234_zheer_zr11_1234.csv", "11in11", "TRNCV-AUC-AA",
  "Human_model/search_trncv_v2_zheer_zr11_1234_zheer_zr8_1234.csv",  "11in8",  "TRNCV-AUC-AA",
  "Human_model/search_trncv_v2_zheer_zr1_1234_zheer_zr1_1234.csv",   "1in1",   "TRNCV-AUC",
  "Human_model/search_trncv_v2_zheer_zr1_1234_zheer_zr6_1234.csv",   "1in6",   "TRNCV-AUC",
  "Human_model/search_trncv_v2_zheer_zr2_1234_zheer_zr2_1234.csv",   "2in2",   "TRNCV-AUC",
  "Human_model/search_trncv_v2_zheer_zr2_1234_zheer_zr6_1234.csv",   "2in6",   "TRNCV-AUC"
)

make_agg <- function(file, tag, ycol) {
  if(!file.exists(file)) return(NULL)
  max_pcas <- ifelse(grepl("^11in", tag), 20, 15)
  readr::read_csv(file, show_col_types = FALSE) %>%
    mutate(n_pcas = as.numeric(n_pcas)) %>%
    filter(n_pcas >= 8 & n_pcas <= max_pcas) %>%
    group_by(n_pcas) %>%
    summarise(min_auc = min(.data[[ycol]], na.rm = TRUE), mean_auc = mean(.data[[ycol]], na.rm = TRUE), max_auc = max(.data[[ycol]], na.rm = TRUE), .groups = "drop") %>%
    mutate(tag = tag, ycol = ycol)
}

all_agg <- purrr::pmap_dfr(cfg, make_agg)

# Figure S5a: Candidate models cross-validated AUCs
plot_pair <- function(tags, base_color, outfile, title, base_size = 7, w = 4.5, h = 3.5) {
  if(nrow(all_agg) == 0) return(NULL)
  dd <- all_agg %>% filter(tag %in% tags)
  if(nrow(dd) == 0) return(NULL)

  col_map <- c(setNames(scales::alpha(base_color, 1.00), tags[1]), setNames(scales::alpha(base_color, 0.55), tags[2]))
  grey_fill_map <- c(setNames(scales::alpha("grey40", 0.28), tags[1]), setNames(scales::alpha("grey40", 0.14), tags[2]))

  p <- ggplot(dd, aes(x = n_pcas)) +
    geom_ribbon(aes(ymin = min_auc, ymax = max_auc, fill = tag), color = NA, show.legend = FALSE) +
    geom_line(aes(y = mean_auc, color = tag), linetype = "dashed", linewidth = 0.8) +
    scale_color_manual(values = col_map, labels = c("Model group 1", "Model group 2")) +
    scale_fill_manual(values = grey_fill_map, labels = c("Model group 1", "Model group 2")) +
    labs(title = title, x = "Number of PCAs", y = "Cross-validated AUC\n(Discovery cohort)") +
    theme_classic(base_size = base_size) +
    theme(
      legend.position = c(0.75, 0.2),
      legend.title = element_blank(),
      legend.background = element_blank(),
      legend.key = element_blank(),
      legend.text = element_text(size = 6),
      plot.title = element_text(size = 7, hjust = 0.5),
      axis.title = element_text(size = 7),
      axis.text = element_text(size = 6),
      panel.grid.major.y = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank()
    )

  # ggsave(outfile, p, width = w, height = h, dpi = 300, device = cairo_pdf)
  return(p)
}

pa1 <- NULL
pa2 <- NULL
pa3 <- NULL

if(nrow(all_agg) > 0) {
  pa1 <- plot_pair(c("2in2", "2in6"), "#1f77b4", file.path(out_dir, "figureS5_a_1.pdf"), "Candidate models: Non-AN vs. advanced neoplasia")
  pa2 <- plot_pair(c("1in1", "1in6"), "#b2182b", file.path(out_dir, "figureS5_a_2.pdf"), "Candidate models: Non-AN vs. CRC")
  pa3 <- plot_pair(c("11in11", "11in8"), "#ff7f0e", file.path(out_dir, "figureS5_a_3.pdf"), "Candidate models: Non-AN vs. AA")
}

p1 <- NULL
p2 <- NULL
p1_crc <- NULL
p1_aa <- NULL
p_s5c <- NULL
row_b_full <- NULL

if(file.exists('./Human_model/selected_model_zheer_with_internal_test.csv')) {
  df <- read.table('./Human_model/selected_model_zheer_with_internal_test.csv', sep = '\t', header = TRUE)

  # Figure S5b: Boxplots of candidate model AUCs in test cohort
  plot_auc_boxplot <- function(data, subset_kind, y_col, fill_color, y_label) {
    plot_data <- data %>%
      filter(select_kind == subset_kind | grepl('rule', group)) %>%
      mutate(select_kind = subset_kind) %>%
      mutate(
        x_label = case_when(
          grepl("rule1", group) ~ "Combined1",
          grepl("rule2", group) ~ "Combined2",
          grepl("rule3", group) ~ "Combined3",
          grepl("in11$|in1$|2in2", group) ~ "Model group1",
          grepl("in8$|in6", group) ~ "Model group2",
          TRUE ~ group
        )
      ) %>%
      filter(x_label %in% c("Model group1", "Model group2", "Combined1", "Combined2", "Combined3")) %>%
      mutate(x_label = factor(x_label, levels = c("Model group1", "Model group2", "Combined1", "Combined2", "Combined3")))

    ggplot(plot_data, aes(x = x_label, y = .data[[y_col]], fill = select_kind)) +
      geom_boxplot(width = 0.6, outlier.shape = NA, alpha = 0.7, linewidth = 0.4) +
      geom_jitter(width = 0.2, size = 0.6, alpha = 0.5, shape = 16) +
      scale_fill_manual(values = fill_color) +
      coord_cartesian(ylim = c(NA, 0.98)) +
      labs(y = y_label, x = NULL) +
      theme_classic(base_size = 7) +
      theme(
        legend.position = "none",
        axis.title = element_text(size = 7),
        axis.text.y = element_text(size = 6),
        axis.text.x = element_text(angle = 45, hjust = 1, size = 6)
      )
  }

  aacrc_models <- df %>% filter(grepl('rule1|2in2|2in6', group))
  p1 <- plot_auc_boxplot(aacrc_models, "crc", "P20.AUC.CRC_HD_XR", "#1f77b4", "AUC of Non-AN vs CRC\n(test cohort)")
  p2 <- plot_auc_boxplot(aacrc_models, "aa", "P20.AUC.AA_HD_XR", "#1f77b4", "AUC of Non-AN vs AN\n(test cohort)")

  crc_models <- df %>% filter(grepl('rule3|^1in6$|^1in1$', group))
  p1_crc <- plot_auc_boxplot(crc_models, "crc", "P20.AUC.CRC_HD_XR", "#b2182b", "AUC of Non-AN vs CRC\n(test cohort)")

  aa_models <- df %>% filter(grepl('rule2|^11in8$|^11in11$', group))
  p1_aa <- plot_auc_boxplot(aa_models, "aa", "P20.AUC.CRC_HD_XR", "#ff7f0e", "AUC of Non-AN vs AN\n(test cohort)")

  row_b_titles <- cowplot::plot_grid(
    ggdraw() + draw_label("Advanced neoplasia-\nassociated candidate models", size = 7, lineheight = 1.1),
    ggdraw() + draw_label("CRC-associated\ncandidate models", size = 7, lineheight = 1.1),
    ggdraw() + draw_label("AA-associated\ncandidate models", size = 7, lineheight = 1.1),
    ncol = 3, rel_widths = c(2, 1, 1)
  )
  row_b_plots <- cowplot::plot_grid(p1, p2, p1_crc, p1_aa, ncol = 4)
  row_b_full <- cowplot::plot_grid(row_b_titles, row_b_plots, ncol = 1, rel_heights = c(0.15, 1))

  # ggsave(file.path(out_dir, 'figureS5_b.pdf'), row_b_full, width = 7.5, height = 3.5, device = cairo_pdf)

  # Figure S5c: Bar chart comparing combined models vs fusion model
  rule_mapping <- c("94" = "Combined\nmodel 1", "108" = "Combined\nmodel 2", "121" = "Combined\nmodel 3", "161" = "Final fusion\nmodel")
  col_crc <- grep("P20[.-]AUC[.-]CRC_HD_XR", colnames(df), value = TRUE)[1]
  col_all <- grep("P20[.-]AUC[.-]ALL_HD_XR", colnames(df), value = TRUE)[1]

  if(!is.na(col_crc) && !is.na(col_all)) {
    df_s5c <- df %>%
      filter(index_no %in% c(94, 108, 121, 161)) %>%
      mutate(Rule = recode(as.character(index_no), !!!rule_mapping)) %>%
      mutate(Rule = factor(Rule, levels = c("Combined\nmodel 1", "Combined\nmodel 2", "Combined\nmodel 3", "Final fusion\nmodel"))) %>%
      select(Rule, AUC_CRC = all_of(col_crc), AUC_ALL = all_of(col_all)) %>%
      tidyr::pivot_longer(cols = starts_with("AUC"), names_to = "Comparison", values_to = "AUC") %>%
      mutate(Comparison = case_when(
        Comparison == "AUC_ALL" ~ "AUC of Non-AN vs AN",
        Comparison == "AUC_CRC" ~ "AUC of Non-AN vs CRC"
      )) %>%
      mutate(Comparison = factor(Comparison, levels = c("AUC of Non-AN vs AN", "AUC of Non-AN vs CRC")))

    p_s5c <- ggplot(df_s5c, aes(x = Rule, y = AUC, fill = Comparison)) +
      geom_bar(stat = "identity", position = position_dodge(width = 0.7), width = 0.6) +
      geom_text(aes(label = sprintf("%.3f", AUC)), position = position_dodge(width = 0.7), vjust = -0.5, size = 2.2) +
      scale_fill_manual(values = c("AUC of Non-AN vs AN" = "#c2c7c9", "AUC of Non-AN vs CRC" = "#879395")) +
      coord_cartesian(ylim = c(0.75, 1.02)) +
      labs(x = NULL, y = "AUC (test cohort)", fill = NULL) +
      theme_classic(base_size = 7) +
      theme(
        legend.position = "bottom",
        legend.key.size = unit(0.3, "cm"),
        legend.text = element_text(size = 5),
        legend.margin = margin(t = -5),
        panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_blank(),
        axis.text.x = element_text(size = 6, lineheight = 0.9, margin = margin(t = 2)),
        axis.title = element_text(size = 7)
      )

    # ggsave(file.path(out_dir, paste0('figureS5_c.pdf')), p_s5c, width = 5.5, height = 4.5, device = cairo_pdf)
  }
}

# Figure S5: Combined plot
if(!is.null(pa1) && !is.null(row_b_full) && !is.null(p_s5c)) {
  row_a <- cowplot::plot_grid(pa1, pa2, pa3, ncol = 3, labels = c("A", "", ""), label_size = 10, label_fontface = "plain")
  row_bc <- cowplot::plot_grid(row_b_full, p_s5c, ncol = 2, rel_widths = c(1.8, 1), labels = c("B", "C"), label_size = 10, label_fontface = "plain")
  final_fig <- cowplot::plot_grid(row_a, row_bc, nrow = 2, rel_heights = c(0.8, 1))
  ggsave(file.path(out_dir, paste0('FigureS5_Combined.pdf')), final_fig, width = 8, height = 4.6, dpi = 300, device = cairo_pdf)
}
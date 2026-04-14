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
  library(gridExtra)
  library(patchwork)
  library(grid)
  library(gtable)
})

source(file.path(script_dir, 'plot_function.r'), chdir = TRUE)
source(file.path(script_dir, 'plotAUC_theme.r'), chdir = TRUE)

load('./Figures/prediction.RData')
load('./Figures/sampleinfo.RData')

clean_merge <- function(pred_df, info_df, ds_name) {
  merge(pred_df, info_df, by = 'Sample') %>%
    rename_all(~gsub('-| ', '', .)) %>%
    mutate(dataset = ds_name) %>%
    rename_with(~"Age", starts_with("Age", ignore.case = TRUE)) %>%
    rename_with(~"RBC", starts_with("RBC", ignore.case = FALSE)) %>%
    rename_with(~"WBC", starts_with("WBC", ignore.case = FALSE)) %>%
    rename_with(~"PLT", starts_with("PLT", ignore.case = FALSE)) %>%
    rename_with(~"HGB", starts_with("HGB", ignore.case = FALSE))
}

trn_info <- clean_merge(trn, trn_info, 'Training')
test_info <- clean_merge(test, test_info, 'Test')
sd_info <- clean_merge(sd, sd_info, 'IND2_SD')
wz_info <- clean_merge(wz, wz_info, 'IND1_WZ')

group_cols <- c("Control" = ggsci::pal_material("blue-grey")(10)[5], "AA" = '#DF8F44FF', "CRC" = "#9F1A1AFF")

trn_test_clean <- bind_rows(trn_info, test_info) %>%
  filter(dataset == 'Test') %>%
  mutate(
    Group_clean = case_when(
      Group == 'Control' | is.na(Group) ~ 'Non-AN',
      Group == 'AA' ~ 'AA',
      Group == 'CRC' ~ 'CRC',
      TRUE ~ 'Non-AN'
    ),
    Group_clean = factor(Group_clean, levels = c('Non-AN', 'AA', 'CRC'))
  )

col_s8 <- c("Non-AN" = "#78909C", "AA" = "#DF8F44FF", "CRC" = "#9F1A1AFF")

plot_scatter_s8 <- function(df, x_var, x_lab_expr, show_leg = FALSE, show_border = FALSE) {
  p <- ggplot(df, aes(x = .data[[x_var]], y = merged_score, color = Group_clean, fill = Group_clean)) +
    geom_point(size = 0.8, alpha = 0.4, shape = 16) +
    geom_smooth(method = lm, linetype = "longdash", se = FALSE, fullrange = TRUE, linewidth = 0.8) +
    ggpubr::stat_cor(aes(color = Group_clean), method = 'spearman',
                     label.x.npc = 0.05, hjust = 0,
                     label.y = c(1.0, 0.90, 0.80),
                     size = 2, show.legend = FALSE, family = "ArialMT") +
    scale_color_manual(values = col_s8) +
    scale_fill_manual(values = col_s8) +
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25), expand = expansion(mult = c(0.05, 0.15))) +
    labs(x = x_lab_expr, y = "rbcDNA predictive score") +
    theme_classic(base_family = "ArialMT") +
    theme(
      axis.text = element_text(color = "black", size = 5),
      axis.title = element_text(color = "black", size = 5),
      axis.ticks = element_line(color = "black", linewidth = 0.2),
      plot.margin = margin(t = 5, r = 5, b = 5, l = 5)
    )

  if (show_border) {
    p <- p + theme(
      panel.border = element_rect(color = "black", fill = NA, linewidth = 0.5),
      axis.line = element_blank()
    )
  } else {
    p <- p + theme(
      panel.border = element_blank(),
      axis.line = element_line(color = "black", linewidth = 0.2)
    )
  }

  if(show_leg){
    p <- p + theme(
      legend.box.margin = margin(b = -10),
      legend.position = "top",
      legend.title = element_blank(),
      legend.key = element_blank(),
      legend.background = element_blank(),
      legend.text = element_text(size = 5),
      legend.key.width = unit(0.5, "cm")
    ) + guides(
      color = guide_legend(override.aes = list(fill = NA, linetype = "longdash", linewidth = 0.8, shape = 16, size = 2, alpha = 1)),
      fill = "none"
    )
  } else {
    p <- p + theme(legend.position = "none")
  }
  return(p)
}

plot_violin_s8 <- function(df, x_var, x_lab) {
  if (x_var != "Sex") {
    df <- df %>% mutate(!!sym(x_var) := factor(.data[[x_var]], levels = c('Never', 'Current', 'Former', 'No record')))
    test_method <- "kruskal.test"
    fill_cols <- ggsci::pal_material("blue-grey", alpha = 0.5)(10)[c(2,4,6,8)]
  } else {
    test_method <- "wilcox.test"
    fill_cols <- ggsci::pal_material("blue-grey", alpha = 0.5)(10)[c(7, 9)]
  }

  n_df <- df %>%
    group_by(Group_clean, !!sym(x_var)) %>%
    summarise(
      n = n(),
      y_pos = max(merged_score, na.rm = TRUE) + 0.04,
      .groups = 'drop'
    ) %>%
    filter(!is.na(!!sym(x_var)))

  ggplot(df, aes(x = .data[[x_var]], y = merged_score, fill = .data[[x_var]])) +
    geom_violin(alpha = 0.6, width = 1, trim = TRUE, drop = FALSE) +
    geom_boxplot(width = 0.2, outlier.shape = NA, alpha = 0.8, fill = "white") +
    geom_jitter(width = 0.1, size = 0.5, alpha = 0.4, shape = 16) +
    geom_text(data = n_df, aes(x = .data[[x_var]], y = y_pos, label = paste0("n=", n)),
              size = 1.8, vjust = 0, inherit.aes = FALSE, family = "ArialMT") +
    scale_fill_manual(values = fill_cols) +
    facet_grid(. ~ Group_clean) +
    coord_cartesian(clip = "off") +
    scale_y_continuous(limits = c(0, 1.15), breaks = seq(0, 1, 0.25), expand = expansion(mult = c(0.05, 0.05))) +
    stat_compare_means(label = "p.signif", method = test_method, label.y = 1.10, label.x.npc = "center", size = 2, family = "ArialMT", hide.ns = FALSE) +
    labs(y = 'rbcDNA predictive scores', x = x_lab) +
    theme_classic(base_family = "ArialMT") +
    theme(
      legend.position = 'none',
      strip.background = element_blank(),
      strip.text = element_text(size = 5, color = "black"),
      axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1, color = "black", size = 5),
      axis.text.y = element_text(color = "black", size = 5),
      axis.title = element_text(color = "black", size = 5),
      panel.border = element_blank(),
      axis.line = element_line(color = "black", linewidth = 0.2),
      axis.ticks = element_line(color = "black", linewidth = 0.2),
      plot.margin = margin(t = 5, r = 5, b = 5, l = 5)
    )
}

p_age <- plot_scatter_s8(trn_test_clean, "Age", "Age", show_leg = TRUE, show_border = TRUE) +
  theme(axis.title.x = element_text(margin = margin(t = 24)))

v_sex <- plot_violin_s8(trn_test_clean, "Sex", "Sex")
v_smoke <- plot_violin_s8(trn_test_clean, "Smokingstatus", "History of smoking")
v_alcohol <- plot_violin_s8(trn_test_clean, "Alcoholconsumptionstatus", "History of alcohol consumption")

row1 <- plot_grid(p_age, v_sex, v_smoke, v_alcohol, ncol = 4, rel_widths = c(1.0, 0.8, 1.2, 1.2), align = "h", axis = "tb")

p_rbc <- plot_scatter_s8(trn_test_clean, "RBC", expression(paste("RBC (\u00D7", 10^{12}, "/L)")), show_border = TRUE)
p_hgb <- plot_scatter_s8(trn_test_clean, "HGB", "HGB (g/L)", show_border = TRUE)
p_wbc <- plot_scatter_s8(trn_test_clean, "WBC", expression(paste("WBC (\u00D7", 10^9, "/L)")), show_border = TRUE)
p_plt <- plot_scatter_s8(trn_test_clean, "PLT", expression(paste("PLT (\u00D7", 10^9, "/L)")), show_border = TRUE)

pC_plots <- plot_grid(p_rbc, p_hgb, p_wbc, p_plt, ncol = 4, align = "h", axis = "tb")

dummy_leg_C <- ggplot(trn_test_clean, aes(x = Age, y = merged_score, color = Group_clean)) +
  geom_point(shape = 16, size = 2) +
  geom_line(stat = "smooth", method = "lm", linewidth = 0.8, linetype = "longdash") +
  scale_color_manual(values = col_s8) +
  theme_void(base_family = "ArialMT") +
  theme(
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.key = element_blank(),
    legend.background = element_blank(),
    legend.text = element_text(size = 5),
    legend.key.width = unit(0.5, "cm")
  ) + guides(color = guide_legend(override.aes = list(fill = NA)))

leg_C <- get_legend(dummy_leg_C)
title_C <- ggdraw() + draw_label(paste0("Internal test cohort (N = ", nrow(trn_test_clean), ")"), fontfamily = "ArialMT", fontface = "plain", size = 5)

row2 <- plot_grid(title_C, pC_plots, leg_C, ncol = 1, rel_heights = c(0.05, 1, 0.1))

g_abc <- plot_grid(row1, row2, ncol = 1, rel_heights = c(1, 0.9))

ggsave(file.path(out_dir, 'figureS8_abc.pdf'), g_abc, width = 12.8, height = 7.5, device = cairo_pdf)

INDs_clean <- bind_rows(sd_info, wz_info) %>%
  mutate(
    Group_clean = case_when(
      Group == 'Control' | is.na(Group) ~ 'Non-AN',
      Group == 'AA' ~ 'AA',
      Group == 'CRC' ~ 'CRC',
      TRUE ~ 'Non-AN'
    ),
    Group_clean = factor(Group_clean, levels = c('Non-AN', 'AA', 'CRC')),
    dataset_clean = case_when(
      dataset == 'IND1_WZ' ~ "WENZHOU\nexternal test set 1",
      dataset == 'IND2_SD' ~ "SHANDONG\nexternal test set 2"
    ),
    dataset_clean = factor(dataset_clean, levels = c("WENZHOU\nexternal test set 1", "SHANDONG\nexternal test set 2"))
  )

plot_violin_ind <- function(df, x_var, x_lab, show_y = TRUE) {
  if (x_var != "Sex") {
    df <- df %>% mutate(!!sym(x_var) := factor(.data[[x_var]], levels = c('Never', 'Current', 'Former', 'No record')))
    test_method <- "kruskal.test"
    fill_cols <- ggsci::pal_material("blue-grey", alpha = 0.5)(10)[c(2,4,6,8)]
  } else {
    test_method <- "wilcox.test"
    fill_cols <- ggsci::pal_material("blue-grey", alpha = 0.5)(10)[c(7, 9)]
  }

  n_df <- df %>%
    group_by(dataset_clean, Group_clean, !!sym(x_var)) %>%
    summarise(
      n = n(),
      y_pos = max(merged_score, na.rm = TRUE) + 0.08,
      .groups = 'drop'
    ) %>%
    filter(!is.na(!!sym(x_var)))

  p <- ggplot(df, aes(x = .data[[x_var]], y = merged_score, fill = .data[[x_var]])) +
    geom_violin(alpha = 0.6, width = 1, trim = TRUE, drop = FALSE) +
    geom_boxplot(width = 0.2, outlier.shape = NA, alpha = 0.8, fill = "white") +
    geom_jitter(width = 0.1, size = 0.5, alpha = 0.4, shape = 16) +
    geom_text(data = n_df, aes(x = .data[[x_var]], y = y_pos, label = paste0("n=", n)),
              size = 1.8, vjust = 0, inherit.aes = FALSE, family = "ArialMT") +
    scale_fill_manual(values = fill_cols) +
    facet_grid(dataset_clean ~ Group_clean, switch = "y") +
    coord_cartesian(clip = "off") +
    scale_y_continuous(limits = c(0, 1.15), breaks = seq(0, 1, 0.25), expand = expansion(mult = c(0.05, 0.05))) +
    stat_compare_means(label = "p.signif", method = test_method, label.y = 1.10, label.x.npc = "center", size = 2, family = "ArialMT", hide.ns = FALSE) +
    theme_classic(base_family = "ArialMT") +
    theme(
      legend.position = 'none',
      strip.background = element_blank(),
      strip.text.x = element_text(size = 5, color = "black"),
      strip.placement = "outside",
      axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1, color = "black", size = 5),
      axis.text.y = element_text(color = "black", size = 5),
      axis.title = element_text(color = "black", size = 5),
      panel.border = element_blank(),
      axis.line = element_line(color = "black", linewidth = 0.2),
      axis.ticks = element_line(color = "black", linewidth = 0.2),
      panel.spacing.y = unit(0.4, "lines"),
      plot.margin = margin(t = 5, r = 5, b = 5, l = 5)
    )

  if(show_y) {
    p <- p + labs(y = 'rbcDNA predictive scores', x = x_lab) +
      theme(strip.text.y.left = element_text(size = 5, color = "black"))
  } else {
    p <- p + labs(y = NULL, x = x_lab) +
      theme(strip.text.y.left = element_blank())
  }

  return(p)
}

p_sex_ind <- plot_violin_ind(INDs_clean, "Sex", "Sex", show_y = TRUE)
p_smoke_ind <- plot_violin_ind(INDs_clean, "Smokingstatus", "History of smoking", show_y = FALSE)
p_alcohol_ind <- plot_violin_ind(INDs_clean, "Alcoholconsumptionstatus", "History of alcohol consumption", show_y = FALSE)

row_D <- plot_grid(p_sex_ind, p_smoke_ind, p_alcohol_ind, ncol = 3, rel_widths = c(1, 1, 1), align = "h", axis = "tb")

vars <- c("Age", "RBC", "HGB", "WBC", "PLT")
var_labels <- c("Age (years)", "RBC (\u00D710\u00B9\u00B2/L)", "HGB (g/L)", "WBC (\u00D710\u2079/L)", "PLT (\u00D710\u2079/L)")

val_matrix <- matrix(nrow = 5, ncol = 6)
ds_list <- c('IND1_WZ', 'IND2_SD')
grp_list <- c('Non-AN', 'AA', 'CRC')

col_idx <- 1
for(ds in ds_list) {
  for(grp in grp_list) {
    sub_df <- INDs_clean %>% filter(dataset == ds, Group_clean == grp)
    for(i in 1:5) {
      if(sum(!is.na(sub_df[[vars[i]]])) > 2) {
        val_matrix[i, col_idx] <- sprintf("%.4f", cor(sub_df[[vars[i]]], sub_df$merged_score, method = "spearman", use = "complete.obs"))
      } else {
        val_matrix[i, col_idx] <- "NA"
      }
    }
    col_idx <- col_idx + 1
  }
}

tbl_matrix <- data.frame(
  C1 = c("Parameters", var_labels),
  C2 = c("Non-AN", val_matrix[,1]),
  C3 = c("AA",     val_matrix[,2]),
  C4 = c("CRC",    val_matrix[,3]),
  C5 = c("Non-AN", val_matrix[,4]),
  C6 = c("AA",     val_matrix[,5]),
  C7 = c("CRC",    val_matrix[,6])
)
colnames(tbl_matrix) <- NULL

tt <- ttheme_minimal(
  core = list(
    bg_params = list(fill = NA, col = NA),
    fg_params = list(fontfamily = "ArialMT", fontsize = 7, hjust = 0.5, x = 0.5),
    padding = unit(c(0.1, 4), "mm")
  )
)
tbl_grob <- tableGrob(tbl_matrix, rows = NULL, cols = NULL, theme = tt)
tbl_grob$widths <- unit(c(0.16, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14), "npc")
tbl_grob$heights <- unit(rep(4, nrow(tbl_grob)), "mm")

tbl_grob <- gtable_add_grob(tbl_grob, grobs = segmentsGrob(y0=unit(1,"npc"), y1=unit(1,"npc"), gp = gpar(lwd = 1)), t = 1, l = 1, r = 7)
tbl_grob <- gtable_add_grob(tbl_grob, grobs = segmentsGrob(y0=unit(0,"npc"), y1=unit(0,"npc"), gp = gpar(lwd = 1)), t = 1, l = 1, r = 7)
tbl_grob <- gtable_add_grob(tbl_grob, grobs = segmentsGrob(y0=unit(0,"npc"), y1=unit(0,"npc"), gp = gpar(lwd = 1.5)), t = nrow(tbl_grob), l = 1, r = 7)

super_matrix <- data.frame(
  C1 = "",
  C2 = "",
  C3 = "WENZHOU\nexternal test set 1",
  C4 = "",
  C5 = "",
  C6 = "SHANDONG\nexternal test set 2",
  C7 = ""
)
colnames(super_matrix) <- NULL

super_tt <- ttheme_minimal(
  core = list(
    bg_params = list(fill = NA, col = NA),
    fg_params = list(fontfamily = "ArialMT", fontsize = 7, hjust = 0.5, x = 0.5),
    padding = unit(c(0, 4), "mm")
  )
)

super_grob <- tableGrob(super_matrix, rows = NULL, cols = NULL, theme = super_tt)
super_grob$widths <- tbl_grob$widths
super_grob <- gtable_add_grob(super_grob, grobs = segmentsGrob(y0=unit(1,"npc"), y1=unit(1,"npc"), gp = gpar(lwd = 1.5)), t = 1, l = 1, r = 7)
combined_table <- gtable_rbind(super_grob, tbl_grob)

table_title <- ggdraw() + draw_label("Spearman correlation between rbcDNA predictive scores and parameters", fontfamily = "ArialMT", size = 7, fontface = "plain")

table_content <- plot_grid(table_title, combined_table, ncol = 1, rel_heights = c(0.1, 1))

table_final <- plot_grid(NULL, table_content, NULL, nrow = 1, rel_widths = c(0.04, 0.92, 0.04))

g_de <- plot_grid(row_D, table_final, ncol = 1, rel_heights = c(1.6, 1))

g_final <- plot_grid(g_abc, g_de, ncol = 1, rel_heights = c(1, 0.9))

ggsave(file.path(out_dir, 'figureS8_all.pdf'), g_final, width = 8, height = 9.5, device = cairo_pdf)
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
  library(tableone)
  library(dplyr)
  library(stringr)
  library(gridExtra)
  library(grid)
  library(gtable)
})

load('./Figures/sampleinfo.RData')

trn_info$Cohort <- "1_Discovery"
test_info$Cohort <- "2_Internal_Test"
wz_info$Cohort <- "3_Wenzhou"
sd_info$Cohort <- "4_Shandong"

common_cols <- intersect(colnames(trn_info), colnames(wz_info))
all_cohorts <- bind_rows(
  trn_info[, common_cols],
  test_info[, common_cols],
  wz_info[, common_cols],
  sd_info[, common_cols]
)

colnames(all_cohorts)[grep('RBC', colnames(all_cohorts))] <- 'RBC'
colnames(all_cohorts)[grep('WBC', colnames(all_cohorts))] <- 'WBC'
colnames(all_cohorts)[grep('PLT', colnames(all_cohorts))] <- 'PLT'
colnames(all_cohorts)[grep('HGB', colnames(all_cohorts))] <- 'HGB'

all_cohorts <- all_cohorts %>%
  rename(
    any_of(c(
      Age = "Age (at collection)",
      Info_group = "Sub-group",
      smoking_state = "Smoking status",
      Alcohol_state = "Alcohol consumption status"
    ))
  )

vars <- c("Age", "Sex", "Group", "Info_group", "smoking_state", "Alcohol_state", "RBC", "HGB", "WBC", "PLT")
cat_vars <- c("Sex", "Group", "Info_group", "smoking_state", "Alcohol_state")

tab1 <- CreateTableOne(vars = vars, strata = "Cohort", data = all_cohorts, factorVars = cat_vars)
tab1_mat <- print(tab1, quote = FALSE, noSpaces = TRUE, printToggle = FALSE, showAllLevels = TRUE)

out_file <- file.path(out_dir, 'figureS3_table.xlsx')
write.xlsx(as.data.frame(tab1_mat), out_file, rowNames = TRUE)

df_raw <- as.data.frame(tab1_mat, stringsAsFactors = FALSE)
df_raw$VarName <- trimws(rownames(tab1_mat))
df_raw$ParentVar <- ""
curr_var <- ""
for (i in 1:nrow(df_raw)) {
  if (df_raw$VarName[i] != "") curr_var <- df_raw$VarName[i]
  df_raw$ParentVar[i] <- curr_var
}

get_val <- function(var, level_val = NULL, cohort) {
  var_idx <- grep(paste0("^", var), df_raw$ParentVar)
  if (length(var_idx) == 0) return("")

  sub_df <- df_raw[var_idx, ]

  if (is.null(level_val)) {
    val <- sub_df[1, cohort]
  } else {
    lvl_idx <- grep(level_val, sub_df$level, fixed = TRUE)
    if (length(lvl_idx) == 0) return("")
    val <- sub_df[lvl_idx[1], cohort]
  }
  if (is.na(val)) "" else val
}

get_p <- function(var) {
  var_idx <- grep(paste0("^", var), df_raw$ParentVar)
  if (length(var_idx) == 0) return("")
  pval <- df_raw[var_idx[1], "p"]
  if (is.na(pval)) "" else pval
}

n_dis <- get_val("n", cohort = "1_Discovery")
n_int <- get_val("n", cohort = "2_Internal_Test")
n_wz  <- get_val("n", cohort = "3_Wenzhou")
n_sd  <- get_val("n", cohort = "4_Shandong")

n_cols <- 6
table_rows <- list()

add_row <- function(content,
                    is_header = FALSE,
                    is_subheader = FALSE,
                    indent = 0,
                    hjust = c(0, rep(0.5, 5)),
                    x = c(0.02, rep(0.5, 5)),
                    fontface = "plain",
                    fontsize = 7,
                    row_height = unit(5, "mm"),
                    bg_fill = "white") {

  if (indent > 0) {
    content[1] <- paste0(strrep(" ", indent * 2), content[1])
  }

  grobs <- lapply(1:n_cols, function(j) {
    textGrob(content[j],
             x = x[j],
             hjust = hjust[j],
             gp = gpar(fontsize = fontsize, fontface = fontface))
  })

  row_grob <- gtable_row(row_height, grobs)

  if (bg_fill != "white") {
    for (j in 1:n_cols) {
      row_grob <- gtable_add_grob(row_grob,
                                  rectGrob(gp = gpar(fill = bg_fill, col = NA)),
                                  t = 1, l = j)
    }
  }

  table_rows <<- c(table_rows, list(row_grob))
}

add_hline <- function(lwd = 1, linetype = "solid") {
  line_grob <- segmentsGrob(x0 = 0, y0 = 0.5, x1 = 1, y1 = 0.5,
                            gp = gpar(lwd = lwd, lty = linetype))
  gt <- gtable(widths = unit(rep(1/n_cols, n_cols), "npc"), heights = unit(2, "mm"))
  gt <- gtable_add_grob(gt, line_grob, t = 1, l = 1, r = n_cols)
  table_rows <<- c(table_rows, list(gt))
}

gtable_row <- function(height, grobs) {
  gt <- gtable(widths = unit(rep(1/n_cols, n_cols), "npc"),
               heights = height)
  for (j in seq_along(grobs)) {
    gt <- gtable_add_grob(gt, grobs[[j]], t = 1, l = j)
  }
  gt
}

# Figure S3: Cohort Demographics Table

add_row(c("", "Discovery\ncohort", "Internal test\ncohort", "P value", "Wenzhou\ncohort", "Shandong\ncohort"),
        is_header = TRUE, fontface = "bold", row_height = unit(7, "mm"))

add_hline(lwd = 1.5)

add_row(c("Participants (n)", n_dis, n_int, "", n_wz, n_sd))

age_dis <- get_val("Age", cohort = "1_Discovery")
age_int <- get_val("Age", cohort = "2_Internal_Test")
age_wz  <- get_val("Age", cohort = "3_Wenzhou")
age_sd  <- get_val("Age", cohort = "4_Shandong")
age_p   <- get_p("Age")
add_row(c("Age (mean (SD))", age_dis, age_int, age_p, age_wz, age_sd))

add_row(c("Sex (%)", "", "", get_p("Sex"), "", ""), is_subheader = TRUE)
sex_levels <- c("Female", "Male")
for (lvl in sex_levels) {
  add_row(c(paste0("    ", lvl),
            get_val("Sex", lvl, "1_Discovery"),
            get_val("Sex", lvl, "2_Internal_Test"),
            "",
            get_val("Sex", lvl, "3_Wenzhou"),
            get_val("Sex", lvl, "4_Shandong")))
}

add_hline(lwd = 0.5)

add_row(c("Non-advanced neoplasia group (%)", "", "", "", "", ""), fontface = "bold")
naa_levels <- c("Negative colonoscopy", "Non-neoplastic findings", "Non-advanced adenoma")
for (lvl in naa_levels) {
  add_row(c(paste0("    ", lvl),
            get_val("Info_group", lvl, "1_Discovery"),
            get_val("Info_group", lvl, "2_Internal_Test"),
            "",
            get_val("Info_group", lvl, "3_Wenzhou"),
            get_val("Info_group", lvl, "4_Shandong")))
}

add_row(c("Advanced adenomas group (%)", "", "", "/", "", ""), fontface = "bold")
adv_levels <- c("High-grade dysplasia", "Villous, focal HGD", "Villous",
                "Tubular adenoma, ≥10 mm", "Sessile serrated lesions, ≥10 mm")
for (lvl in adv_levels) {
  search_lvl <- lvl
  if (lvl == "Villous, focal HGD") search_lvl <- "Villous, focal HGD"
  if (lvl == "Tubular adenoma, ≥10 mm") search_lvl <- "Tubular adenoma"
  if (lvl == "Sessile serrated lesions, ≥10 mm") search_lvl <- "Sessile serrated lesions"

  add_row(c(paste0("    ", lvl),
            get_val("Info_group", search_lvl, "1_Discovery"),
            get_val("Info_group", search_lvl, "2_Internal_Test"),
            "",
            get_val("Info_group", search_lvl, "3_Wenzhou"),
            get_val("Info_group", search_lvl, "4_Shandong")))
}

add_row(c("CRC stages (%)", "", "", "/", "", ""), fontface = "bold")
crc_levels <- c("I", "II", "III", "IV")
for (lvl in crc_levels) {
  add_row(c(paste0("    ", lvl),
            get_val("Info_group", lvl, "1_Discovery"),
            get_val("Info_group", lvl, "2_Internal_Test"),
            "",
            get_val("Info_group", lvl, "3_Wenzhou"),
            get_val("Info_group", lvl, "4_Shandong")))
}

add_hline(lwd = 0.5)
add_row(c("Smoking status (%)", "", "", get_p("smoking_state"), "", ""), fontface = "bold")
smoke_levels <- c("Current", "Former", "Never", "No record")
for (lvl in smoke_levels) {
  add_row(c(paste0("    ", lvl),
            get_val("smoking_state", lvl, "1_Discovery"),
            get_val("smoking_state", lvl, "2_Internal_Test"),
            "",
            get_val("smoking_state", lvl, "3_Wenzhou"),
            get_val("smoking_state", lvl, "4_Shandong")))
}

add_row(c("Alcohol consumption status (%)", "", "", get_p("Alcohol_state"), "", ""), fontface = "bold")
alcohol_levels <- c("Current", "Former", "Never", "No record")
for (lvl in alcohol_levels) {
  add_row(c(paste0("    ", lvl),
            get_val("Alcohol_state", lvl, "1_Discovery"),
            get_val("Alcohol_state", lvl, "2_Internal_Test"),
            "",
            get_val("Alcohol_state", lvl, "3_Wenzhou"),
            get_val("Alcohol_state", lvl, "4_Shandong")))
}

add_hline(lwd = 0.5)
add_row(c("Complete blood counts (mean (SD))", "", "", "", "", ""), fontface = "bold")
cbc_vars <- c("RBC", "HGB", "WBC", "PLT")
cbc_labels <- c("RBC (×10¹²/L)", "HGB (g/L)", "WBC (×10⁹/L)", "PLT (×10⁹/L)")
for (i in seq_along(cbc_vars)) {
  add_row(c(paste0("    ", cbc_labels[i]),
            get_val(cbc_vars[i], cohort = "1_Discovery"),
            get_val(cbc_vars[i], cohort = "2_Internal_Test"),
            get_p(cbc_vars[i]),
            get_val(cbc_vars[i], cohort = "3_Wenzhou"),
            get_val(cbc_vars[i], cohort = "4_Shandong")))
}

add_hline(lwd = 1.5)

tg_body <- do.call(gtable_rbind, table_rows)
col_widths <- unit(c(0.3, 0.14, 0.14, 0.12, 0.15, 0.15), "npc")
tg_body$widths <- col_widths

pdf_file <- file.path(out_dir, "FigureS3.pdf")
cairo_pdf(pdf_file, width = 8, height = 11, family = "ArialMT")

title_height <- unit(1.2, "cm")
phase_height <- unit(0.5, "cm")
table_height <- grobHeight(tg_body)

grid.newpage()
pushViewport(viewport(width = 0.96, x = 0.5, just = "center"))
pushViewport(viewport(
  y = 1, just = "top",
  height = title_height + phase_height + table_height,
  layout = grid.layout(3, 1, heights = unit.c(title_height, phase_height, table_height))
))

pushViewport(viewport(layout.pos.row = 1))
grid.text("Figure S3", x = 0.02, just = c("left", "bottom"), y = 0.2, gp = gpar(fontsize = 14, fontface = "bold"))
popViewport()

pushViewport(viewport(layout.pos.row = 2))
x_cum <- cumsum(as.numeric(col_widths)) / sum(as.numeric(col_widths))
x_left <- c(0, x_cum[-length(x_cum)])

dev_x_mid <- (x_left[2] + x_cum[4]) / 2
grid.text("Development phase", x = unit(dev_x_mid, "npc"), y = 0.4,
          gp = gpar(fontsize = 7, fontface = "bold"))

val_x_mid <- (x_left[5] + x_cum[6]) / 2
grid.text("Validation phase", x = unit(val_x_mid, "npc"), y = 0.4,
          gp = gpar(fontsize = 7, fontface = "bold"))

grid.segments(x0 = unit(x_left[2], "npc"), x1 = unit(x_cum[4], "npc"),
              y0 = 0, y1 = 0, gp = gpar(lwd = 1))
grid.segments(x0 = unit(x_left[5], "npc"), x1 = unit(x_cum[6], "npc"),
              y0 = 0, y1 = 0, gp = gpar(lwd = 1))
popViewport()

pushViewport(viewport(layout.pos.row = 3))
pushViewport(viewport(y = 1, just = "top", height = table_height))
grid.draw(tg_body)
popViewport(2)

popViewport()
invisible(dev.off())

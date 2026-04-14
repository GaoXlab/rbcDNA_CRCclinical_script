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
  library(GenomicRanges)
  library(karyoploteR)
  library(ggsci)
})

load_top_granges <- function(filepath) {
  df <- read.table(filepath, sep = '\t', header = FALSE, nrows = 1000)
  colnames(df)[1:3] <- c('chr', 'start', 'end')
  df$chr <- paste0('chr', df$chr)
  return(GRanges(df))
}

zr2.hg38 <- load_top_granges('./Human_model/results/2_FeatureSelection/all.zheer_zr2_1234.bed.out')
zr6.hg38 <- load_top_granges('./Human_model/results/2_FeatureSelection/all.zheer_zr6_1234.bed.out')
zr8.hg38 <- load_top_granges('./Human_model/results/2_FeatureSelection/all.zheer_zr8_1234.bed.out')

pp <- getDefaultPlotParams(plot.type = 1)
pp$leftmargin <- 0.1
pp$ideogramheight <- 100
pp$data1height <- 200
pp$data1inmargin <- 20
pp$data1outmargin <- 40
pp$topmargin <- 20
pp$bottommargin <- 20

# Figure S4: Karyotype plot visualizing the distribution of top 1000 features across chromosomes
pdf(file.path(out_dir, 'figureS4.pdf'), width = 10, height = 6.5)
kp <- plotKaryotype(genome = "hg38", chromosomes = paste0("chr", c(1:22)), plot.type = 1, plot.params = pp)
kpPlotRegions(kp, data = zr2.hg38, data.panel = 1, col = pal_npg("nrc")(9)[4], r0 = 0.1, r1 = 0.9, border = NA)
kpPlotRegions(kp, data = zr6.hg38, data.panel = 1, col = pal_npg("nrc")(9)[5], r0 = 0.1, r1 = 0.9, border = NA)
kpPlotRegions(kp, data = zr8.hg38, data.panel = 1, col = pal_npg("nrc")(9)[1], r0 = 0.1, r1 = 0.9, border = NA)
dev.off()
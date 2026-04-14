args <- commandArgs(trailingOnly = TRUE)
working_dir <- args[1]
setwd(working_dir)

args_all <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args_all[grep("--file=", args_all)])
script_dir <- dirname(normalizePath(script_path))

out_dir <- file.path(working_dir, "Figures")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

options(repos = c(CRAN = "https://mirrors.tuna.tsinghua.edu.cn/CRAN/"))
options(BioC_mirror = "https://mirrors.tuna.tsinghua.edu.cn/bioconductor")

suppressPackageStartupMessages({
  library(stringr)
  library(scales)
  library(ggsci)
  library(pheatmap)
  library(GenomeInfoDb)
  library(GenomicRanges)
  library(reshape2)
  library(ggplot2)
  library(ggpubr)
  library(annotatr)
  library(dplyr)
})

source(file.path(script_dir, 'function.r'), chdir = TRUE)
source(file.path(script_dir, 'plot_function.r'), chdir = TRUE)
source(file.path(script_dir, 'plotAUC_theme.r'), chdir = TRUE)

load('./Figures/sampleinfo.RData')
nrc_color <- pal_npg("nrc", alpha = 0.7)(9)

# Figure 2a: Feature Selection and Heatmap Visualization

load_top_features <- function(filepath, type_name) {
  df <- read.table(filepath, sep = '\t', header = FALSE)
  df <- head(df, 1000)
  colnames(df)[1:3] <- c('chr', 'start', 'end')
  df$feature <- str_c(df$chr, ':', df$start, '-', df$end)
  df$type <- type_name
  rownames(df) <- str_c('chr', df$feature)
  return(df)
}

zr2_feas_1000 <- load_top_features('./Human_model/results/2_FeatureSelection/all.zheer_zr2_1234.bed.out', 'zr2')
zr6_feas_1000 <- load_top_features('./Human_model/results/2_FeatureSelection/all.zheer_zr6_1234.bed.out', 'zr6')
zr8_feas_1000 <- load_top_features('./Human_model/results/2_FeatureSelection/all.zheer_zr8_1234.bed.out', 'zr8')

zr6_feas_1000$type[zr6_feas_1000$feature %in% zr2_feas_1000$feature] <- 'zr6overlap2zr2'
zr8_feas_1000$type[zr8_feas_1000$feature %in% zr2_feas_1000$feature] <- 'zr8overlap2zr2'
zr8_feas_1000$type[zr8_feas_1000$feature %in% zr6_feas_1000$feature] <- 'zr8overlap2zr6'

zr2.hg38 <- GRanges(zr2_feas_1000)
zr6.hg38 <- GRanges(zr6_feas_1000)
zr8.hg38 <- GRanges(zr8_feas_1000)
zr68.hg38 <- GRanges(rbind(zr6_feas_1000, zr8_feas_1000))

coverage_frac <- function(A, B) {
  B2 <- disjoin(B, with.revmap = FALSE)
  hits <- findOverlaps(A, B2)
  out <- numeric(length(A))
  if (length(hits) == 0L) return(out)
  int <- pintersect(ranges(A)[queryHits(hits)], ranges(B2)[subjectHits(hits)])
  ovlen <- tapply(width(int), queryHits(hits), sum)
  out[as.integer(names(ovlen))] <- as.numeric(ovlen)
  return(out / width(A))
}

clean_by <- function(A, B, thresh = 0.9) {
  A[coverage_frac(A, B) < thresh]
}

A1 <- clean_by(zr2.hg38, zr68.hg38, 0.9)
A1_df <- as.data.frame(A1)
colnames(A1_df)[1] <- 'chr'

zr2clean6.hg38 <- GRanges(rbind(zr6_feas_1000, A1_df[colnames(zr2_feas_1000)]))
B1 <- clean_by(zr8.hg38, zr2clean6.hg38, 0.9)
B1_df <- as.data.frame(B1)
colnames(B1_df)[1] <- 'chr'

zr2clean8clean.hg38 <- GRanges(rbind(A1_df[colnames(zr2_feas_1000)], B1_df[colnames(zr2_feas_1000)]))
C1 <- clean_by(zr6.hg38, zr2clean8clean.hg38, 0.9)

final_features <- c(A1, B1, C1)
top_feas_1000 <- rbind(zr2_feas_1000, zr6_feas_1000, zr8_feas_1000)
top_feas_1000 <- top_feas_1000[top_feas_1000$feature %in% final_features$feature, ]

load_normalized_gam <- function(filepath, valid_features, sample_ids) {
  df <- read.table(filepath, sep = '\t', header = TRUE, comment.char = "")
  colnames(df) <- gsub('^X|.uniq.nodup.bam', '', colnames(df))
  rownames(df) <- str_c('chr', df[,1], ':', df[,2], '-', df[,3])
  return(df[intersect(rownames(df), valid_features), sample_ids, drop = FALSE])
}

trn_samples <- trn_info$Sample
all_df_zr2 <- load_normalized_gam('./Human_model/normalized_results/zheer_zr2_1234/train_gam.tab.zheer_zr2_1234',
                                  rownames(top_feas_1000[top_feas_1000$type == 'zr2', ]), trn_samples)
all_df_zr6 <- load_normalized_gam('./Human_model/normalized_results/zheer_zr6_1234/train_gam.tab.zheer_zr6_1234',
                                  rownames(top_feas_1000[top_feas_1000$type != 'zr8overlap2zr6', ]), trn_samples)
all_df_zr8 <- load_normalized_gam('./Human_model/normalized_results/zheer_zr8_1234/train_gam.tab.zheer_zr8_1234',
                                  rownames(top_feas_1000), trn_samples)

all_df <- as.data.frame(t(rbind(all_df_zr2, all_df_zr6, all_df_zr8)))
all_df_filter <- all_df[as.character(trn_samples), str_c("chr", unique(top_feas_1000$feature))]

trnval_df_filter <- merge(all_df_filter, trn_info[, c('Sample', 'Group', 'Sub-group')], by.x = 'row.names', by.y = 'Sample')
colnames(trnval_df_filter)[grep('Sub-group', colnames(trnval_df_filter))] <- 'Info_group'
rownames(trnval_df_filter) <- trnval_df_filter$Row.names
trnval_df_filter <- trnval_df_filter[, -1]

sample_annotations <- trnval_df_filter[, c('Group', 'Info_group')]
sample_annotations$SampleID <- rownames(trnval_df_filter)

sample_annotations <- sample_annotations %>%
  mutate(
    Info_group_clean = gsub(' ', '', Info_group),
    Info_group = case_when(
      Info_group_clean == '(2b)High-gradedysplasia' ~ 'aa_hgd',
      Info_group_clean == '(2c)Villous' ~ 'aa_vil',
      Info_group_clean == '(2c)Villous,focalHGD' ~ 'aa_vilhgd',
      Info_group_clean %in% c('(2d)Tubularadenoma,≥10mm', 'AA') ~ 'aa_tub10',
      Info_group_clean == '(2e)Serratedadenoma,≥10mm' ~ 'aa_ssa10',
      Info_group_clean == 'I' ~ 'crc_1',
      Info_group_clean == 'II' ~ 'crc_2',
      Info_group_clean == 'III' ~ 'crc_3',
      Info_group_clean == 'IV' ~ 'crc_4',
      Info_group_clean == 'Negativecolonoscopy' ~ 'ctrl_hd',
      Info_group_clean == 'Non-advancedadenoma(NAA)' ~ 'ctrl_xr',
      Info_group_clean == 'Nonneoplasticfindings' ~ 'ctrl_nofind',
      TRUE ~ Info_group_clean
    ),
    Group = case_when(
      Group == 'Non-AN control' ~ 'Non-AN',
      TRUE ~ 'AN'
    ),
    Group = factor(Group, levels = c('Non-AN', 'AN'))
  ) %>%
  as.data.frame()

rownames(sample_annotations) <- sample_annotations$SampleID
sample_annotations$SampleID <- NULL

ann_colors <- list(
  Group = c("Non-AN" = ggsci::pal_material("blue-grey")(10)[2], "AN" = "#E64B35FF"),
  Info_group = c("ctrl_hd" = ggsci::pal_material("blue-grey")(10)[2], "ctrl_nofind" = ggsci::pal_material("blue-grey")(10)[3], "ctrl_xr" = ggsci::pal_material("blue-grey")(10)[4],
                 "aa_ssa10" = ggsci::pal_material("orange")(10)[2], "aa_tub10" = ggsci::pal_material("orange")(10)[2],
                 "aa_vil" = ggsci::pal_material("orange")(10)[3], "aa_vilhgd" = ggsci::pal_material("orange")(10)[3], "aa_hgd" = ggsci::pal_material("orange")(10)[4],
                 "crc_1" = ggsci::pal_material("red")(10)[2], "crc_2" = ggsci::pal_material("red")(10)[3], "crc_3" = ggsci::pal_material("red")(10)[3], "crc_4" = ggsci::pal_material("red")(10)[4]),
  type = c("zr6" = "#b2182b", "zr8" = "#ff7f0e", "zr2" = "#1f77b4"),
  overlap = c("overlap" = nrc_color[3], "unique" = "grey")
)

row_annotations <- top_feas_1000[, c('feature', 'type'), drop = FALSE]
feat_counts <- table(top_feas_1000$feature)
row_annotations$overlap <- ifelse(feat_counts[row_annotations$feature] > 1, "overlap", "unique")
row_annotations$type <- gsub('overlap2.*', '', row_annotations$type)
row_annotations <- row_annotations[!duplicated(row_annotations$feature), ]
rownames(row_annotations) <- str_c("chr", row_annotations$feature)

heatmap_matrix <- t(trnval_df_filter[order(trnval_df_filter$Group), rownames(row_annotations)])
bk <- c(seq(-10, -0.1, by = 0.1), seq(0, 10, by = 0.1))
final_annot_col <- sample_annotations[colnames(heatmap_matrix), c('Info_group', 'Group'), drop = FALSE]

pdf(file.path(out_dir, 'figure2_a.pdf'), width = 6, height = 9.5)
pheatmap(heatmap_matrix, scale = 'row',
         clustering_distance_rows = 'correlation', clustering_distance_cols = 'correlation',
         annotation_col = final_annot_col,
         annotation_colors = ann_colors,
         annotation_row = row_annotations[, setdiff(colnames(row_annotations), c('overlap', 'feature')), drop = FALSE],
         cutree_rows = 2, show_rownames = FALSE, show_colnames = FALSE,
         treeheight_row = 20, treeheight_col = 40, legend = TRUE, annotation_legend = TRUE, annotation_names_row = FALSE, annotation_names_col = FALSE,
         legend_breaks = c(-10, -5, 0, 5, 10), legend_labels = c('-10', '-5', '0', '5', '10'), breaks = bk,
         color = c(colorRampPalette(colors = c("#084594", "#08519c", "white"))(length(bk)/2),
                   colorRampPalette(colors = c("white", "#cb181d", "firebrick3"))(length(bk)/2))
)
dev.off()

# Figure 2b: Regional Coverage Profiles (100kb / 10kb windows)

load('./Figures/train_gam_100k.RData')

trn_samples_aa <- intersect(trn_info$Sample[trn_info$Group == 'AA'], colnames(trn_100k))
trn_samples_crc <- intersect(trn_info$Sample[trn_info$Group == 'CRC'], colnames(trn_100k))
trn_samples_0 <- intersect(trn_info$Sample[trn_info$Group == 'Non-AN control'], colnames(trn_100k))

plot_regional_coverage <- function(data, target_chr, target_start, target_end, y_min, y_max, y_label, n_aa, n_crc) {

  region_data <- data[(data$chr == target_chr) & (data$start >= target_start) & (data$end <= target_end), ]
  sort_names <- sort(apply(region_data, 2, sum))

  s_0_top <- names(head(tail(sort(sort_names[trn_samples_0]), 300), 100))
  s_aa_top <- names(tail(head(sort(sort_names[trn_samples_aa]), n_aa), 100))
  s_crc_top <- names(head(sort(sort_names[trn_samples_crc]), n_crc))

  HD_med <- MNdna_profiles_df1(data, 'HD_med', intersect(colnames(data), s_0_top))
  AA_med <- MNdna_profiles_df1(data, 'AA_med', intersect(colnames(data), s_aa_top))
  CRC_med <- MNdna_profiles_df1(data, 'CRC_med', intersect(colnames(data), s_crc_top))

  HD_med <- cbind(data[, c('chr', 'start', 'end')], HD_med)
  AA_med <- cbind(data[, c('chr', 'start', 'end')], AA_med)
  CRC_med <- cbind(data[, c('chr', 'start', 'end')], CRC_med)

  pad_start <- target_start - 700000
  pad_end <- target_end + 500000

  HD_plot <- HD_med[(HD_med$chr == target_chr) & (HD_med$start >= pad_start) & (HD_med$start <= pad_end), ]
  AA_plot <- AA_med[(AA_med$chr == target_chr) & (AA_med$start >= pad_start) & (AA_med$start <= pad_end), ]
  CRC_plot <- CRC_med[(CRC_med$chr == target_chr) & (CRC_med$start >= pad_start) & (CRC_med$start <= pad_end), ]

  grp_hd <- paste0("Non-AN (n=", length(s_0_top), ")")
  grp_aa <- paste0("AA (n=", min(length(s_aa_top), n_aa), ")")
  grp_crc <- paste0("CRC (n=", min(length(s_crc_top), n_crc), ")")

  HD_plot$Group <- grp_hd
  AA_plot$Group <- grp_aa
  CRC_plot$Group <- grp_crc

  plot_data <- rbind(HD_plot, AA_plot, CRC_plot)
  plot_data$Group <- factor(plot_data$Group, levels = c(grp_hd, grp_aa, grp_crc))

  x_label <- sprintf("Chr%s:%s-%s", target_chr, pad_start, pad_end)
  feature_title <- sprintf("Feature (Chr%s:%s-%s)", target_chr, target_start, target_end)

  p <- ggplot() +
    geom_vline(xintercept = c(target_start, target_end), colour = "#FDC173", linewidth = 0.1, linetype = 'dashed') +
    geom_hline(yintercept = 0, colour = "grey", linewidth = 0.1, linetype = 'dashed') +
    geom_rect(aes(xmin = target_start, xmax = target_end, ymin = y_min, ymax = y_max), fill = '#FDC173', alpha = 0.1) +
    geom_line(data = plot_data, aes(x = start, y = median, color = Group), linewidth = 0.6) +
    scale_color_manual(values = setNames(c("#78909C", "#DF8F44FF", "#9F1A1AFF"), c(grp_hd, grp_aa, grp_crc))) +
    annotate("text", x = target_start + (target_end - target_start)/2, y = y_max * 1, label = feature_title, family = "sans", size = 3) +
    xlab(x_label) + ylab(y_label) +
    theme_classic(base_family = "sans") +
    theme(
      legend.position = c(0.15, 0.15),
      legend.title = element_blank(),
      legend.text = element_text(size = 8),
      legend.background = element_blank(),
      legend.key = element_blank(),
      axis.text = element_text(size = 9, color = "black"),
      axis.title = element_text(size = 10, color = "black"),
      axis.line = element_line(color = "black", linewidth = 0.5),
      axis.ticks = element_line(color = "black", linewidth = 0.5)
    )

  return(p)
}

pdf(file.path(out_dir, 'figure2_b.pdf'), width = 4.5, height = 3.2)
print(plot_regional_coverage(trn_100k, 14, 50100000, 50810000, -6, 1, 'GAM-normalized read counts\nin 100kb region', 100, 100))
print(plot_regional_coverage(trn_100k, 6, 157810000, 158790000, -5, 0.5, 'GAM-normalized read counts\nin 100kb region', 100, 100))
dev.off()

# Figure 2c: Genomic Element Overlaps

calc_percent <- function(gr_regions, ann) {
  hits <- findOverlaps(gr_regions, ann)
  if (length(hits) == 0) return(0)
  ov <- pintersect(gr_regions[queryHits(hits)], ann[subjectHits(hits)])
  return(sum(width(GenomicRanges::reduce(ov))) / sum(width(gr_regions)) * 100)
}

zr268.hg38 <- GRanges(rbind(zr2_feas_1000, zr6_feas_1000, zr8_feas_1000))

ann_intergenic <- build_annotations(genome = 'hg38', annotations = 'hg38_genes_intergenic')
ann_promoter <- build_annotations(genome = 'hg38', annotations = 'hg38_genes_promoters')
ann_exons <- build_annotations(genome = 'hg38', annotations = 'hg38_genes_exons')
ann_firstexons <- build_annotations(genome = 'hg38', annotations = 'hg38_genes_firstexons')
ann_introns <- build_annotations(genome = 'hg38', annotations = 'hg38_genes_introns')
ann_UTR5 <- build_annotations(genome = 'hg38', annotations = 'hg38_genes_5UTRs')
ann_UTR3 <- build_annotations(genome = 'hg38', annotations = 'hg38_genes_3UTRs')

strand(ann_promoter) <- "*"
strand(ann_exons) <- "*"
strand(ann_UTR5) <- "*"
strand(ann_UTR3) <- "*"
strand(ann_introns) <- "*"
strand(ann_intergenic) <- "*"

gr_regions <- GenomicRanges::reduce(zr268.hg38)
seqlevelsStyle(gr_regions) <- "UCSC"
strand(gr_regions) <- "*"

total_length <- sum(width(gr_regions))

cat_promoter <- ann_promoter
assigned <- GenomicRanges::reduce(cat_promoter)

cat_UTR5 <- GenomicRanges::setdiff(ann_UTR5, assigned)
assigned <- GenomicRanges::reduce(c(assigned, cat_UTR5))

cat_UTR3 <- GenomicRanges::setdiff(ann_UTR3, assigned)
assigned <- GenomicRanges::reduce(c(assigned, cat_UTR3))

cat_exons <- GenomicRanges::setdiff(ann_exons, assigned)
assigned <- GenomicRanges::reduce(c(assigned, cat_exons))

cat_introns <- GenomicRanges::setdiff(ann_introns, assigned)
assigned <- GenomicRanges::reduce(c(assigned, cat_introns))

cat_intergenic <- GenomicRanges::setdiff(ann_intergenic, assigned)
assigned <- GenomicRanges::reduce(c(assigned, cat_intergenic))

cat_other <- GenomicRanges::setdiff(gr_regions, assigned)

df_genomic <- data.frame(
  Percentage = c(
    calc_percent(gr_regions, cat_promoter),
    calc_percent(gr_regions, cat_exons),
    calc_percent(gr_regions, cat_UTR5),
    calc_percent(gr_regions, cat_UTR3),
    calc_percent(gr_regions, cat_introns),
    calc_percent(gr_regions, cat_intergenic),
    sum(width(cat_other)) / total_length * 100
  ),
  Element = c('promoter', 'exon', "5' UTR", "3' UTR", 'intron', 'intergenic\nregion', 'other'),
  Category = 'genomic'
)

df_genomic$Element <- factor(df_genomic$Element, levels = c('promoter', 'exon', "5' UTR", "3' UTR", 'intron', 'intergenic\nregion', 'other'))

col_genomic <- c('promoter' = '#E64B35FF', 'exon' = '#4DBBD5FF', "5' UTR" = '#00A087FF',
                 "3' UTR" = '#3C5488FF', 'intron' = '#F39B7FFF', 'intergenic\nregion' = '#8491B4FF', 'other' = '#CCCCCC')

p_genomic <- ggplot(df_genomic, aes(x = Category, y = Percentage, fill = Element)) +
  geom_bar(stat = "identity", width = 0.55) +
  geom_text(aes(label = sprintf("%.1f", Percentage)), position = position_stack(vjust = 0.5), size = 2.8, family = "ArialMT") +
  scale_fill_manual(values = col_genomic) +
  scale_y_continuous(expand = c(0, 0)) +
  ylab("Proportion of rbcDNA features\nassociated with advanced neoplasia\nacross genomic annotations (%)") +
  theme_classic(base_family = "ArialMT") +
  theme(
    axis.title.x = element_blank(),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    axis.line.x = element_blank(),
    axis.text.y = element_text(size = 9, color = "black"),
    axis.title.y = element_text(size = 10, color = "black"),
    legend.position = "right",
    legend.title = element_blank(),
    legend.text = element_text(size = 9),
    legend.key.size = unit(0.5, "cm")
  )

ggsave(file.path(out_dir, 'figure2_c.pdf'), p_genomic, width = 3.5, height = 4.2, device = cairo_pdf)

# Figure 2d: Chromatin State Enrichment vs Random Background

feature_chromanno_zr268 <- read.table('./figure_scripts/FeatureAnno/zr268.grouped_output.txt', sep = '\t', header = TRUE, row.names = 1)
target_sum <- as.data.frame(apply(feature_chromanno_zr268, 2, sum))
colnames(target_sum) <- 'target_sum'

bg_sum_all_rep10 <- c()
for(seed in 1:10) {
    set.seed(seed)
    background_chrom <- read.table(str_c("./figure_scripts/FeatureAnno/raw_", seed, ".grouped_output.txt"), sep = '\t', header = TRUE, row.names = 1)

    bg_sum <- as.data.frame(apply(background_chrom, 2, sum))
    colnames(bg_sum) <- 'bg_sum'
    bg_sum <- merge(bg_sum, target_sum, by = 'row.names')

    bg_sum$FC <- bg_sum$target_sum / bg_sum$bg_sum
    bg_sum$log2FC <- log2(bg_sum$FC)
    bg_sum$random_label <- seed
    bg_sum_all_rep10 <- rbind(bg_sum_all_rep10, bg_sum[, c('Row.names', 'bg_sum', 'target_sum', 'FC', 'log2FC', 'random_label')])
}

chrom_state_summary <- bg_sum_all_rep10 %>%
  group_by(Row.names) %>%
  summarise(
    mean_log2FC = mean(log2FC, na.rm = TRUE),
    sd_log2FC = sd(log2FC, na.rm = TRUE)
  ) %>%
  rename(State = Row.names)

chrom_state_summary$State <- factor(chrom_state_summary$State, levels = sort(unique(as.character(chrom_state_summary$State))))

p_chromstate <- ggplot(chrom_state_summary, aes(x = State, y = mean_log2FC, fill = mean_log2FC)) +
  geom_bar(stat = "identity", width = 0.75) +
  geom_errorbar(aes(ymin = mean_log2FC - sd_log2FC, ymax = mean_log2FC + sd_log2FC), width = 0.25, linewidth = 0.5) +
  coord_flip() +
  labs(y = expression(paste("Log"[2], "(fold change)")), x = "Chromatin states") +
  scale_fill_gradient2(low = "#3C5488FF", mid = "white", high = "#E64B35FF", midpoint = 0, name = expression(paste("Log"[2], "(fc)"))) +
  scale_y_continuous(limits = c(-2.5, 2.5), breaks = seq(-2, 2, 1)) +
  theme_classic(base_family = "sans") +
  theme(
    axis.text.y = element_text(size = 9, color = "black"),
    axis.text.x = element_text(size = 9, color = "black"),
    axis.title = element_text(size = 10, color = "black"),
    legend.position = "right",
    legend.title = element_text(size = 9),
    legend.text = element_text(size = 8),
    legend.key.height = unit(0.5, "cm"),
    axis.line = element_line(color = "black", linewidth = 0.5),
    axis.ticks = element_line(color = "black", linewidth = 0.5),
    plot.margin = margin(t = 5, r = 5, b = 5, l = 5)
  )

ggsave(file.path(out_dir, 'figure2_d.pdf'), p_chromstate, width = 4.8, height = 4.2, device = cairo_pdf)

# Figure 2e: Pathway Enrichment Analysis (C2:CP from MSigDB)

top_feas_1000 <- rbind(zr2_feas_1000, zr6_feas_1000, zr8_feas_1000)
zr2zr6zr8_top1000_features <- get_region_anno(str_c("chr", unique(top_feas_1000$feature)), 'Figures/zr2zr6zr8_top1000')

top20_sel <- zr2zr6zr8_top1000_features %>%
  filter(label %in% c('C2_CP', 'C2_KEGG'), !grepl('WP_', id)) %>%
  distinct(id, .keep_all = TRUE)

top20_sel$id <- factor(top20_sel$id, levels = top20_sel$id)

p_pathway <- ggplot(data = top20_sel, aes(x = fold_enrichment_hyper, y = reorder(id, fold_enrichment_hyper), fill = -log10(p_adjust_hyper))) +
  scale_fill_material("red") +
  geom_bar(stat = "identity", width = 0.5, alpha = 0.8) +
  scale_x_continuous(expand = c(0, 0)) +
  labs(x = "Fold enrichment", y = "", title = "msigdb, canonical pathways") +
  geom_text(size = 3.8, aes(x = 0.05, label = id), hjust = 0) +
  theme_classic() +
  mytheme + theme(legend.position = 'right')

ggsave(file.path(out_dir, 'figure2_e.pdf'), p_pathway, width = 9, height = 5)
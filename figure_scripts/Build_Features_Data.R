library(stringr)
library(dplyr)
library(GenomicRanges)

zr2_feas = read.table('Human_model/results/2_FeatureSelection/all.zheer_zr2_1234.bed.out', sep='\t', head=FALSE)
rownames(zr2_feas) = str_c('chr', zr2_feas[,1], ':', zr2_feas[,2], '-', zr2_feas[,3])
zr2_feas_1000 = head(zr2_feas, 1000)
zr2_feas_1000$feature = str_c(zr2_feas_1000[,1], ':', zr2_feas_1000[,2], '-', zr2_feas_1000[,3])
colnames(zr2_feas_1000)[1:3] = c('chr', 'start', 'end')
zr2_feas_1000$type = 'zr2'

zr6_feas = read.table('Human_model/results/2_FeatureSelection/all.zheer_zr6_1234.bed.out', sep='\t', head=FALSE)
rownames(zr6_feas) = str_c('chr', zr6_feas[,1], ':', zr6_feas[,2], '-', zr6_feas[,3])
zr6_feas_1000 = head(zr6_feas, 1000)
zr6_feas_1000$feature = str_c(zr6_feas_1000[,1], ':', zr6_feas_1000[,2], '-', zr6_feas_1000[,3])
colnames(zr6_feas_1000)[1:3] = c('chr', 'start', 'end')
zr6_feas_1000$type = 'zr6'
zr6_feas_1000[which(zr6_feas_1000$feature %in% zr2_feas_1000$feature), 'type'] = 'zr6overlap2zr2'

zr8_feas = read.table('Human_model/results/2_FeatureSelection/all.zheer_zr8_1234.bed.out', sep='\t', head=FALSE)
rownames(zr8_feas) = str_c('chr', zr8_feas[,1], ':', zr8_feas[,2], '-', zr8_feas[,3])
zr8_feas_1000 = head(zr8_feas, 1000)
zr8_feas_1000$feature = str_c(zr8_feas_1000[,1], ':', zr8_feas_1000[,2], '-', zr8_feas_1000[,3])
colnames(zr8_feas_1000)[1:3] = c('chr', 'start', 'end')
zr8_feas_1000$type = 'zr8'

zr268 = as.data.frame(rbind(zr2_feas_1000, zr6_feas_1000, zr8_feas_1000))
zr268 = zr268[!duplicated(zr268$feature),]
write.table(rownames(zr268), str_c("figure_scripts/FeatureAnno/zr268.bed"), sep='\t', row.names=FALSE, col.names=FALSE, quote=FALSE)

zr268.hg38 = GRanges(zr268)
gr_regions = zr268.hg38
strand(gr_regions) <- "*"
feature_bed = as.data.frame(gr_regions)
rownames(feature_bed) = str_c(feature_bed$seqnames, ':', feature_bed$start, '-', feature_bed$end)

random_lists_10 <- vector("list", 10)
bed_df <- read.table("figure_scripts/FeatureAnno/total_1000w.bed", header = FALSE, col.names = c("chr", "start", "end"))
bed_df$width <- bed_df$end - bed_df$start
bed_df$chr <- str_c('chr', bed_df$chr)
rownames(bed_df) = str_c(bed_df$chr, ':', bed_df$start, '-', bed_df$end)

target_lengths <- width(gr_regions) - 1
length_dist <- table(target_lengths)

for(i in 1:10){
    set.seed(i)
    sampled_df_list <- list()
    for (len in as.numeric(names(length_dist))) {
      n <- length_dist[as.character(len)]
      candidate_rows <- bed_df %>% dplyr::filter(width == len)
      if (nrow(candidate_rows) >= n) {
        sampled_df <- candidate_rows %>% sample_n(n)
      } else if (nrow(candidate_rows) > 0) {
        sampled_df <- candidate_rows %>% sample_n(nrow(candidate_rows), replace = FALSE)
        warning(sprintf("Only %d candidates for length %d (needed %d)", nrow(candidate_rows), len, n))
      } else {
        next 
      }
      sampled_df_list[[as.character(len)]] <- sampled_df
    }
    sampled_df_all <- bind_rows(sampled_df_list)
    write.table(rownames(sampled_df_all), str_c("figure_scripts/FeatureAnno/raw_", i, ".txt"), sep='\t', row.names=FALSE, col.names=FALSE, quote=FALSE)
    random_gr <- GRanges(sampled_df_all)
    random_lists_10[[i]] <- random_gr
}
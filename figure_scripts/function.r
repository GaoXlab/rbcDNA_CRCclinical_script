suppressPackageStartupMessages({
    library(stringr)
    library(ggsci)
    library(ggplot2)
    library(dbplyr)
    library(rGREAT)
    library(ggpubr)
    library(tidyr)
    library(clusterProfiler)
    library(dplyr)
})

get_region_anno <- function(featurelist, filename){
    all_sig <- c()
    region_df <- tibble(region = featurelist) %>%
        separate(region, into = c("chr", "pos"), sep = ":") %>%
        separate(pos, into = c("start", "end"), sep = "-") %>%
        mutate(across(start:end, as.numeric))

    print(nrow(region_df))

    up.distal.bed <- region_df[, c('chr','start','end')]
    colnames(up.distal.bed) <- c('chr','start','end')
    gr <- GRanges(up.distal.bed)
    res <- great(gr, "msigdb:C2:CP", "txdb:hg38", basal_upstream = 0, basal_downstream = 0, extension = 500000)

    tb <- getEnrichmentTable(res)
    sig <- tb[which(tb$p_adjust_hyper < 0.05), ]
    tb <- tb[order(tb$fold_enrichment_hyper, decreasing = TRUE), ]

    if(nrow(sig) > 0){
        sig$label <- 'C2_CP'
        all_sig <- as.data.frame(rbind(all_sig, sig))
    }
    write.xlsx(list(all_sig), 'feature_region_annotation.xlsx', sheetName = filename, append = TRUE)
    return(all_sig)
}

mytheme <- theme(
    axis.title = element_text(size = 13),
    axis.text = element_text(size = 11),
    axis.text.y = element_blank(),
    axis.ticks.length.y = unit(0, "cm"),
    plot.title = element_text(size = 13, hjust = 0.5, face = "bold"),
    legend.position = 'bottom',
    legend.title = element_text(size = 13),
    legend.text = element_text(size = 11)
)
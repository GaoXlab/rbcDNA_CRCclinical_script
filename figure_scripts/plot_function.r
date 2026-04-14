suppressPackageStartupMessages({
    library(ggplot2)
    library(dplyr)
    library(cutpointr)
    library(pROC)
    library(stringr)
})

Cutoff <- function(desired_specificity, pred){
    pred$Target <- factor(pred$Target)
    cutoff_atspe <- pred %>%
        cutpointr::roc(x = merged_score, class = Target,
            pos_class = "1",
            neg_class = "0",
            direction = ">=") %>%
        dplyr::mutate(sens = tp / (tp + fn),
               spec = 1 - fpr) %>%
        dplyr::filter(spec > desired_specificity, is.finite(x.sorted)) %>%
        dplyr::pull(x.sorted) %>%
        min()
    return(cutoff_atspe)
}

mytheme <- theme_classic(base_size = 6) + theme(
    axis.text.x = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks.x = element_blank(),
    strip.text.y = element_blank(),
    strip.text.x = element_text(size = 5),
    strip.background.x = element_blank(),
    strip.placement = "outside",
    axis.title.x = element_blank(),
    legend.position = "none",
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
)

theme_bar <- theme_bw() + theme(
    legend.position = 'bottom',
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
)

theme_sig <- theme_classic() + theme(
    legend.position = 'none',
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
    strip.background = element_blank()
)

MNdna_profiles_df1 <- function(df, label, label_samples){
    df <- df[, label_samples]
    df <- cbind(rownames(df), apply(df, 1, median), apply(df, 1, min), apply(df, 1, max))
    colnames(df) <- c('region', 'median', 'min', 'max')
    df <- as.data.frame(df)
    df$label <- label
    df$median <- as.numeric(df$median)
    df$min <- as.numeric(df$min)
    df$max <- as.numeric(df$max)
    return(df)
}

get_sensitivity <- function(pred, threshold) {
    pred$binary <- as.numeric(pred$merged_score >= threshold)
    n <- sum(pred$Target == 1, na.rm = TRUE)

    detected <- sum(pred$Target == 1 & pred$binary == 1, na.rm = TRUE)
    exact_prop <- ifelse(n > 0, detected / n, NA)

    if (n > 0) {
        roc <- suppressWarnings(reportROC::reportROC(gold = pred$Target,
                                          predictor.binary = pred$binary,
                                          plot = FALSE, important = "se")[c("SEN", "SEN.low", "SEN.up")])
        sen <- as.numeric(roc)
    } else {
        sen <- c(NA, NA, NA)
    }

    res <- data.frame(
        Label = "Total", Freq = n, Detected = detected,
        SEN = round(exact_prop * 100, 2),
        SEN.low = pmax(0, round(sen[2] * 100, 2)),
        SEN.up = pmin(100, round(sen[3] * 100, 2)),
        perc = str_c(round(exact_prop * 100), '%\n(', detected, '/', n, ')'),
        CI95 = paste0(round(exact_prop * 100), "(", round(pmax(0, round(sen[2] * 100))), "-", round(pmin(100, round(sen[3] * 100))), ")")
    )
    return(res)
}

get_specificity <- function(pred, threshold) {
    pred$binary <- as.numeric(pred$merged_score >= threshold)
    n <- sum(pred$Target == 0, na.rm = TRUE)

    detected <- sum(pred$Target == 0 & pred$binary == 0, na.rm = TRUE)
    exact_prop <- ifelse(n > 0, detected / n, NA)

    if (n > 0) {
        roc <- suppressWarnings(reportROC::reportROC(gold = pred$Target,
                                          predictor.binary = pred$binary,
                                          plot = FALSE, important = "se")[c("SPE", "SPE.low", "SPE.up")])
        spe <- as.numeric(roc)
    } else {
        spe <- c(NA, NA, NA)
    }

    res <- data.frame(
        Label = "HD_Total", Freq = n, Detected = detected,
        SPE = round(exact_prop * 100, 2),
        SPE.low = pmax(0, round(spe[2] * 100, 2)),
        SPE.up = pmin(100, round(spe[3] * 100, 2)),
        perc = str_c(round(exact_prop * 100), '%\n(', detected, '/', n, ')'),
        CI95 = paste0(round(exact_prop * 100), "(", round(pmax(0, round(spe[2] * 100))), "-", round(pmin(100, round(spe[3] * 100))), ")")
    )
    return(res)
}

evaluate_all_sets <- function(inds, cutoff, label) {
    sen <- get_sensitivity(inds, cutoff)
    spe <- get_specificity(inds, cutoff)

    sen$classify <- label
    spe$classify <- label

    return(list(sensitivity = sen, specificity = spe))
}
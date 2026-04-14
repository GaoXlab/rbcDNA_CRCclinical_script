suppressPackageStartupMessages({
    library(pROC)
    library(stringr)
    library(reportROC)
    library(ggplot2)
    library(cowplot)
})

plot_auc_95CI_1 <- function(test, color1, color2){
    par(mar = c(4, 4, 2, 1))
    test$Target <- as.factor(test$Target)
    test$merged_score <- as.numeric(test$merged_score)
    set.seed(1234)

    roc3 <- pROC::roc(test$Target, test$merged_score, levels = c(0, 1), percent = TRUE)
    roc3_random <- pROC::roc(sample(test$Target), test$merged_score, levels = c(0, 1), percent = TRUE)
    sp.obj3 <- pROC::ci.sp(roc3, sensitivities = seq(0, 100, 1), boot.n = 100, conf.level = 0.95)

    p_aa <- plot(roc3, col = color1, lwd = 3, cex.axis = 0.8, cex.lab = 1.2,
                 legacy.axes = TRUE,
                 xlab = "100-Specificity (%)", ylab = "Sensitivity (%)",
                 family = "ArialMT", las = 1,
                 mgp = c(2, 0.7, 0))

    p_aa <- plot(roc3_random, add = TRUE, col = 'grey', lwd = 1, legacy.axes = TRUE)
    p_aa <- plot(sp.obj3, type = "shape", col = color2, lty = 0, add = TRUE)
    p_aa <- plot(roc3, add = TRUE, col = color1, lwd = 4, legacy.axes = TRUE)

    print(str_c('Test:', round(pROC::ci.auc(roc3)[2],0), " (", round(pROC::ci.auc(roc3)[1],0), "-", round(pROC::ci.auc(roc3)[3],0), ")"))

    auc_val <- round(pROC::ci.auc(roc3)[2], 0)
    auc_low <- round(pROC::ci.auc(roc3)[1], 0)
    auc_up  <- round(pROC::ci.auc(roc3)[3], 0)

    auc_text <- paste0("Advanced neoplasia: ", auc_val, " (", auc_low, "-", auc_up, ")")

    text_x <- 0

    text(x = text_x, y = 32, labels = "Test cohort", col = "black", cex = 1, adj = 1, family = "ArialMT")
    text(x = text_x, y = 24, labels = "AUC (%, 95CI%)", col = "black", cex = 1, adj = 1, family = "ArialMT")
    text(x = text_x, y = 16, labels = auc_text, col = substr(color1, 1, 7), cex = 1, adj = 1, family = "ArialMT")
    text(x = text_x, y = 8,  labels = "Random Classifiers", col = "grey", cex = 1, adj = 1, family = "ArialMT")

    return(p_aa)
}

plot_auc_95CI_1_withFIT <- function(test, color1, color2){
    test$Target <- as.factor(test$Target)
    test$merged_score <- as.numeric(test$merged_score)
    set.seed(1234)

    roc3 <- pROC::roc(test$Target, test$merged_score, levels = c(0, 1), percent = TRUE)
    roc3_random <- pROC::roc(sample(test$Target), test$merged_score, levels = c(0, 1), percent = TRUE)
    sp.obj3 <- pROC::ci.sp(roc3, sensitivities = seq(0, 100, 1), boot.n = 100, conf.level = 0.95)

    p_aa <- plot(roc3, col = color1, lwd = 3, cex.axis = 0.8, cex.lab = 1.2,
                 legacy.axes = TRUE,
                 xlab = "100-Specificity (%)", ylab = "Sensitivity (%)", family="ArialMT", las = 1,
                 mgp = c(2, 0.7, 0))

    p_aa <- plot(roc3_random, add = TRUE, col = 'grey', lwd = 1, legacy.axes = TRUE)
    p_aa <- plot(sp.obj3, type = "shape", col = color2, lty = 0, add = TRUE)
    p_aa <- plot(roc3, add = TRUE, col = color1, lwd = 4, legacy.axes = TRUE)

    print(str_c('Test:', round(pROC::ci.auc(roc3)[2],0), " (", round(pROC::ci.auc(roc3)[1],0), "-", round(pROC::ci.auc(roc3)[3],0), ")"))

    fit_sen <- (nrow(test[(test$Target == 1) & (test$FIT_results == 1), ]) / nrow(test[(test$Target == 1), ])) * 100
    fit_spe <- (nrow(test[(test$Target == 0) & (test$FIT_results == 0), ]) / nrow(test[(test$Target == 0), ])) * 100

    p_aa <- points(x = fit_spe, y = fit_sen, col = '#FBB040', pch = 19, lwd = 2)
    text(x = fit_spe - 4, y = fit_sen - 5, labels = "qFIT", col = '#FBB040', cex = 0.75)

    mn_sen <- (nrow(test[(test$Target == 1) & (test$predLabel == 1), ]) / nrow(test[(test$Target == 1), ])) * 100
    mn_spe <- (nrow(test[(test$Target == 0) & (test$predLabel == 0), ]) / nrow(test[(test$Target == 0), ])) * 100

    p_aa <- points(x = mn_spe, y = mn_sen, col = color1, pch = 17, lwd = 2)
    text(x = mn_spe, y = mn_sen + 9, labels = "rbcDNA", col = color1, cex = 0.75)

    auc_val <- round(pROC::ci.auc(roc3)[2], 0)
    auc_low <- round(pROC::ci.auc(roc3)[1], 0)
    auc_up  <- round(pROC::ci.auc(roc3)[3], 0)

    auc_text <- paste0("Advanced neoplasia: ", auc_val, " (", auc_low, "-", auc_up, ")")

    text_x <- 0

    text(x = text_x, y = 24, labels = "AUC (%, 95CI%)", col = "black", cex = 1, adj = 1)
    text(x = text_x, y = 16, labels = auc_text, col = color1, cex = 1, adj = 1)
    text(x = text_x, y = 8,  labels = "Random Classifiers", col = "grey", cex = 1, adj = 1)

    return(p_aa)
}

plot_auc_95CI_2_withFIT <- function(test1, test2){
    color1 <- '#009687FF'
    color2 <- '#4CB6AC99'
    test1$Target <- as.factor(test1$Target)
    test1$merged_score <- as.numeric(test1$merged_score)
    set.seed(1234)

    roc3 <- pROC::roc(test1$Target, test1$merged_score, levels = c(0, 1), percent = TRUE)
    roc3_random <- pROC::roc(sample(test1$Target), test1$merged_score, levels = c(0, 1), percent = TRUE)
    sp.obj3 <- pROC::ci.sp(roc3, sensitivities = seq(0, 100, 1), boot.n = 100, conf.level = 0.95)

    p_aa <- plot(roc3, col = color1, lwd = 3, cex.axis = 0.8, cex.lab = 1.2,
                 legacy.axes = TRUE,
                 xlab = "100-Specificity (%)", ylab = "Sensitivity (%)",
                 family = "ArialMT", las = 1,
                 mgp = c(2, 0.7, 0))

    p_aa <- plot(roc3_random, col = 'grey', lwd = 1, add = TRUE, legacy.axes = TRUE)
    p_aa <- plot(sp.obj3, type = "shape", col = color2, lty = 0, add = TRUE)
    p_aa <- plot(roc3, add = TRUE, col = color1, lwd = 4, legacy.axes = TRUE)

    print(str_c('Test:', round(pROC::ci.auc(roc3)[2],0), " (", round(pROC::ci.auc(roc3)[1],0), "-", round(pROC::ci.auc(roc3)[3],0), ")"))

    fit_sen <- (nrow(test1[(test1$Target == 1) & (test1$FIT_results == 1), ]) / nrow(test1[(test1$Target == 1), ])) * 100
    fit_spe <- (nrow(test1[(test1$Target == 0) & (test1$FIT_results == 0), ]) / nrow(test1[(test1$Target == 0), ])) * 100
    p_aa <- points(x = fit_spe, y = fit_sen, col = '#FBB040', pch = 19, lwd = 2)
    text(x = fit_spe + 4, y = fit_sen - 6, labels = "qFIT", col = '#FBB040', cex = 0.75, family = "ArialMT")

    mn_sen <- (nrow(test1[(test1$Target == 1) & (test1$predLabel == 1), ]) / nrow(test1[(test1$Target == 1), ])) * 100
    mn_spe <- (nrow(test1[(test1$Target == 0) & (test1$predLabel == 0), ]) / nrow(test1[(test1$Target == 0), ])) * 100
    p_aa <- points(x = mn_spe, y = mn_sen, col = color1, pch = 17, lwd = 2)
    text(x = mn_spe, y = mn_sen + 6, labels = "rbcDNA", col = color1, cex = 0.75, family = "ArialMT")

    color1_2 <- '#077085FF'
    color2_2 <- '#82E1C57F'
    test2$Target <- as.factor(test2$Target)
    test2$merged_score <- as.numeric(test2$merged_score)

    roc3_2 <- pROC::roc(test2$Target, test2$merged_score, levels = c(0, 1), percent = TRUE)
    roc3_2_random <- pROC::roc(sample(test2$Target), test2$merged_score, levels = c(0, 1), percent = TRUE)
    sp.obj3_2 <- pROC::ci.sp(roc3_2, sensitivities = seq(0, 100, 1), boot.n = 100, conf.level = 0.95)

    p_aa <- plot(roc3_2, col = color1_2, lwd = 3, cex.axis = 1.2, cex.lab = 1.2, add = TRUE, legacy.axes = TRUE)
    p_aa <- plot(roc3_2_random, col = 'grey', lwd = 1, add = TRUE, legacy.axes = TRUE)
    p_aa <- plot(sp.obj3_2, type = "shape", col = color2_2, lty = 0, add = TRUE)
    p_aa <- plot(roc3_2, add = TRUE, col = color1_2, lwd = 4, legacy.axes = TRUE)

    print(str_c('Test:', round(pROC::ci.auc(roc3_2)[2],0), " (", round(pROC::ci.auc(roc3_2)[1],0), "-", round(pROC::ci.auc(roc3_2)[3],0), ")"))

    fit_sen2 <- (nrow(test2[(test2$Target == 1) & (test2$FIT_results == 1), ]) / nrow(test2[(test2$Target == 1), ])) * 100
    fit_spe2 <- (nrow(test2[(test2$Target == 0) & (test2$FIT_results == 0), ]) / nrow(test2[(test2$Target == 0), ])) * 100
    p_aa <- points(x = fit_spe2, y = fit_sen2, col = '#FBB040', pch = 19, lwd = 2)
    text(x = fit_spe2 - 4, y = fit_sen2 - 6, labels = "qFIT", col = '#FBB040', cex = 0.75, family = "ArialMT")

    mn_sen2 <- (nrow(test2[(test2$Target == 1) & (test2$predLabel == 1), ]) / nrow(test2[(test2$Target == 1), ])) * 100
    mn_spe2 <- (nrow(test2[(test2$Target == 0) & (test2$predLabel == 0), ]) / nrow(test2[(test2$Target == 0), ])) * 100
    p_aa <- points(x = mn_spe2, y = mn_sen2, col = color1_2, pch = 17, lwd = 2)
    text(x = mn_spe2 - 10, y = mn_sen2 - 4, labels = "rbcDNA", col = color1_2, cex = 0.75, family = "ArialMT")

    auc_val_1 <- round(pROC::ci.auc(roc3)[2], 0)
    auc_low_1 <- round(pROC::ci.auc(roc3)[1], 0)
    auc_up_1  <- round(pROC::ci.auc(roc3)[3], 0)

    auc_val_2 <- round(pROC::ci.auc(roc3_2)[2], 0)
    auc_low_2 <- round(pROC::ci.auc(roc3_2)[1], 0)
    auc_up_2  <- round(pROC::ci.auc(roc3_2)[3], 0)

    auc_text_1 <- paste0("CRC vs. non-AN: ", auc_val_1, " (", auc_low_1, "-", auc_up_1, ")")
    auc_text_2 <- paste0("AA vs. non-AN: ", auc_val_2, " (", auc_low_2, "-", auc_up_2, ")")

    text_x <- 0

    text(x = text_x, y = 32, labels = "AUC (%, 95CI%)", col = "black", cex = 1, adj = 1, family = "ArialMT")
    text(x = text_x, y = 24, labels = auc_text_1, col = color1, cex = 1, adj = 1, family = "ArialMT")
    text(x = text_x, y = 16, labels = auc_text_2, col = color1_2, cex = 1, adj = 1, family = "ArialMT")
    text(x = text_x, y = 8,  labels = "Random Classifiers", col = "grey", cex = 1, adj = 1, family = "ArialMT")

    return(p_aa)
}

plot_auc_95CI_AACRC <- function(test1, test2){
    test1$Target <- as.factor(test1$Target)
    test1$merged_score <- as.numeric(test1$merged_score)
    test2$Target <- as.factor(test2$Target)
    test2$merged_score <- as.numeric(test2$merged_score)
    roc1 <- pROC::roc(test1$Target, test1$merged_score, levels = c(0, 1))
    roc3 <- pROC::roc(test2$Target, test2$merged_score, levels = c(0, 1))
    set.seed(1234)
    roc1_random <- pROC::roc(sample(test1$Target), test1$merged_score, levels = c(0, 1))
    roc3_random <- pROC::roc(sample(test2$Target), test2$merged_score, levels = c(0, 1))
    sp.obj1 <- pROC::ci.sp(roc1, sensitivities = seq(0, 1, .01), boot.n = 100, conf.level = 0.95)
    sp.obj3 <- pROC::ci.sp(roc3, sensitivities = seq(0, 1, .01), boot.n = 100, conf.level = 0.95)

    p <- plot(roc1, col = 'darkred', lwd = 4, cex.axis = 1.2, cex.lab = 2)
    p <- plot(roc3, add = TRUE, col = "#F0643F", lwd = 4, cex.axis = 1.2, cex.lab = 2)
    p <- plot(roc1_random, add = TRUE, col = 'grey', lwd = 1)
    p <- plot(roc3_random, add = TRUE, col = 'grey', lwd = 1)
    p <- plot(sp.obj1, type = "shape", col = rgb(255, 0, 0, 20, maxColorValue = 255), lty = 0, add = TRUE)
    p <- plot(sp.obj3, type = "shape", col = rgb(240, 100, 63, 20, maxColorValue = 255), lty = 0, add = TRUE)
    p <- plot(roc1, add = TRUE, col = 'darkred', lwd = 4)
    p <- plot(roc3, add = TRUE, col = "#F0643F", lwd = 4)

    print(str_c('CRC:', round(pROC::ci.auc(roc1)[2]*100,0), " (", round(pROC::ci.auc(roc1)[1]*100,0), "-", round(pROC::ci.auc(roc1)[3]*100,0), ")"))
    print(str_c('AA:', round(pROC::ci.auc(roc3)[2]*100,0), " (", round(pROC::ci.auc(roc3)[1]*100,0), "-", round(pROC::ci.auc(roc3)[3]*100,0), ")"))
    return(p)
}

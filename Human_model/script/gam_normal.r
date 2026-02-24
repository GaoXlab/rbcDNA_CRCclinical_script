lib_path <- Sys.getenv("LUCKY_R_LIB_PATH")

.libPaths(lib_path)
library(lme4)
library(mgcv)
# read file name from cli argument
args = commandArgs(trailingOnly=TRUE)
filename = args[1]
output_filename = args[2]
# read csv file
df_all = read.csv(filename, sep = ',')
# filter df_all by label=train
df <- df_all[df_all$label=='train',]
#output df to df.csv
write.csv(df, "df.csv", row.names=FALSE)
df_output <- data.frame(df_all$seqID)
df_trans_output = data.frame(df_all$seqID)
df_median_output = data.frame(df_all$seqID)

cname <- colnames(df)

# get mean value of g2 and r2
r_depleted_mean <- mean(df$g2)
r_enriched_mean = mean(df$r2)

for(i in 5:ncol(df)){
    name <- cname[i]
    #output i to terminal
    gam_model <- gam(get(cname[i]) ~ s(r_depleted, r_enriched), data = df)
    res <- predict(gam_model, newdata=df_all, type="response")
    res_mean <- predict(gam_model, newdata=data.frame(r_depleted=r_depleted_mean, r_enriched=r_enriched_mean), type="response")

    resi <- df_all[, name] - res

    # round to 6 decimal places
    df_output[,name] <- round(resi, 8)
}
write.csv(df_output, output_filename, row.names=FALSE)


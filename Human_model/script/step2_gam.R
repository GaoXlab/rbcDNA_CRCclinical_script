.libPaths('YOU_SHOULD_SET_YOUR_R_!!3.6!!_LIB_PATH_HERE')
library(stringr)
# library(openxlsx)
library(dplyr)
library(parallel)
library(mgcv)
start_time <- Sys.time()
options(show.error.locations = TRUE)
# Configuration
# num_cores <- detectCores() - 1  # Use all but one core
num_cores <- 16  # Use all but one core
batch_size <- 10000  # Process features in batches

# Optimized GAM prediction function
gam_nor_2f_predict <- function(data_train, data_test, qc_column1, qc_column2) {
  # Prepare data
  colnames(data_train)[2] <- 'fea'
  colnames(data_train)[grep(qc_column1,  colnames(data_train))] = 'qc1'
  colnames(data_train)[grep(qc_column2,  colnames(data_train))] = 'qc2'
  colnames(data_test)[2] <- 'fea'
  colnames(data_test)[grep(qc_column1,  colnames(data_test))] = 'qc1'
  colnames(data_test)[grep(qc_column2,  colnames(data_test))] = 'qc2'
  
  # Fit GAM model with optimized parameters
  gam_model <- gam(fea ~ s(qc1, qc2), 
                  data = data_train)
  
  # Make predictions
  train_pred <- predict(gam_model, type = "response")
  test_pred <- predict(gam_model, newdata = data_test, type = "response")
  
  # Calculate residuals
  train_resid <- data_train$fea - train_pred
  test_resid <- data_test$fea - test_pred
  
  # Return results
  list(
    train = data.frame(
      Row.names = rownames(data_train),
      predictions = train_pred,
      residuals = round(train_resid, 8)
    ),
    test = data.frame(
      Row.names = rownames(data_test),
      predictions = test_pred,
      residuals = round(test_resid, 8)
    )
  )
}

# Batch processing function
process_batch <- function(batch_features, df_10kb, train_HD, test_else, r4g4_samples) {
  results <- mclapply(batch_features, function(fea) {
    tryCatch({
      # Prepare data with preserved column names
      train_t <- as.data.frame(t(df_10kb[fea, train_HD, drop = FALSE]))
      test_t <- as.data.frame(t(df_10kb[fea, test_else, drop = FALSE]))
      # Add sample IDs as column
      train_t$sample_id <- rownames(train_t)
      test_t$sample_id <- rownames(test_t)
      # Merge while preserving sample IDs
      train_m <- merge(train_t, r4g4_samples, by.x = 'sample_id', by.y = 'seqID', sort=FALSE); rownames(train_m) = train_m$sample_id
      test_m <- merge(test_t, r4g4_samples, by.x = 'sample_id', by.y = 'seqID', sort=FALSE); rownames(test_m) = test_m$sample_id
      
      # Verify sample correspondence
      if (nrow(train_m) != length(train_HD) || nrow(test_m) != length(test_else)) {
        stop("Sample correspondence error in feature: ", fea)
      }
      
      # GAM prediction
      result <- gam_nor_2f_predict(train_m, test_m, 'r_depleted_value', 'r_enriched_value')
      # Calculate transformed values
      train_sd <- sd(train_m[,2])
      train_mean <- mean(train_m[,2])
      
      result$train$resi_value <- result$train$residuals * train_sd + train_mean
      result$test$resi_value <- result$test$residuals * train_sd + train_mean
      
      # Store results separately to maintain correct structure
      list(
        train = list(
          predictions = result$train[, c('Row.names', 'predictions')],
          residuals = result$train[, c('Row.names', 'residuals')],
          resi_value = result$train[, c('Row.names', 'resi_value')]
        ),
        test = list(
          predictions = result$test[, c('Row.names', 'predictions')],
          residuals = result$test[, c('Row.names', 'residuals')],
          resi_value = result$test[, c('Row.names', 'resi_value')]
        )
      )
    }, error = function(e) {
      print(paste("Error processing feature", fea, ":", e$message))
      warning(paste("Error processing feature", fea, ":", e$message))
      NULL
    })
  }, mc.cores = num_cores)
  # Filter out NULL results
  results <- Filter(Negate(is.null), results)
  
  # Combine results with consistent sample ordering
  combine_with_ids <- function(results, type1, type2) {
    # Get all sample IDs in correct order
    expected_samples <- c(train_HD, test_else)
    
    # Initialize empty matrix with correct dimensions
    combined <- matrix(nrow = length(expected_samples), 
                      ncol = length(results),
                      dimnames = list(expected_samples, NULL))
    
    # Fill matrix while preserving sample order
    for (i in seq_along(results)) {
      res <- results[[i]][[type1]][[type2]]
      # Match samples by ID
      sample_order <- match(expected_samples, as.character(res[,1]))
      # if (any(is.na(sample_order))) {
      #   stop("Missing samples in feature ", i)
      # }
      combined[,i] <- res[sample_order, 2]
    }
    
    # Convert to data frame with proper column names
    colnames(combined) <- batch_features[seq_along(results)]
    as.data.frame(combined)
  }
  
  train_predictions <- combine_with_ids(results, "train", "predictions")
  test_predictions <- combine_with_ids(results, "test", "predictions")
  
  train_residuals <- combine_with_ids(results, "train", "residuals")
  test_residuals <- combine_with_ids(results, "test", "residuals")
  
  train_resi_values <- combine_with_ids(results, "train", "resi_value")
  test_resi_values <- combine_with_ids(results, "test", "resi_value")
  
  if(all(colnames(na.omit(test_predictions))==colnames(na.omit(train_predictions)))){
    all_predictions = rbind(na.omit(test_predictions), na.omit(train_predictions))
  }
  if(all(colnames(na.omit(test_residuals))==colnames(na.omit(train_residuals)))){
    all_residuals = rbind(na.omit(test_residuals), na.omit(train_residuals))
  } 
  if(all(colnames(na.omit(test_resi_values))==colnames(na.omit(train_resi_values)))){
    all_resi_values = rbind(na.omit(test_resi_values), na.omit(train_resi_values))
  }
  # Return results with consistent sample ordering
  list(
    predictions = all_predictions,
    residuals = all_residuals,
    resi_value = all_resi_values
  )
}

# Main processing function
process_features <- function(input_file) {
  # Load data
  # r4g4_samples <- read.xlsx('./modelData/r4g4_samples.xlsx', sep = '\t')
  r4g4_samples <- read.table(str_c('./',input_file,'/info.csv'), sep = ',', head=TRUE)
  df_10kb <- read.table(str_c('./',input_file,'/all.',input_file,'.raw.tab'),
                       sep = '\t', 
                       header = TRUE, 
                       comment.char = '', 
                       check.names = FALSE)
  
  # Prepare data
  rownames(df_10kb) <- str_c('chr', df_10kb[,1], ':', df_10kb[,2], '-', df_10kb[,3])
  colnames(df_10kb) <- gsub('.uniq.nodup.bam', '', colnames(df_10kb))
  df_10kb <- df_10kb[, c('#chr','start','end',intersect(colnames(df_10kb), as.character(r4g4_samples$seqID)))]
  r4g4_samples <- r4g4_samples[which(r4g4_samples$seqID %in% intersect(colnames(df_10kb), as.character(r4g4_samples$seqID))), ]
  
  train_HD <- as.character(read.table(str_c('./modelData/',input_file,'.neg.ids.txt'),  # gsub('_gam|_gam1|_gam2','',input_file)
                                     sep = '\t', 
                                     header = FALSE)[,1])
  test_else <- setdiff(colnames(df_10kb), c(train_HD, c('#chr', 'start', 'end')))
  
  # Process features in batches
  all_features <- rownames(df_10kb)
  num_batches <- ceiling(length(all_features) / batch_size)
  
  for (batch_num in 1:num_batches) {
    start_idx <- (batch_num - 1) * batch_size + 1
    end_idx <- min(batch_num * batch_size, length(all_features))
    batch_features <- all_features[start_idx:end_idx]
    
    cat("Processing batch", batch_num, "of", num_batches, "...\n")
    batch_results <- process_batch(batch_features, df_10kb, train_HD, test_else, r4g4_samples)
    
    # Save batch results
    saveRDS(batch_results, file.path(output_dir, paste0("batch_", batch_num, ".rds")))
  }
  
  # Combine all batch results
  combine_results(output_dir, df_10kb, input_file)
}

# Function to combine batch results
combine_results <- function(output_dir, df_10kb, input_file) {
  batch_files <- list.files(output_dir, pattern = "batch_.*\\.rds", full.names = TRUE)
  
  # Initialize result lists
  all_predictions <- list()
  all_residuals <- list()
  all_resi_values <- list()
  
  # Load and combine batch results
  for (batch_file in batch_files) {
    batch_results <- readRDS(batch_file)
    all_predictions <- append(all_predictions, list(batch_results$predictions))
    all_residuals <- append(all_residuals, list(batch_results$residuals))
    all_resi_values <- append(all_resi_values, list(batch_results$resi_value))
  }
  # Combine results
  final_predictions <- do.call(cbind, all_predictions)
  final_residuals <- do.call(cbind, all_residuals)
  final_resi_values <- do.call(cbind, all_resi_values)
  
  # Write final results
  write_results(final_predictions, final_residuals, final_resi_values, df_10kb, input_file)
}

# Function to write final results
write_results <- function(predictions, residuals, resi_values, df_10kb, input_file) {
  # Write predictions with preserved sample IDs
  # 显示不一致的列名（特征名）检查
  features_in_df <- rownames(df_10kb)
  features_in_pred <- colnames(predictions)
  
  # 找出不一致的特征
  missing_in_pred <- setdiff(features_in_df, features_in_pred)
  missing_in_df <- setdiff(features_in_pred, features_in_df)
  
  if(length(missing_in_pred) > 0) {
    cat("以下特征在原始数据中存在但在预测结果中缺失:\n")
    print(head(missing_in_pred, 10))  # 只显示前10个以免输出太长
    if(length(missing_in_pred) > 10) cat("... (共", length(missing_in_pred), "个)\n")
  }
  
  if(length(missing_in_df) > 0) {
    cat("以下特征在预测结果中存在但在原始数据中缺失:\n")
    print(head(missing_in_df, 10))
    if(length(missing_in_df) > 10) cat("... (共", length(missing_in_df), "个)\n")
  }
  # 显示样本名检查
  samples_in_pred <- rownames(predictions)
  samples_in_df <- colnames(df_10kb)[4:ncol(df_10kb)]  # 前3列是chr,start,end

  missing_samples_pred <- setdiff(samples_in_df, samples_in_pred)
  missing_samples_df <- setdiff(samples_in_pred, samples_in_df)

  if(length(missing_samples_pred) > 0) {
    cat("以下样本在原始数据中存在但在预测结果中缺失:\n")
    print(head(missing_samples_pred, 10))
    if(length(missing_samples_pred) > 10) cat("... (共", length(missing_samples_pred), "个)\n")
  }

  if(length(missing_samples_df) > 0) {
    cat("以下样本在预测结果中存在但在原始数据中缺失:\n")
    print(head(missing_samples_df, 10))
    if(length(missing_samples_df) > 10) cat("... (共", length(missing_samples_df), "个)\n")
  }

  
  # all(rownames(df_10kb) == colnames(predictions))
  trans_2 <- cbind(df_10kb[, c('#chr','start','end')], t(predictions)[rownames(df_10kb), ])
  colnames(trans_2) <- c('#chr', 'start', 'end', rownames(predictions))
  write.table(trans_2, 
              file.path(output_dir, str_c('train_prediction.tab.', input_file)), 
              sep = '\t', 
              row.names = FALSE, 
              col.names = TRUE,
              quote = FALSE)
  
  # Write residuals
  trans_2 <- cbind(df_10kb[, c('#chr', 'start', 'end')], t(residuals)[rownames(df_10kb), ])
  write.table(trans_2, 
              file.path(output_dir, str_c('train_gam.tab.', input_file)), 
              sep = '\t', 
              row.names = FALSE, 
              quote = FALSE)
  
  # Write transformed values
  trans_3 <- cbind(df_10kb[, c('#chr', 'start', 'end')], t(resi_values)[rownames(df_10kb), ])
  write.table(trans_3, 
              file.path(output_dir, str_c('train_gam_trans.tab.', input_file)), 
              sep = '\t', 
              row.names = FALSE, 
              quote = FALSE)
}


# Main execution
args <- commandArgs(trailingOnly = TRUE)
input_file <- args[1]
output_dir <- str_c("./normalized_results/", as.character(input_file))

# Create output directory if it doesn't exist
if (!dir.exists(output_dir)) {
  dir.create(output_dir)
}

process_features(input_file)
elapsed_time <- Sys.time() - start_time
print(elapsed_time)

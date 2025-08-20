# =============================================================================
# HRV Pipeline Validation Analysis
# ICC and Bland-Altman Analysis for PhysioKit vs NeuroKit2
# =============================================================================

# Install required packages if not already installed
packages <- c("readxl", "irr", "BlandAltmanLeh", "ggplot2", "dplyr", "gridExtra", "corrplot")
new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

# Load libraries
library(readxl)
library(irr)
library(BlandAltmanLeh)
library(ggplot2)
library(dplyr)
library(gridExtra)
library(corrplot)

# =============================================================================
# 1. LOAD AND PREPARE DATA
# =============================================================================

# Load the Excel file
data <- read_excel("Final_Validity.xlsx")

# Display basic info
cat("Dataset loaded successfully!\n")
cat("Number of files analyzed:", nrow(data), "\n")
cat("Columns:", ncol(data), "\n\n")

# Check for any missing values
missing_summary <- sapply(data, function(x) sum(is.na(x)))
if(any(missing_summary > 0)) {
  cat("Missing values found:\n")
  print(missing_summary[missing_summary > 0])
} else {
  cat("No missing values found - excellent!\n\n")
}

# Check for any non-finite values (Inf, -Inf, NaN)
infinite_summary <- sapply(data, function(x) sum(!is.finite(x)))
if(any(infinite_summary > 0)) {
  cat("Non-finite values (Inf/NaN) found:\n")
  print(infinite_summary[infinite_summary > 0])
  cat("\n")
}

# =============================================================================
# 2. ICC ANALYSIS FOR ALL METRICS
# =============================================================================

cat("=== INTRACLASS CORRELATION COEFFICIENT (ICC) ANALYSIS ===\n\n")

# Define metric pairs for ICC analysis
metrics <- list(
  "Number of Peaks" = c("NumberPeaks_Neuro", "NumberPeaks_PhysioKit"),
  "Heart Rate (BPM)" = c("HR_Neuro", "HR_PhysioKit"),
  "Mean RR (ms)" = c("MeanRR_Neuro", "MeanRR_PhysioKit"),
  "RMSSD (ms)" = c("RMSSD_Neuro", "RMSSD_PhysioKit"),
  "pNN50 (%)" = c("pnn50_Neuro", "pnn50_PhysioKit"),
  "SDNN (ms)" = c("SDNN_Neuro", "SDNN_PhysioKIt"),  # Note: keeping your typo
  "SD1 (ms)" = c("SD1_Neuro", "SD1_PhysioKit"),
  "SD2 (ms)" = c("SD2_Neuro", "SD2_PhysioKit"),
  "SD1/SD2 Ratio" = c("SD1_SD2_Neuro", "SD1_SD2_PhysioKit"),
  "LF/HF Ratio" = c("LF_HF_Neuro", "LF_HF_PhysioKit"),
  "LF (n.u.)" = c("lf_nu_Neuro", "lf_nu_PhysioKit"),
  "HF (n.u.)" = c("hf_nu_neuro", "hf_nu_PhysioKit")
)

# Calculate ICC for each metric
icc_results <- data.frame(
  Metric = character(),
  ICC_Value = numeric(),
  Lower_CI = numeric(),
  Upper_CI = numeric(),
  Interpretation = character(),
  stringsAsFactors = FALSE
)

for(metric_name in names(metrics)) {
  col_names <- metrics[[metric_name]]
  
  # Check if both columns exist and have data
  if(all(col_names %in% names(data))) {
    metric_data <- data[, col_names]
    
    # Remove any rows with missing values for this metric pair
    complete_data <- metric_data[complete.cases(metric_data), ]
    
    if(nrow(complete_data) > 2) {  # Need at least 3 observations for ICC
      # Calculate ICC(3,1) - two-way mixed effects, absolute agreement, single measurement
      icc_result <- icc(complete_data, model = "twoway", type = "agreement", unit = "single")
      
      # Interpret ICC value with error handling
      icc_val <- icc_result$value
      
      # Check for NA or invalid values
      if(is.na(icc_val) || is.null(icc_val)) {
        interpretation <- "Unable to calculate"
        icc_val <- NA
      } else if(icc_val >= 0.90) {
        interpretation <- "Excellent"
      } else if(icc_val >= 0.75) {
        interpretation <- "Good"
      } else if(icc_val >= 0.50) {
        interpretation <- "Moderate"
      } else {
        interpretation <- "Poor"
      }
      
      # Store results with error handling
      icc_results <- rbind(icc_results, data.frame(
        Metric = metric_name,
        ICC_Value = ifelse(is.na(icc_val), NA, round(icc_val, 4)),
        Lower_CI = ifelse(is.na(icc_result$lbound), NA, round(icc_result$lbound, 4)),
        Upper_CI = ifelse(is.na(icc_result$ubound), NA, round(icc_result$ubound, 4)),
        Interpretation = interpretation
      ))
      
      # Print results with NA handling
      if(is.na(icc_val)) {
        cat(sprintf("%-20s ICC = NA (unable to calculate) - %s\n", 
                    metric_name, interpretation))
      } else {
        cat(sprintf("%-20s ICC = %.4f [%.4f, %.4f] - %s\n", 
                    metric_name, icc_val, icc_result$lbound, icc_result$ubound, interpretation))
      }
    } else {
      cat(sprintf("%-20s: Insufficient data for ICC calculation\n", metric_name))
    }
  } else {
    cat(sprintf("%-20s: Column(s) not found\n", metric_name))
  }
}

cat("\n")

# =============================================================================
# 3. CORRELATION ANALYSIS
# =============================================================================

cat("=== PEARSON CORRELATION ANALYSIS ===\n\n")

correlation_results <- data.frame(
  Metric = character(),
  Correlation = numeric(),
  P_Value = numeric(),
  stringsAsFactors = FALSE
)

for(metric_name in names(metrics)) {
  col_names <- metrics[[metric_name]]
  
  if(all(col_names %in% names(data))) {
    x <- data[[col_names[1]]]
    y <- data[[col_names[2]]]
    
    # Remove missing values
    complete_idx <- complete.cases(x, y)
    x_clean <- x[complete_idx]
    y_clean <- y[complete_idx]
    
    if(length(x_clean) > 2) {
      cor_test <- cor.test(x_clean, y_clean, method = "pearson")
      
      correlation_results <- rbind(correlation_results, data.frame(
        Metric = metric_name,
        Correlation = round(cor_test$estimate, 4),
        P_Value = round(cor_test$p.value, 6)
      ))
      
      cat(sprintf("%-20s r = %.4f, p = %.6f\n", 
                  metric_name, cor_test$estimate, cor_test$p.value))
    }
  }
}

cat("\n")

# =============================================================================
# 4. BLAND-ALTMAN ANALYSIS
# =============================================================================

cat("=== BLAND-ALTMAN ANALYSIS ===\n\n")

# Key metrics for Bland-Altman plots (core HRV metrics)
key_metrics <- list(
  "Heart Rate (BPM)" = c("HR_Neuro", "HR_PhysioKit"),
  "RMSSD (ms)" = c("RMSSD_Neuro", "RMSSD_PhysioKit"),
  "SDNN (ms)" = c("SDNN_Neuro", "SDNN_PhysioKIt"),
  "LF/HF Ratio" = c("LF_HF_Neuro", "LF_HF_PhysioKit")
)

# Create Bland-Altman plots
ba_plots <- list()
ba_stats <- data.frame(
  Metric = character(),
  Mean_Bias = numeric(),
  Lower_LoA = numeric(),
  Upper_LoA = numeric(),
  SD_Diff = numeric(),
  stringsAsFactors = FALSE
)

for(i in 1:length(key_metrics)) {
  metric_name <- names(key_metrics)[i]
  col_names <- key_metrics[[metric_name]]
  
  if(all(col_names %in% names(data))) {
    neuro_vals <- data[[col_names[1]]]
    physio_vals <- data[[col_names[2]]]
    
    # Remove missing values
    complete_idx <- complete.cases(neuro_vals, physio_vals)
    neuro_clean <- neuro_vals[complete_idx]
    physio_clean <- physio_vals[complete_idx]
    
    if(length(neuro_clean) > 2) {
      # Calculate Bland-Altman statistics
      mean_vals <- (neuro_clean + physio_clean) / 2
      diff_vals <- neuro_clean - physio_clean
      
      mean_bias <- mean(diff_vals)
      sd_diff <- sd(diff_vals)
      lower_loa <- mean_bias - 1.96 * sd_diff
      upper_loa <- mean_bias + 1.96 * sd_diff
      
      # Store statistics
      ba_stats <- rbind(ba_stats, data.frame(
        Metric = metric_name,
        Mean_Bias = round(mean_bias, 4),
        Lower_LoA = round(lower_loa, 4),
        Upper_LoA = round(upper_loa, 4),
        SD_Diff = round(sd_diff, 4)
      ))
      
      # Create Bland-Altman plot
      ba_data <- data.frame(
        Mean = mean_vals,
        Difference = diff_vals
      )
      
      p <- ggplot(ba_data, aes(x = Mean, y = Difference)) +
        geom_point(alpha = 0.6, size = 2) +
        geom_hline(yintercept = mean_bias, color = "blue", linetype = "solid", size = 1) +
        geom_hline(yintercept = lower_loa, color = "red", linetype = "dashed", size = 1) +
        geom_hline(yintercept = upper_loa, color = "red", linetype = "dashed", size = 1) +
        geom_hline(yintercept = 0, color = "black", linetype = "dotted", alpha = 0.5) +
        labs(
          title = paste("Bland-Altman Plot:", metric_name),
          subtitle = paste("Bias =", round(mean_bias, 3), 
                           "| LoA: [", round(lower_loa, 3), ",", round(upper_loa, 3), "]"),
          x = paste("Mean of NeuroKit2 and PhysioKit", metric_name),
          y = "NeuroKit2 - PhysioKit"
        ) +
        theme_minimal() +
        theme(
          plot.title = element_text(size = 12, face = "bold"),
          plot.subtitle = element_text(size = 10),
          axis.title = element_text(size = 10)
        )
      
      ba_plots[[i]] <- p
      
      cat(sprintf("%-20s: Bias = %7.4f, LoA = [%7.4f, %7.4f]\n", 
                  metric_name, mean_bias, lower_loa, upper_loa))
    }
  }
}

cat("\n")

# =============================================================================
# 5. SUMMARY STATISTICS
# =============================================================================

cat("=== VALIDATION SUMMARY ===\n\n")

# Count ICC interpretations
icc_summary <- table(icc_results$Interpretation)
cat("ICC Results Summary:\n")
for(interp in names(icc_summary)) {
  cat(sprintf("  %s: %d metrics\n", interp, icc_summary[interp]))
}

cat("\n")

# Overall assessment
excellent_count <- sum(icc_results$ICC_Value >= 0.90, na.rm = TRUE)
good_count <- sum(icc_results$ICC_Value >= 0.75 & icc_results$ICC_Value < 0.90, na.rm = TRUE)
total_metrics <- nrow(icc_results)

cat("Overall Validation Assessment:\n")
cat(sprintf("  Total metrics analyzed: %d\n", total_metrics))
cat(sprintf("  Excellent agreement (ICC ≥ 0.90): %d (%.1f%%)\n", 
            excellent_count, 100 * excellent_count / total_metrics))
cat(sprintf("  Good agreement (ICC ≥ 0.75): %d (%.1f%%)\n", 
            good_count, 100 * good_count / total_metrics))

# =============================================================================
# 6. SAVE RESULTS AND PLOTS
# =============================================================================

# Display the plots
if(length(ba_plots) > 0) {
  grid.arrange(grobs = ba_plots, ncol = 2)
}

# Save detailed results
write.csv(icc_results, "ICC_Results.csv", row.names = FALSE)
write.csv(ba_stats, "Bland_Altman_Statistics.csv", row.names = FALSE)
write.csv(correlation_results, "Correlation_Results.csv", row.names = FALSE)

cat("\nResults saved to:\n")
cat("  - ICC_Results.csv\n")
cat("  - Bland_Altman_Statistics.csv\n")
cat("  - Correlation_Results.csv\n")

# Save high-quality plots
ggsave("Bland_Altman_Plots.png", 
       arrangeGrob(grobs = ba_plots, ncol = 2), 
       width = 12, height = 10, dpi = 300)

cat("  - Bland_Altman_Plots.png\n\n")


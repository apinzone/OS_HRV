
# Install required packages
packages <- c("readxl", "irr", "BlandAltmanLeh", "ggplot2", "dplyr", "gridExtra", "DescTools", "extrafont")
new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

# Load libraries
library(readxl)
library(irr)
library(BlandAltmanLeh)
library(ggplot2)
library(dplyr)
library(gridExtra)
library(extrafont)
library(DescTools)

# Load data
file_path <- "C:/Users/Anthony/Desktop/peak_detector/Validation Study/all_validity_data.xlsx"
chronos_data <- read_excel(file_path, sheet = "ChronOS")
neurokit_data <- read_excel(file_path, sheet = "Neurokit")
truth_data <- read_excel(file_path, sheet = "Truth")
data <- merge(chronos_data, neurokit_data, by = "filename", all = TRUE)

# Define metrics
metrics <- list(
  "Number of R-peaks" = c("total_peaks", "nk_num_beats"),
  "Mean RR (ms)" = c("chronos_mean_rr_ms", "nk_mean_rr_ms"),
  "RMSSD (ms)" = c("chronos_rmssd_ms", "nk_rmssd_ms"),
  "pNN50 (%)" = c("chronos_pnn50_percent", "nk_pnn50_percent"),
  "SDNN (ms)" = c("chronos_sdnn_ms", "nk_sdnn_ms"),
  "SD1 (ms)" = c("chronos_sd1_ms", "nk_sd1_ms"),
  "SD2 (ms)" = c("chronos_sd2_ms", "nk_sd2_ms"),
  "SD1/SD2 Ratio" = c("chronos_sd1_sd2_ratio", "nk_sd1_sd2_ratio"),
  "Sample Entropy" = c("chronos_sampen", "nk_sampen"),
  "VLF (ms²)" = c("chronos_vlf_power_ms2", "nk_vlf_power_ms2"),
  "LF (ms²)" = c("chronos_lf_power_ms2", "nk_lf_power_ms2"),
  "HF (ms²)" = c("chronos_hf_power_ms2", "nk_hf_power_ms2"),
  "LF/HF Ratio" = c("chronos_lf_hf_ratio", "nk_lf_hf_ratio"),
  "Total Power (ms²)" = c("chronos_total_power_ms2", "nk_total_power_ms2")
)

# ICC Analysis
cat("=== INTRACLASS CORRELATION COEFFICIENT (ICC) ANALYSIS ===\n\n")

icc_results <- data.frame(Metric = character(), ICC_Value = numeric(), 
                          Lower_CI = numeric(), Upper_CI = numeric(), 
                          Interpretation = character(), stringsAsFactors = FALSE)

for(metric_name in names(metrics)) {
  col_names <- metrics[[metric_name]]
  if(all(col_names %in% names(data))) {
    metric_data <- data[, col_names]
    complete_data <- metric_data[complete.cases(metric_data), ]
    
    if(nrow(complete_data) > 2) {
      icc_result <- icc(complete_data, model = "twoway", type = "agreement", unit = "single")
      icc_val <- icc_result$value
      
      interpretation <- if(is.na(icc_val)) "Unable to calculate"
      else if(icc_val >= 0.90) "Excellent"
      else if(icc_val >= 0.75) "Good"
      else if(icc_val >= 0.50) "Moderate"
      else "Poor"
      
      icc_results <- rbind(icc_results, data.frame(
        Metric = metric_name,
        ICC_Value = ifelse(is.na(icc_val), NA, round(icc_val, 4)),
        Lower_CI = ifelse(is.na(icc_result$lbound), NA, round(icc_result$lbound, 4)),
        Upper_CI = ifelse(is.na(icc_result$ubound), NA, round(icc_result$ubound, 4)),
        Interpretation = interpretation
      ))
      
      if(is.na(icc_val)) {
        cat(sprintf("%-20s ICC = NA (unable to calculate) - %s\n", 
                    metric_name, interpretation))
      } else {
        cat(sprintf("%-20s ICC = %.4f [%.4f, %.4f] - %s\n", 
                    metric_name, icc_val, icc_result$lbound, icc_result$ubound, interpretation))
      }
    }
  }
}

cat("\n")

# Lin's CCC Analysis
cat("=== LIN'S CONCORDANCE CORRELATION COEFFICIENT ANALYSIS ===\n\n")

ccc_results <- data.frame(Metric = character(), CCC_Value = numeric(),
                          Lower_CI = numeric(), Upper_CI = numeric(), stringsAsFactors = FALSE)

for(metric_name in names(metrics)) {
  col_names <- metrics[[metric_name]]
  if(all(col_names %in% names(data))) {
    x <- data[[col_names[1]]]
    y <- data[[col_names[2]]]
    
    complete_idx <- complete.cases(x, y)
    x_clean <- x[complete_idx]
    y_clean <- y[complete_idx]
    
    if(length(x_clean) > 2) {
      ccc_result <- CCC(x_clean, y_clean, ci = "z-transform", conf.level = 0.95)
      
      ccc_results <- rbind(ccc_results, data.frame(
        Metric = metric_name,
        CCC_Value = round(ccc_result$rho.c[1], 4),
        Lower_CI = round(ccc_result$rho.c[2], 4),
        Upper_CI = round(ccc_result$rho.c[3], 4)
      ))
      
      cat(sprintf("%-20s CCC = %.4f [%.4f, %.4f]\n", 
                  metric_name, ccc_result$rho.c[1], ccc_result$rho.c[2], ccc_result$rho.c[3]))
    }
  }
}

cat("\n")

# Bland-Altman Analysis 
cat("=== BLAND-ALTMAN ANALYSIS ===\n\n")

# Plot 1 metrics (Time Domain)
plot1_metrics <- list(
  "Mean RR (ms)" = c("chronos_mean_rr_ms", "nk_mean_rr_ms"),
  "RMSSD (ms)" = c("chronos_rmssd_ms", "nk_rmssd_ms"),
  "SDNN (ms)" = c("chronos_sdnn_ms", "nk_sdnn_ms"),
  "Sample Entropy" = c("chronos_sampen", "nk_sampen")
)

ba_plots_1 <- list()
ba_stats <- data.frame(Metric = character(), Mean_Bias = numeric(),
                       Lower_LoA = numeric(), Upper_LoA = numeric(),
                       SD_Diff = numeric(), stringsAsFactors = FALSE)

for(i in 1:length(plot1_metrics)) {
  metric_name <- names(plot1_metrics)[i]
  col_names <- plot1_metrics[[metric_name]]
  
  if(all(col_names %in% names(data))) {
    chronos_vals <- data[[col_names[1]]]
    neurokit_vals <- data[[col_names[2]]]
    
    complete_idx <- complete.cases(chronos_vals, neurokit_vals)
    chronos_clean <- chronos_vals[complete_idx]
    neurokit_clean <- neurokit_vals[complete_idx]
    
    if(length(chronos_clean) > 2) {
      mean_vals <- (chronos_clean + neurokit_clean) / 2
      diff_vals <- chronos_clean - neurokit_clean
      
      mean_bias <- mean(diff_vals)
      sd_diff <- sd(diff_vals)
      lower_loa <- mean_bias - 1.96 * sd_diff
      upper_loa <- mean_bias + 1.96 * sd_diff
      
      ba_stats <- rbind(ba_stats, data.frame(
        Metric = metric_name,
        Mean_Bias = round(mean_bias, 4),
        Lower_LoA = round(lower_loa, 4),
        Upper_LoA = round(upper_loa, 4),
        SD_Diff = round(sd_diff, 4)
      ))
      
      ba_data <- data.frame(Mean = mean_vals, Difference = diff_vals)
      
      # Original styling with LARGER FONTS
      p <- ggplot(ba_data, aes(x = Mean, y = Difference)) +
        geom_point(alpha = 0.9, size = 2.2, color = "black", shape = 1, stroke = 0.8) +
        geom_hline(yintercept = 0, color = "gray60", linetype = "dotted", size = 0.8) +
        geom_hline(yintercept = mean_bias, color = "black", linetype = "solid", size = 1.2) +
        geom_hline(yintercept = lower_loa, color = "black", linetype = "dashed", size = 1) +
        geom_hline(yintercept = upper_loa, color = "black", linetype = "dashed", size = 1) +
        labs(
          title = paste0(LETTERS[i], ") ", metric_name),
          subtitle = sprintf("Bias = %.3f | LoA: [%.3f, %.3f]", mean_bias, lower_loa, upper_loa),
          x = paste("Mean of ChronOS and NeuroKit2", metric_name),
          y = "Interprogram Difference"
        ) +
        scale_x_continuous(expand = expansion(mult = c(0.05, 0.05))) +
        scale_y_continuous(expand = expansion(mult = c(0.05, 0.05))) +
        theme_classic() +
        theme(
          plot.title = element_text(size = 18, face = "bold", family = "serif"),     
          plot.subtitle = element_text(size = 16, family = "serif"),                 
          axis.title = element_text(size = 16, family = "serif"),                    
          axis.text = element_text(size = 14, family = "serif"),                    
          text = element_text(family = "serif")
        ) 
      
      ba_plots_1[[i]] <- p
      
      cat(sprintf("%-20s: Bias = %7.4f, LoA = [%7.4f, %7.4f]\n", 
                  metric_name, mean_bias, lower_loa, upper_loa))
    }
  }
}

# Plot 2 metrics (Frequency Domain)
plot2_metrics <- list(
  "VLF (ms²)" = c("chronos_vlf_power_ms2", "nk_vlf_power_ms2"),
  "LF (ms²)" = c("chronos_lf_power_ms2", "nk_lf_power_ms2"),
  "HF (ms²)" = c("chronos_hf_power_ms2", "nk_hf_power_ms2"),
  "LF/HF Ratio" = c("chronos_lf_hf_ratio", "nk_lf_hf_ratio")
)

ba_plots_2 <- list()

for(i in 1:length(plot2_metrics)) {
  metric_name <- names(plot2_metrics)[i]
  col_names <- plot2_metrics[[metric_name]]
  
  if(all(col_names %in% names(data))) {
    chronos_vals <- data[[col_names[1]]]
    neurokit_vals <- data[[col_names[2]]]
    
    complete_idx <- complete.cases(chronos_vals, neurokit_vals)
    chronos_clean <- chronos_vals[complete_idx]
    neurokit_clean <- neurokit_vals[complete_idx]
    
    if(length(chronos_clean) > 2) {
      mean_vals <- (chronos_clean + neurokit_clean) / 2
      diff_vals <- chronos_clean - neurokit_clean
      
      mean_bias <- mean(diff_vals)
      sd_diff <- sd(diff_vals)
      lower_loa <- mean_bias - 1.96 * sd_diff
      upper_loa <- mean_bias + 1.96 * sd_diff
      
      ba_stats <- rbind(ba_stats, data.frame(
        Metric = metric_name,
        Mean_Bias = round(mean_bias, 4),
        Lower_LoA = round(lower_loa, 4),
        Upper_LoA = round(upper_loa, 4),
        SD_Diff = round(sd_diff, 4)
      ))
      
      ba_data <- data.frame(Mean = mean_vals, Difference = diff_vals)
      
      p <- ggplot(ba_data, aes(x = Mean, y = Difference)) +
        geom_point(alpha = 0.9, size = 2.2, color = "black", shape = 1, stroke = 0.8) +
        geom_hline(yintercept = 0, color = "gray60", linetype = "dotted", size = 0.8) +
        geom_hline(yintercept = mean_bias, color = "black", linetype = "solid", size = 1.2) +
        geom_hline(yintercept = lower_loa, color = "black", linetype = "dashed", size = 1) +
        geom_hline(yintercept = upper_loa, color = "black", linetype = "dashed", size = 1) +
        labs(
          title = paste0(LETTERS[i], ") ", metric_name),
          subtitle = sprintf("Bias = %.3f | LoA: [%.3f, %.3f]", mean_bias, lower_loa, upper_loa),
          x = paste("Mean of ChronOS and NeuroKit2", metric_name),
          y = "Interprogram Difference"
        ) +
        scale_x_continuous(expand = expansion(mult = c(0.05, 0.05))) +
        scale_y_continuous(expand = expansion(mult = c(0.05, 0.05))) +
        theme_classic() +
        theme(
          plot.title = element_text(size = 18, face = "bold", family = "serif"),      
          plot.subtitle = element_text(size = 16, family = "serif"),             
          axis.title = element_text(size = 16, family = "serif"),                    
          axis.text = element_text(size = 14, family = "serif"),                   
          text = element_text(family = "serif")
        ) 
      
      ba_plots_2[[i]] <- p
      
      cat(sprintf("%-20s: Bias = %7.4f, LoA = [%7.4f, %7.4f]\n", 
                  metric_name, mean_bias, lower_loa, upper_loa))
    }
  }
}

cat("\n")


ggsave("Bland_Altman_Plot1_Word.png", 
       arrangeGrob(grobs = ba_plots_1, ncol = 2), 
       width = 12, height = 10, 
       dpi = 300,          
       bg = "white",
       units = "in",
       type = "cairo")      

ggsave("Bland_Altman_Plot2_Word.png", 
       arrangeGrob(grobs = ba_plots_2, ncol = 2), 
       width = 12, height = 10, 
       dpi = 300, 
       bg = "white",
       units = "in",
       type = "cairo")

# Also save as TIFF if needed for publication
ggsave("Bland_Altman_Plot1_Publication.tiff", 
       arrangeGrob(grobs = ba_plots_1, ncol = 2), 
       width = 12, height = 10, 
       dpi = 300,           
       compression = "lzw", 
       bg = "white",
       units = "in")        

ggsave("Bland_Altman_Plot2_Publication.tiff", 
       arrangeGrob(grobs = ba_plots_2, ncol = 2), 
       width = 12, height = 10, 
       dpi = 300,           
       compression = "lzw", 
       bg = "white",
       units = "in")        

# Ground Truth Peak Detection Validation
cat("=== R-PEAK DETECTION VALIDATION AGAINST GROUND TRUTH ===\n\n")

calculate_peak_metrics <- function(detected_peaks, true_peaks, tolerance_ms = 50) {
  detected_peaks <- as.numeric(detected_peaks[!is.na(detected_peaks)])
  true_peaks <- as.numeric(true_peaks[!is.na(true_peaks)])
  detected_peaks <- sort(detected_peaks)
  true_peaks <- sort(true_peaks)
  
  true_positives <- 0
  false_positives <- 0
  matched_true_peaks <- rep(FALSE, length(true_peaks))
  
  for (detected_peak in detected_peaks) {
    differences <- abs(true_peaks - detected_peak)
    within_tolerance <- which(differences <= tolerance_ms & !matched_true_peaks)
    
    if (length(within_tolerance) > 0) {
      closest_idx <- within_tolerance[which.min(differences[within_tolerance])]
      true_positives <- true_positives + 1
      matched_true_peaks[closest_idx] <- TRUE
    } else {
      false_positives <- false_positives + 1
    }
  }
  
  false_negatives <- sum(!matched_true_peaks)
  sensitivity <- ifelse(true_positives + false_negatives > 0, 
                        true_positives / (true_positives + false_negatives), 0)
  ppv <- ifelse(true_positives + false_positives > 0, 
                true_positives / (true_positives + false_positives), 0)
  f1_score <- ifelse(sensitivity + ppv > 0, 
                     2 * (sensitivity * ppv) / (sensitivity + ppv), 0)
  
  return(list(
    true_positives = true_positives,
    false_positives = false_positives,
    false_negatives = false_negatives,
    sensitivity = sensitivity,
    ppv = ppv,
    f1_score = f1_score,
    total_detected = length(detected_peaks),
    total_true = length(true_peaks)
  ))
}

cat("ChronOS vs Ground Truth:\n")
chronos_peaks <- truth_data$chronos_time_ms
ground_truth_peaks <- truth_data$fantasia_time_ms

chronos_metrics <- calculate_peak_metrics(chronos_peaks, ground_truth_peaks, tolerance_ms = 50)

cat(sprintf("  True Positives: %d\n", chronos_metrics$true_positives))
cat(sprintf("  False Positives: %d\n", chronos_metrics$false_positives))
cat(sprintf("  False Negatives: %d\n", chronos_metrics$false_negatives))
cat(sprintf("  Total Detected: %d\n", chronos_metrics$total_detected))
cat(sprintf("  Total True: %d\n", chronos_metrics$total_true))
cat(sprintf("  Sensitivity: %.4f (%.2f%%)\n", chronos_metrics$sensitivity, chronos_metrics$sensitivity * 100))
cat(sprintf("  PPV: %.4f (%.2f%%)\n", chronos_metrics$ppv, chronos_metrics$ppv * 100))
cat(sprintf("  F1 Score: %.4f\n", chronos_metrics$f1_score))

cat("\n=== PEAK DETECTION SUMMARY ===\n\n")
cat("Tolerance: ±50 ms\n")
cat(sprintf("ChronOS Performance:\n"))
cat(sprintf("  Sensitivity: %.2f%% (ability to detect true peaks)\n", chronos_metrics$sensitivity * 100))
cat(sprintf("  PPV: %.2f%% (proportion of detected peaks that are true)\n", chronos_metrics$ppv * 100))
cat(sprintf("  F1 Score: %.4f (overall performance)\n", chronos_metrics$f1_score))

# Save results
write.csv(icc_results, "ICC_Results.csv", row.names = FALSE)
write.csv(ccc_results, "CCC_Results.csv", row.names = FALSE)
write.csv(ba_stats, "Bland_Altman_Statistics.csv", row.names = FALSE)

peak_validation_results <- data.frame(
  Method = "ChronOS",
  True_Positives = chronos_metrics$true_positives,
  False_Positives = chronos_metrics$false_positives,
  False_Negatives = chronos_metrics$false_negatives,
  Total_Detected = chronos_metrics$total_detected,
  Total_True = chronos_metrics$total_true,
  Sensitivity = round(chronos_metrics$sensitivity, 4),
  PPV = round(chronos_metrics$ppv, 4),
  F1_Score = round(chronos_metrics$f1_score, 4),
  stringsAsFactors = FALSE
)

write.csv(peak_validation_results, "Peak_Detection_Validation.csv", row.names = FALSE)

cat("\nResults saved:\n")
cat("  - Bland_Altman_Plot1_Word.png (optimized for Word)\n")
cat("  - Bland_Altman_Plot2_Word.png (optimized for Word)\n")
cat("  - Bland_Altman_Plot1_Publication.tiff (publication quality)\n")
cat("  - Bland_Altman_Plot2_Publication.tiff (publication quality)\n")
cat("  - ICC_Results.csv\n")
cat("  - CCC_Results.csv\n")
cat("  - Bland_Altman_Statistics.csv\n")
cat("  - Peak_Detection_Validation.csv\n")
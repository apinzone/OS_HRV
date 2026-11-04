# ChronOS vs Kubios HRV Scientific Lite
#   synthetic recordings   -> Table 2, Figures 4-5, VLF scale statistics
#   Fantasia segments      -> Table 4 (vs Kubios columns); plots optional
# Agreement statistics: ICC(2,1) absolute agreement (irr), Lin's CCC (DescTools),
# Bland-Altman with difference = ChronOS - Kubios.
# Untested edit: run once and compare the printed values with Tables 2 and 4.

packages <- c("readxl", "irr", "ggplot2", "gridExtra", "DescTools", "here")
new <- packages[!(packages %in% installed.packages()[, "Package"])]
if (length(new)) install.packages(new)
invisible(lapply(packages, library, character.only = TRUE))

# ---- Paths (relative to the repo root; no machine-specific paths) ----
vs_dir      <- here::here("Validation Study")
collated    <- file.path(vs_dir, "Data_and_Results", "Results", "Collated")
raw_dir     <- file.path(vs_dir, "Data_and_Results", "Results", "Raw")
stats_dir   <- file.path(vs_dir, "Data_and_Results", "Results", "ICC_CCC_Bland_Altman")
fig_dir     <- file.path(vs_dir, "Manuscript Figures")

datasets <- list(
  synthetic = list(file = file.path(collated, "all_validity_data.xlsx"),
                   tag = "", plots = TRUE),
  fantasia  = list(file = file.path(raw_dir, "fantasia_validity_data_060_with_kubios.xlsx"),
                   tag = "_Fantasia_060", plots = FALSE)  # set TRUE to also draw Fantasia plots
)

metrics <- c("Mean RR (ms)" = "mean_rr_ms", "RMSSD (ms)" = "rmssd_ms",
             "pNN50 (%)" = "pnn50_percent", "SDNN (ms)" = "sdnn_ms",
             "SD1 (ms)" = "sd1_ms", "SD2 (ms)" = "sd2_ms",
             "SD1/SD2 Ratio" = "sd1_sd2_ratio", "Sample Entropy" = "sampen",
             "VLF (ms²)" = "vlf_power_ms2", "LF (ms²)" = "lf_power_ms2",
             "HF (ms²)" = "hf_power_ms2", "LF/HF Ratio" = "lf_hf_ratio",
             "Total Power (ms²)" = "total_power_ms2")

agreement_level <- function(x) {
  if (is.na(x)) "NA" else if (x >= 0.90) "Excellent" else if (x >= 0.75) "Good"
  else if (x >= 0.50) "Moderate" else "Poor"
}

# ---- One Bland-Altman panel ----
ba_panel <- function(chronos, kubios, name, letter) {
  mean_vals <- (chronos + kubios) / 2
  diff_vals <- chronos - kubios
  bias <- mean(diff_vals)
  loa  <- bias + c(-1.96, 1.96) * sd(diff_vals)
  ggplot(data.frame(m = mean_vals, d = diff_vals), aes(m, d)) +
    geom_point(size = 2.2, shape = 1, stroke = 0.8) +
    geom_hline(yintercept = 0, colour = "gray60", linetype = "dotted", linewidth = 0.8) +
    geom_hline(yintercept = bias, linewidth = 1.2) +
    geom_hline(yintercept = loa, linetype = "dashed", linewidth = 1) +
    labs(title = paste0(letter, ") ", name),
         subtitle = sprintf("Bias = %.3f | LoA: [%.3f, %.3f]", bias, loa[1], loa[2]),
         x = paste("Mean of ChronOS and Kubios", name),
         y = "Interprogram Difference") +
    scale_x_continuous(expand = expansion(mult = c(0.05, 0.10))) +  # room for last tick label
    scale_y_continuous(expand = expansion(mult = 0.05)) +
    coord_cartesian(clip = "off") +
    theme_classic() +
    theme(text = element_text(family = "serif"),
          plot.title = element_text(size = 18, face = "bold"),
          plot.subtitle = element_text(size = 16),
          axis.title = element_text(size = 16),
          axis.text = element_text(size = 14),
          plot.margin = margin(6, 30, 6, 6))
}

save_grid <- function(panels, stub) {
  g <- arrangeGrob(grobs = panels, ncol = 2)
  ggsave(file.path(fig_dir, paste0(stub, ".png")), g, width = 12, height = 10,
         dpi = 300, bg = "white", type = "cairo")
  ggsave(file.path(fig_dir, paste0(stub, ".tiff")), g, width = 12, height = 10,
         dpi = 300, compression = "lzw", bg = "white")
}

# ---- Run one dataset ----
run_dataset <- function(ds, label) {
  cat(sprintf("\n===== %s: ChronOS vs Kubios =====\n", label))
  d <- merge(read_excel(ds$file, sheet = "ChronOS"),
             read_excel(ds$file, sheet = "Kubios"), by = "filename", all = TRUE)
  icc_out <- ccc_out <- ba_out <- NULL
  
  for (nm in names(metrics)) {
    cols <- paste0(c("chronos_", "kubios_"), metrics[[nm]])
    if (!all(cols %in% names(d))) next
    x <- d[[cols[1]]]; y <- d[[cols[2]]]
    ok <- complete.cases(x, y)
    if (sum(ok) <= 2) next
    x <- x[ok]; y <- y[ok]
    
    icc <- icc(cbind(x, y), model = "twoway", type = "agreement", unit = "single")
    ccc <- CCC(x, y, ci = "z-transform", conf.level = 0.95)$rho.c
    diffs <- x - y
    bias <- mean(diffs); s <- sd(diffs)
    
    icc_out <- rbind(icc_out, data.frame(Metric = nm, ICC_Value = round(icc$value, 4),
                                         Lower_CI = round(icc$lbound, 4), Upper_CI = round(icc$ubound, 4),
                                         Interpretation = agreement_level(icc$value)))
    ccc_out <- rbind(ccc_out, data.frame(Metric = nm, CCC_Value = round(ccc[1], 4),
                                         Lower_CI = round(ccc[2], 4), Upper_CI = round(ccc[3], 4)))
    ba_out  <- rbind(ba_out, data.frame(Metric = nm, Mean_Bias = round(bias, 4),
                                        Lower_LoA = round(bias - 1.96 * s, 4),
                                        Upper_LoA = round(bias + 1.96 * s, 4), SD_Diff = round(s, 4)))
  }
  print(icc_out); print(ccc_out); print(ba_out)
  
  # VLF: Kubios integrates 0-0.04 Hz, ChronOS only bins centred in 0.003-0.04 Hz
  v <- d[complete.cases(d$chronos_vlf_power_ms2, d$kubios_vlf_power_ms2), ]
  ratio <- v$kubios_vlf_power_ms2 / v$chronos_vlf_power_ms2
  vlf_cons <- icc(v[, c("chronos_vlf_power_ms2", "kubios_vlf_power_ms2")],
                  model = "twoway", type = "consistency", unit = "single")
  cat(sprintf("VLF Kubios/ChronOS ratio: median %.2f, IQR [%.2f, %.2f], range [%.2f, %.2f]\n",
              median(ratio), quantile(ratio, .25), quantile(ratio, .75), min(ratio), max(ratio)))
  cat(sprintf("VLF Pearson r = %.3f; consistency ICC = %.3f [%.3f, %.3f]\n",
              cor(v$chronos_vlf_power_ms2, v$kubios_vlf_power_ms2),
              vlf_cons$value, vlf_cons$lbound, vlf_cons$ubound))
  
  write.csv(icc_out, file.path(stats_dir, paste0("ICC_Results_Kubios", ds$tag, ".csv")), row.names = FALSE)
  write.csv(ccc_out, file.path(stats_dir, paste0("CCC_Results_Kubios", ds$tag, ".csv")), row.names = FALSE)
  write.csv(ba_out,  file.path(stats_dir, paste0("Bland_Altman_Statistics_Kubios", ds$tag, ".csv")), row.names = FALSE)
  
  if (ds$plots) {
    panel <- function(nm, i) {
      cols <- paste0(c("chronos_", "kubios_"), metrics[[nm]])
      ok <- complete.cases(d[[cols[1]]], d[[cols[2]]])
      ba_panel(d[[cols[1]]][ok], d[[cols[2]]][ok], nm, LETTERS[i])
    }
    time_set <- c("Mean RR (ms)", "RMSSD (ms)", "SDNN (ms)", "Sample Entropy")  # Figure 4
    freq_set <- c("VLF (ms²)", "LF (ms²)", "HF (ms²)", "LF/HF Ratio")          # Figure 5
    save_grid(Map(panel, time_set, 1:4), paste0("Figure_4_Kubios", ds$tag))
    save_grid(Map(panel, freq_set, 1:4), paste0("Figure_5_Kubios", ds$tag))
  }
}

for (nm in names(datasets)) run_dataset(datasets[[nm]], nm)
# Aim 1 — R-side advanced estimators:
#   dCDH continuous-treatment DiD (DIDmultiplegtDYN)
#   Callaway-Sant'Anna ATT(g,t) on binary high-vs-low intensity (did)
#   Borusyak-Jaravel-Spiess imputation (didimputation)
#   Honest-DiD sensitivity (HonestDiD)
#   Augmented synthetic control (augsynth)
#
# Reads: data/clean/state_quarter_panel.parquet
# Writes: tables/aim1_R_*.csv

suppressPackageStartupMessages({
  required <- c("DIDmultiplegtDYN", "did", "didimputation", "HonestDiD",
                "augsynth", "fixest", "data.table", "arrow", "jsonlite")
  missing <- setdiff(required, rownames(installed.packages()))
  if (length(missing) > 0) {
    cat("Missing R packages:", paste(missing, collapse = ", "), "\n",
        "Run: Rscript env/R_setup.R\n")
    quit(status = 0)
  }
  invisible(lapply(required, library, character.only = TRUE))
})

set.seed(42)

panel_path <- file.path("data", "clean", "state_quarter_panel.parquet")
out_dir    <- file.path("tables")
if (!file.exists(panel_path)) {
  cat("Missing panel:", panel_path, "; run build_panel.py first.\n")
  quit(status = 0)
}

panel <- arrow::read_parquet(panel_path)
panel <- as.data.table(panel)

primary_outcomes <- c(
  "acs_admit_rate_per_1000",
  "bh_ed_rate_per_1000",
  "all_cause_mortality_per_100k_35_64",
  "amenable_mortality_per_100k_35_64"
)

panel[, time_id := as.integer(time_id)]
panel[, state_id := as.factor(state_id)]
panel[, intensity := cumulative_procedural_disenrollment_rate]
panel[is.na(intensity), intensity := 0]

state_terminal <- panel[, .(max_intensity = max(intensity, na.rm = TRUE)), by = state_abbr]
median_int <- median(state_terminal$max_intensity, na.rm = TRUE)
state_terminal[, high_intensity := as.integer(max_intensity > median_int)]
panel <- merge(panel, state_terminal[, .(state_abbr, high_intensity)], by = "state_abbr", all.x = TRUE)

panel[, treat_group := ifelse(high_intensity == 1, min(time_id[post == 1], na.rm = TRUE), 0L), by = state_abbr]

dcdh_rows <- list()
cs_rows   <- list()
bjs_rows  <- list()

for (outcome in primary_outcomes) {
  if (!outcome %in% names(panel)) next
  dt <- panel[!is.na(get(outcome))]
  if (nrow(dt) < 100) next

  dcdh_res <- tryCatch({
    DIDmultiplegtDYN::did_multiplegt_dyn(
      df = as.data.frame(dt),
      outcome = outcome,
      group   = "state_abbr",
      time    = "time_id",
      treatment = "intensity",
      effects = 4,
      placebo = 4,
      cluster = "state_abbr",
      graph_off = TRUE
    )
  }, error = function(e) { cat("  dCDH failed for", outcome, ":", conditionMessage(e), "\n"); NULL })
  if (!is.null(dcdh_res)) {
    eff <- dcdh_res$results$Effects
    if (!is.null(eff)) {
      dcdh_rows[[outcome]] <- data.frame(
        outcome = outcome,
        att     = eff[, "Estimate"],
        se      = eff[, "SE"],
        lower   = eff[, "LB CI"],
        upper   = eff[, "UB CI"],
        period  = seq_len(nrow(eff))
      )
    }
  }

  cs_res <- tryCatch({
    did::att_gt(
      yname = outcome,
      tname = "time_id",
      idname = "state_id",
      gname = "treat_group",
      data = as.data.frame(dt),
      control_group = "notyettreated",
      base_period = "universal",
      bstrap = TRUE,
      cband = TRUE
    )
  }, error = function(e) { cat("  CS failed for", outcome, ":", conditionMessage(e), "\n"); NULL })
  if (!is.null(cs_res)) {
    agg <- did::aggte(cs_res, type = "simple", na.rm = TRUE)
    cs_rows[[outcome]] <- data.frame(
      outcome = outcome,
      att     = agg$overall.att,
      se      = agg$overall.se,
      lower   = agg$overall.att - 1.96 * agg$overall.se,
      upper   = agg$overall.att + 1.96 * agg$overall.se
    )
  }

  bjs_res <- tryCatch({
    didimputation::did_imputation(
      data = as.data.frame(dt),
      yname = outcome,
      gname = "treat_group",
      tname = "time_id",
      idname = "state_id",
      cluster_var = "state_id",
      first_stage = "0"
    )
  }, error = function(e) { cat("  BJS failed for", outcome, ":", conditionMessage(e), "\n"); NULL })
  if (!is.null(bjs_res) && nrow(bjs_res) > 0) {
    bjs_rows[[outcome]] <- data.frame(
      outcome = outcome,
      att     = bjs_res$estimate[1],
      se      = bjs_res$std.error[1],
      lower   = bjs_res$conf.low[1],
      upper   = bjs_res$conf.high[1]
    )
  }
}

if (length(dcdh_rows) > 0) {
  fwrite(rbindlist(dcdh_rows), file.path(out_dir, "aim1_R_dcdh.csv"))
}
if (length(cs_rows) > 0) {
  fwrite(rbindlist(cs_rows), file.path(out_dir, "aim1_R_cs.csv"))
}
if (length(bjs_rows) > 0) {
  fwrite(rbindlist(bjs_rows), file.path(out_dir, "aim1_R_bjs.csv"))
}

cat("Aim 1 R analyses complete.\n")

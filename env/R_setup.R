# R packages required for advanced estimators.
# Run once: Rscript env/R_setup.R

required <- c(
  "DIDmultiplegtDYN",   # de Chaisemartin-D'Haultfoeuille continuous-treatment DiD
  "did",                # Callaway-Sant'Anna ATT(g,t)
  "didimputation",      # Borusyak-Jaravel-Spiess imputation
  "HonestDiD",          # Rambachan-Roth sensitivity
  "augsynth",           # Augmented synthetic control
  "fixest",             # Fast fixed-effects regression
  "tidyverse",
  "data.table",
  "jsonlite",
  "arrow"
)

installed <- rownames(installed.packages())
to_install <- setdiff(required, installed)
if (length(to_install) > 0) {
  install.packages(to_install, repos = "https://cloud.r-project.org")
}

# augsynth and HonestDiD may need GitHub installation
if (!"augsynth" %in% rownames(installed.packages())) {
  if (!"remotes" %in% rownames(installed.packages())) {
    install.packages("remotes", repos = "https://cloud.r-project.org")
  }
  remotes::install_github("ebenmichael/augsynth")
}
if (!"HonestDiD" %in% rownames(installed.packages())) {
  remotes::install_github("asheshrambachan/HonestDiD")
}

cat("R environment ready.\n")

# Protocol: Distribution-Shift Stress Test of an Equity-Designed Medicaid Risk Model under the 2023–2024 Unwinding, with Individual-Level Causal Effects of Procedural Disenrollment on Chronic-Disease Outcomes

**Authors.** Sanjay Basu MD PhD<sup>1,2</sup>; Sadiq Y. Patel PhD<sup>2,3</sup>; Aaron Baum MD PhD<sup>2,4</sup>
**Slug:** `medicaid-unwinding-chronic-disease-fairness`
**Date locked:** 2026-05-08
**Reporting guidelines:** STROBE + RECORD (causal aim); TRIPOD-AI + PROBAST (Signal stress-test); CONSORT-AI guidance referenced where applicable
**IRB.** Existing Waymark IRB-equivalent / Privacy and Compliance determination covering the Patel-Baum-Basu 2024 *Sci Rep* analyses extends to this follow-up; documented in repository.
**Pre-registration target:** OSF before any analysis is run; protocol committed in the public repo at this path.

This protocol is finalized prior to data acquisition and analysis. Deviations after this date are documented in the manuscript with justification.

---

## 1. Background

The 2023–2024 Medicaid "unwinding" of pandemic-era continuous-coverage protections produced the largest insurance-coverage disruption in United States history, with approximately 25 million enrollees losing Medicaid coverage and approximately 70 percent of disenrollments classified as procedural (paperwork-driven) rather than for ineligibility. State-level variation in procedural-disenrollment intensity was substantial.

A Medicaid risk-stratification model named Signal was published by Patel, Baum, and Basu in *Scientific Reports* in 2024 (DOI 10.1038/s41598-023-51114-z). Signal is a two-stage XGBoost model that predicts non-emergent acute care utilization (emergency department visits and hospital admissions for non-emergent conditions) and total cost from Medicaid eligibility, inpatient, long-term care, other-services, and pharmacy claims plus social-risk features. The 2024 paper reported that Signal tripled the sensitivity of identifying high-risk patients (11.3% vs 3.4%) at near-identical specificity (99.8% vs 99.5%) and produced an approximately tenfold improvement in cost-prediction $R^2$ (0.195–0.412 vs 0.022–0.050) across population subgroups, relative to a standard cost-based comparator. The 2024 paper also reported that Signal **reversed** the lower sensitivity of risk prediction for Black versus White patients that was present in the standard cost-based model.

The 2023–2024 unwinding constitutes a real-world distribution shift in the Medicaid population. Whether the equity advantage of an equity-designed model persists under such a shift is unresolved. Concurrently, the individual-level effect of procedural disenrollment on chronic-disease outcomes among Medicaid members has not been rigorously estimated — only three peer-reviewed empirical studies addressed health outcomes in the 18 months following April 2023, and none used target-trial-emulation with individual-level claims data.

A companion paper (Basu and Berkowitz, 2026) demonstrates a related under-documentation mechanism for state Section 1115 frailty-determination algorithms during community-engagement-requirement implementation. The present study extends the inquiry to the risk-stratification layer using the unwinding as a natural distribution-shift stress test, and adds the individual-level causal estimate of disenrollment effects on chronic-disease outcomes.

## 2. Aims and hypotheses

### Aim 1 — Individual-level causal effect of procedural disenrollment on chronic-disease outcomes

**H1a (primary).** Among Waymark Medicaid members enrolled at the start of the 2023–2024 unwinding window, procedural disenrollment is associated with worse 6- and 12-month chronic-disease outcomes (rising HbA1c among diabetics, rising blood-pressure measurements among hypertensives, rising ambulatory-care-sensitive admission rate, rising emergency-department visit rate, rising all-cause mortality) than retention.

**H1b (secondary).** The estimated effect in H1a is heterogeneous across race/ethnicity, language, disability status, urbanicity, and baseline Signal-score quintile.

**H1c (secondary).** The state-level administrative-processing-intensity instrument (from KFF) yields a local average treatment effect that is consistent in sign and within an order of magnitude of the propensity-score-adjusted ATE.

### Aim 2 — Signal stress-test under unwinding-induced distribution shift

**H2a (primary).** The calibration of the published Signal model (intercept, slope, expected calibration error) drifts across three windows — pre-unwinding (2018–2022), unwinding (2023-Q2 to 2024-Q2), and post-unwinding (2024-Q3 onward) — when applied to the same Waymark Medicaid cohort.

**H2b (primary).** The race-stratified equity advantage of Signal over the standard cost-based model — reported in Patel-Baum-Basu 2024 — persists post-unwinding as measured by sensitivity, specificity, calibration parity, equal opportunity, predictive parity, and demographic parity at the published top-decile-risk threshold.

**H2c (primary).** The relative performance of Signal versus HHS-HCC v07 and CDPS+Rx (public comparators applied to the same Waymark cohort) is stable across the three windows.

**H2d (secondary).** Subgroup performance drift is concentrated among members whose chronic-disease, language, and SDOH covariate distributions shifted most between pre-unwinding and post-unwinding windows.

### Aim 3 — Synthesis: do Signal predictions miss something disenrollment-vulnerable populations need most?

**H3 (primary).** Among Waymark Medicaid members with similar baseline Signal predicted-outcome scores, members who were procedurally disenrolled during the unwinding had systematically worse actual outcomes than retained members.

**H3 mediation (secondary).** Mediation analysis decomposes the disenrollment-outcome association (Aim 1) into the share mediated by Signal miscalibration (Aim 2) versus residual disenrollment effect.

## 3. Data sources

| Source | Vintage | Use | Public? |
|---|---|---|---|
| Waymark Medicaid claims/EHR via Vault → lighthouse / coredb / dbt_tuva_core | 2018-Q1 through latest available | Aim 1 cohort, Aim 2 cohort, Aim 3 cohort | No (internal); IRB-equivalent determination on file |
| Patel-Baum-Basu 2024 Signal public code (https://github.com/sadiqypatel/Medicaid_Risk_Model) | 2024 release | Audited model artifacts (Stage 1 + Stage 2, all-cause and non-emergent variants) | Yes |
| HHS-HCC v07 public model artifacts (CMS) | Benefit Year 2024 | Public comparator | Yes |
| CDPS+Rx v7 public model artifacts (UCSD) | v7 | Public comparator | Yes |
| KFF Medicaid Enrollment & Unwinding Tracker | Jan 2023 – Dec 2024 | State-level administrative-processing-intensity instrument; supplemental state-by-demographic disenrollment | Yes |
| CDC WONDER mortality | 2018–2024 | External-mortality validation if NDI linkage in Waymark is incomplete | Yes |
| Census ACS state-year covariates | 2018–2024 | State-level covariates | Yes |

No raw or de-identified patient data are committed to GitHub. Code and synthetic fixtures are public.

## 4. Sample and inclusion criteria

### Aim 1 sample (causal aim)

Unit of analysis: Waymark Medicaid member with continuous coverage during a baseline window of January–March 2023.

Index date: April 1, 2023 (start of unwinding administrative process).

Inclusion: aged 19–64 at index; any month of Medicaid coverage in 2023-Q1; non-missing race/ethnicity, age, sex, geography (state and ZIP-3), and at least one observed claim during 2018-Q1 through 2023-Q1.

Exclusion: dual-eligible Medicare-Medicaid members at index (separated into pre-specified sensitivity subsample); pregnancy-only coverage at index (separated into sensitivity subsample); end-stage renal disease at index (different risk-adjustment tracks).

Treatment assignment: Procedural disenrollment any time during 2023-Q2 through 2024-Q2 (treatment) versus continuous Medicaid coverage during the same window (comparator).

Follow-up: 6 months and 12 months post-index for chronic-disease outcomes.

### Aim 2 sample (Signal stress-test)

Same Waymark Medicaid panel, but partitioned into three time windows for application of Signal:
- Pre-unwinding window: 2018-Q1 through 2022-Q4
- Unwinding window: 2023-Q2 through 2024-Q2
- Post-unwinding window: 2024-Q3 onward

Within each window, members with a complete one-year lookback for feature construction and a one-year lookahead for outcome observation are eligible.

### Aim 3 sample

Intersection of Aim 1 and Aim 2 samples.

## 5. Variables

### Aim 1

**Treatment.** Indicator for procedural disenrollment between 2023-Q2 and 2024-Q2 versus continuous coverage. Coding from Waymark coverage spans plus state-MCO-linked disenrollment-reason codes.

**Outcomes (primary).**
1. HbA1c change (mg/dL) at 6 and 12 months among members with diagnosed diabetes at index
2. Systolic blood pressure change (mmHg) at 6 and 12 months among members with diagnosed hypertension at index
3. Ambulatory-care-sensitive admission rate per member-year, 12-month follow-up (AHRQ PQI chronic-care set)
4. Emergency-department visit rate per member-year, 12-month follow-up
5. All-cause mortality at 12 months (Waymark internal mortality flag; CDC WONDER cross-validation if needed)

**Confounders for adjustment.** Demographics (age, sex, race/ethnicity, language, geography, urbanicity, ZIP-3 ADI / SVI / ICE); baseline chronic conditions (CCS-mapped from ICD-10); baseline 12-month utilization (PCP visits, ED visits, admissions, prescription fills, specialist visits); baseline Signal score; MCO assignment; state; prior coverage continuity; pregnancy status; SUD diagnosis; major mental-health diagnosis; HCBS use.

**Effect modifiers (subgroups).** Race/ethnicity (NH-White, NH-Black, Hispanic, NH-Asian, NH-AIAN, multiracial); primary language (English vs non-English); disability status (any ADL or claims-based disability flag); urbanicity (metro vs non-metro); age band (19–34, 35–54, 55–64); sex; baseline Signal-score quintile.

### Aim 2

**Predictors.** Identical to Patel-Baum-Basu 2024: Medicaid eligibility, inpatient, long-term care, other-services, and pharmacy claims plus social-risk features as encoded in the public Signal repository.

**Outcomes (target labels for the model).** Identical to Patel-Baum-Basu 2024: prospective non-emergent acute care utilization (binary, top-decile risk threshold for sensitivity / specificity comparisons) and prospective annual cost (continuous).

**Subgroups (fairness analysis).** Identical to Aim 1 effect-modifier set; in addition, dual-eligibility, mental-health diagnosis presence, SUD diagnosis presence.

**Comparator models.** HHS-HCC v07 (public, scored on the same Waymark cohort using published coefficients); CDPS+Rx v7 (public, scored on the same Waymark cohort using published coefficients); standard cost-based model as defined in Patel-Baum-Basu 2024.

### Aim 3

**Mediator.** Signal predicted-outcome score at index (continuous and discretized to deciles).

**Outcome.** Aim 1 outcome variables.

**Treatment.** Aim 1 treatment variable.

## 6. Statistical analysis plan

### Aim 1 — primary specification (target trial emulation)

Target trial emulated per Hernán-Robins (2016) with grace period for treatment assignment and intention-to-treat-style follow-up.

Primary estimator: doubly-robust augmented inverse-probability-weighted estimator (AIPW) of the average treatment effect, with separate propensity model (logistic regression with regularization, or gradient boosting) and outcome model (gradient boosting). Cross-fitting via 5-fold sample splitting for orthogonality. Inference via influence-function-based standard errors.

Secondary estimator: targeted maximum likelihood estimation (TMLE) using the same propensity and outcome learners.

Heterogeneous treatment effects: causal forests (Athey-Wager 2019) using the `econml` package with subgroup CATE estimates and 95% bootstrap CIs.

Instrumental-variable robustness: state-level administrative-processing-intensity (KFF) as instrument for individual-level disenrollment, using two-stage least-squares (2SLS) for binary outcomes and survival outcomes via Frangakis-Rubin / weak-IV-robust inference.

Pre-trend / falsification: re-estimate effect using outcomes measured 2018–2022 (pre-unwinding) with placebo-treatment assignment by state; should produce null effects.

Inference: cluster-robust SE at the state level for IV; HC3 for the AIPW/TMLE; bootstrap 95% CIs throughout.

Subgroup analyses: pre-specified set above. Multiple-comparisons: BH FDR within hypothesis families F1 (primary outcomes, ~5 tests) and F2 (subgroup interactions, ~30 tests).

### Aim 2 — Signal stress-test under distribution shift

For each of the three windows (pre, unwinding, post) and for each model in {Signal Stage 1, Signal Stage 2, HHS-HCC v07, CDPS+Rx v7, standard cost-based comparator}:

1. Apply published model coefficients to the Waymark cohort within the window
2. Compute prospective performance against window-specific outcomes:
   - Sensitivity, specificity at the published top-decile-risk threshold
   - AUROC, AUPRC for top-decile classification
   - Calibration intercept, slope, ECE
   - Reliability diagrams stratified by subgroup
   - Brier score
   - Equal opportunity, predictive parity, demographic parity, calibration parity by subgroup
   - SHAP values per subgroup for top-cost-contributing categories
3. Bootstrap 95% CIs (1,000 iterations, stratified by state and panel month)
4. Pre-specified test of equity-advantage persistence: difference in (sensitivity-Black − sensitivity-White) between pre-unwinding and post-unwinding windows, with bootstrap CI; null if CI excludes zero
5. AEquity-style learning-curve sensitivity by subgroup

Multiple comparisons: BH FDR within hypothesis family F3 (window × model × metric × subgroup, ~300 tests; FDR within metric × model).

Distribution-shift quantification: Wasserstein-1 distance and KL divergence between feature distributions across windows; correlated with subgroup performance drift.

### Aim 3 — synthesis (mediation)

Two-step mediation analysis (VanderWeele 2015) with the Signal score at index as the mediator:

$$\text{Outcome} \sim \text{Treatment} + \text{Mediator(Signal score)} + \text{Confounders}$$

Counterfactual mediation: natural direct effect (NDE) and natural indirect effect (NIE) via the inverse-odds-ratio-weighting approach (Tchetgen Tchetgen-Shpitser 2012). Bootstrap 95% CIs.

Sensitivity to unmeasured mediator-outcome confounding: E-value (VanderWeele-Ding 2017).

## 7. Decision rules (pre-specified)

- **Reject null for H1a** if AIPW ATT 95% CI excludes zero in the predicted (worsening) direction for at least three of five primary outcomes (FDR-adjusted within F1).
- **Reject null for H1b** if causal-forest CATE confidence intervals across pre-specified subgroups are not all overlapping (Wald test on CATE differences, FDR-adjusted within F2).
- **Reject null for H1c** if 2SLS LATE has 95% CI overlapping the AIPW ATT.
- **Reject null for H2a** if any of {calibration intercept, slope, ECE} has a 95% bootstrap CI excluding zero for difference between any pair of windows.
- **Reject null for H2b (equity persistence)** if the difference (sensitivity-Black − sensitivity-White) post-unwinding has 95% CI excluding zero, and the magnitude is consistent with the published 2024 reversal.
- **Reject null for H2c** if Signal-vs-HHS-HCC AUROC difference has stable sign and magnitude across the three windows (CI-overlap test).
- **Reject null for H3** if conditional-on-Signal-score outcome difference between disenrolled and retained members has 95% CI excluding zero.
- **Reject null for H3 mediation** if NIE 95% CI excludes zero.

If the IV (Aim 1) and AIPW estimates disagree in sign or differ by more than an order of magnitude, the manuscript reports both and frames Aim 1 conservatively.

## 8. Reproducibility

- Code under `packaging/medicaid-unwinding-chronic-disease-fairness/code/` with `uv lock` for environment pinning
- Synthetic-data fixtures under `tests/fixtures/` for unit testing without raw data
- Public GitHub repo: `https://github.com/sanjaybasu/medicaid-unwinding-chronic-disease-fairness`
- Patel-Baum-Basu 2024 Signal code (https://github.com/sadiqypatel/Medicaid_Risk_Model) is referenced as a submodule or vendored as `code/external/signal_patel_baum_basu_2024/` with an attribution and license header
- Aggregate result tables, figures, and audit metadata (model versions, vintage, sample sizes) are committed to the public repo
- Patient-level data, model deployment artifacts, and intermediate analytic datasets remain internal to Waymark and are not committed; this constraint is documented in the manuscript Limitations and in the repository README
- Deterministic seeds: 42 throughout

## 9. Author and affiliations

Sanjay Basu MD PhD<sup>1,2</sup>; Sadiq Y. Patel PhD<sup>2,3</sup>; Aaron Baum MD PhD<sup>2,4</sup>

<sup>1</sup> Department of Medicine, University of California, San Francisco, San Francisco, CA
<sup>2</sup> Waymark, San Francisco, CA
<sup>3</sup> School of Social Policy and Practice, University of Pennsylvania, Philadelphia, PA
<sup>4</sup> Icahn School of Medicine at Mount Sinai, New York, NY

Sanjay Basu is the corresponding author.

## 10. Funding and conflicts

Funding: None declared.

Conflicts of interest: SB receives grants from NIH and CDC, outside the submitted work; receives salary support from HealthRight360 and Waymark; is a Board member and staff member at Waymark; conducts clinical work at HealthRight360, a federally-qualified health center treating patients receiving Medicaid. SP and AB receive salary support from Waymark and are co-authors of the original Patel-Baum-Basu 2024 *Sci Rep* paper that defined the Signal model audited in this study.

## 11. Ethics

The Patel-Baum-Basu 2024 study received an existing Waymark IRB-equivalent / Privacy and Compliance determination. The present follow-up extends that determination through the same Waymark internal review process; documentation is on file.

#!/usr/bin/env Rscript
# fable ETS (STL + ETS on the seasonally adjusted series) and dynamic-harmonic-regression ARIMA on the M5 hierarchy,
# forecast reconciliation (bottom-up, OLS, WLS structural, WLS variance, MinT shrink; top-down on the strict geographic
# sub-hierarchy), rolling-origin forecasts, Ljung-Box residual tests, fabletools accuracy.
#
# Inputs : data/processed/bottom_weekly.csv  (store x department weekly revenue, from sql/03_item_store_week.sql)
#          data/processed/origins.json       (rolling-origin design written by the Python package; single source of truth)
# Outputs: data/processed/r_forecasts.csv, r_accuracy.csv, r_residual_tests.csv, r_session_info.txt
# Usage  : Rscript R/forecast_reconciliation.R [processed_dir]   (set HIER_MAX_ORIGINS=n to run only the last n origins)

user_lib <- Sys.getenv("R_LIBS_USER_EXTRA", unset = file.path(Sys.getenv("HOME"), "R", "lib"))
if (dir.exists(user_lib)) .libPaths(c(user_lib, .libPaths()))
suppressPackageStartupMessages({
  library(fable); library(feasts); library(tsibble); library(dplyr); library(tidyr)
  library(readr); library(jsonlite); library(furrr); library(future)
})

args <- commandArgs(trailingOnly = TRUE)
pdir <- if (length(args) >= 1) args[1] else "data/processed"
PERIOD <- 52L
K_FOURIER <- 4L
ACC_MEASURES <- list(MASE = MASE, RMSSE = RMSSE, MAE = MAE, CRPS = CRPS)
LB_LAG <- 52L

bw <- read_csv(file.path(pdir, "bottom_weekly.csv"), show_col_types = FALSE)
design <- fromJSON(file.path(pdir, "origins.json"))
H <- design$horizon
origins <- c(design$validation, design$test)
max_o <- as.integer(Sys.getenv("HIER_MAX_ORIGINS", unset = "0"))
if (max_o > 0) origins <- tail(origins, max_o)
test_origins <- design$test
PERIOD <- as.integer(design$season)

bottom <- bw %>% as_tsibble(index = w, key = c(state, store, cat, dept))
full <- bottom %>% aggregate_key((state / store) * (cat / dept), revenue = sum(revenue))   # 154 nodes, M5 levels 1-9
geo  <- bottom %>% aggregate_key(state / store / dept, revenue = sum(revenue))             # 84 nodes, strictly hierarchical

node_id_of <- function(df) {
  g <- ifelse(!is_aggregated(df$store), paste0("S:", df$store),
              ifelse(!is_aggregated(df$state), paste0("ST:", df$state), NA_character_))
  p <- if ("dept" %in% names(df)) ifelse(!is_aggregated(df$dept), paste0("D:", df$dept),
              if ("cat" %in% names(df)) ifelse(!is_aggregated(df$cat), paste0("C:", df$cat), NA_character_) else NA_character_)
       else NA_character_
  out <- ifelse(is.na(g) & is.na(p), "T", ifelse(is.na(g), p, ifelse(is.na(p), g, paste0(g, "|", p))))
  out
}
strip <- function(x) as.character(x)

fit_models <- function(tr) {
  tr %>% model(
    ets = decomposition_model(STL(revenue ~ season(period = !!PERIOD, window = "periodic")), ETS(season_adjust)),
    arima = ARIMA(revenue ~ fourier(K = !!K_FOURIER, period = !!PERIOD) + PDQ(0, 0, 0))
  )
}

score <- function(fc, data, structure, o, base, recon) {
  rows <- fc %>%
    mutate(
      yhat = .mean,
      lo80 = quantile(revenue, 0.10), hi80 = quantile(revenue, 0.90),
      lo95 = quantile(revenue, 0.025), hi95 = quantile(revenue, 0.975)
    ) %>%
    as_tibble() %>%
    mutate(node_id = node_id_of(.), structure = structure, base_model = base, recon = recon, origin = o, h = w - o) %>%
    select(structure, base_model, recon, origin, h, node_id, yhat, lo80, hi80, lo95, hi95)
  acc <- accuracy(fc, data, measures = ACC_MEASURES) %>% as_tibble() %>%
    mutate(node_id = node_id_of(.), structure = structure, base_model = base, recon = recon, origin = o) %>%
    select(structure, base_model, recon, origin, node_id, any_of(c("MASE", "RMSSE", "MAE", "CRPS")))
  list(rows = rows, acc = acc)
}

# one reconciliation method for one base model; each method is fitted in its own call
reconcile_one <- function(fit, base, method) {
  f <- fit %>% select(m = all_of(base))
  r <- switch(method,
    bu = f %>% reconcile(x = bottom_up(m)),
    ols = f %>% reconcile(x = min_trace(m, method = "ols")),
    wls_struct = f %>% reconcile(x = min_trace(m, method = "wls_struct")),
    wls_var = f %>% reconcile(x = min_trace(m, method = "wls_var")),
    mint = f %>% reconcile(x = min_trace(m, method = "mint_shrink")),
    td_fp = f %>% reconcile(x = top_down(m, method = "forecast_proportions")),
    td_ap = f %>% reconcile(x = top_down(m, method = "average_proportions"))
  )
  r %>% select(x)
}

one_structure <- function(data, tr, structure, o, methods) {
  fit <- fit_models(tr)
  out <- list(); acc <- list()
  for (base in c("ets", "arima")) {
    fb <- forecast(fit %>% select(x = all_of(base)), h = H)
    sc <- score(fb, data, structure, o, base, "base")
    out[[length(out) + 1]] <- sc$rows; acc[[length(acc) + 1]] <- sc$acc
    for (meth in methods) {
      fr <- tryCatch(forecast(reconcile_one(fit, base, meth), h = H), error = function(e) {
        message("FAILED ", structure, " ", base, " ", meth, " origin ", o, ": ", conditionMessage(e)); NULL })
      if (is.null(fr)) next
      sc <- score(fr, data, structure, o, base, meth)
      out[[length(out) + 1]] <- sc$rows; acc[[length(acc) + 1]] <- sc$acc
    }
  }
  list(fc = bind_rows(out), acc = bind_rows(acc), fit = fit)
}

run_origin <- function(o) {
  a <- one_structure(full, full %>% filter(w <= o), "full154", o, c("bu", "ols", "wls_struct", "wls_var", "mint"))
  g <- one_structure(geo, geo %>% filter(w <= o), "geo84", o, c("bu", "ols", "mint", "td_fp", "td_ap"))
  res <- NULL
  if (o == max(test_origins)) {
    res <- augment(a$fit) %>% as_tibble() %>% mutate(node_id = node_id_of(.), .model = as.character(.model)) %>%
      select(node_id, .model, w, .innov, revenue)
  }
  list(fc = bind_rows(a$fc, g$fc), res = res, acc = bind_rows(a$acc, g$acc))
}

plan(multisession, workers = as.integer(Sys.getenv("HIER_R_WORKERS", unset = "4")))
t0 <- Sys.time()
out <- future_map(origins, run_origin, .options = furrr_options(seed = TRUE, packages = c("fable", "feasts", "tsibble", "dplyr", "tidyr")),
                  .progress = FALSE)
message("origins done in ", round(difftime(Sys.time(), t0, units = "mins"), 1), " min")

fcs <- bind_rows(lapply(out, `[[`, "fc")) %>% arrange(structure, base_model, recon, origin, h, node_id)
write_csv(fcs, file.path(pdir, "r_forecasts.csv"))
message("negative reconciled point forecasts: ", sum(fcs$yhat < 0), " of ", nrow(fcs))

# ---- Ljung-Box and related residual diagnostics on the fit at the last test origin (all 154 nodes) -----------------------------
res <- bind_rows(lapply(out, `[[`, "res"))
lb <- res %>%
  group_by(node_id, .model) %>%
  summarise(
    n = sum(!is.na(.innov)),
    lb_stat = { x <- .innov[!is.na(.innov)]; unname(Box.test(x, lag = LB_LAG, type = "Ljung-Box")$statistic) },
    lb_pvalue = { x <- .innov[!is.na(.innov)]; Box.test(x, lag = LB_LAG, type = "Ljung-Box")$p.value },
    lag1_acf = { x <- .innov[!is.na(.innov)]; cor(x[-1], x[-length(x)]) },
    shapiro_p = { x <- .innov[!is.na(.innov)]; shapiro.test(x)$p.value },
    resid_mean_over_mean_revenue = mean(.innov, na.rm = TRUE) / mean(revenue, na.rm = TRUE),
    .groups = "drop"
  ) %>% arrange(.model, node_id)
write_csv(lb, file.path(pdir, "r_residual_tests.csv"))

# ---- fabletools accuracy (before / after reconciliation), per origin x node ------------------------------------------------------
acc <- bind_rows(lapply(out, `[[`, "acc")) %>% arrange(structure, base_model, recon, origin, node_id)
write_csv(acc, file.path(pdir, "r_accuracy.csv"))
writeLines(capture.output(sessionInfo()), file.path(pdir, "r_session_info.txt"))
message("R done")

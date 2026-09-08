# used
library(here)
library(tidyverse)
library(imputeTS)
library(mgcv)
library(data.table)
library(ranger)

# script settings go here
train_cutoff <- as.Date("2025-09-30")

# unzip the training and validation datasets programatically
unzip(here("data", "raw", "turingAI_forecasting_challenge_dataset.csv.zip"),
      exdir = here("data", "interim"))
unzip(here("data", "raw", "turingAI_forecasting_challenge_validation_dataset.zip"),
      exdir = here("data", "interim"))

# fetch data
fp_train <- here("data", "interim", "turingAI_forecasting_challenge_dataset.csv")
fp_val <- here("data", "interim", "turingAI_forecasting_challenge_validation_dataset.csv")
fp_covselect <- here("data", "raw", "covariate_selection.csv")

dat_train <- read_csv(fp_train)
dat_train <- dat_train[dat_train$dt <= train_cutoff, ]
dat_val <- read_csv(fp_val)
metric_selection <- read_csv(fp_covselect)

# replace cov values of -9999 with Nan
dat1 <- rbind(dat_train, dat_val)
dat1 <- dat1 %>%
  mutate(across(where(is.numeric), ~na_if(., -9999)))

# standardize Dates and Times
dat2 <- dat1 %>%
  mutate(dt = as.POSIXct(dt, tz = "UTC"), date_only = as.Date(dt),
    # If recorded after midday, attribute to the next day's operational cycle
    operational_day = if_else(format(dt, "%H:%M:%S") > "12:00:00", date_only + 1, date_only))

# average all timesteps to 1 day at noon
dat3 <- dat2 %>%
  group_by(operational_day, metric_name, coverage) %>%
  summarise(
    daily_value = mean(value, na.rm = TRUE), 
    .groups = "drop"
  )

# what goes on here?
metric_selection <- metric_selection %>% mutate(across(everything(), str_trim))

forbidden_names <- metric_selection %>% filter(toupper(Keep) == "NO") %>% pull(`Metric Name`)

initial_unique <- dat3 %>%
  filter(metric_name != "estimated_avoidable_deaths") %>%
  pull(metric_name) %>%
  unique()

dat3 <- dat3 %>%
  filter(!(metric_name %in% forbidden_names & !metric_name %in% c("estimated_avoidable_deaths", "ICU_admissions")))

metrics_to_keep <- metric_selection %>% filter(toupper(Keep) == "YES")
unique_audit_metrics <- unique(metrics_to_keep$`Metric Name`)
for (m_name in unique_audit_metrics) {
  if (m_name %in% c("estimated_avoidable_deaths", "ICU_admissions")) next
  
  metric_rows <- metrics_to_keep %>% filter(`Metric Name` == m_name)
  locs <- metric_rows$`Which Locations`[1]
  
  if (!is.na(locs) && locs != "" && toupper(locs) != "ALL") {
    keep_locs <- str_split(locs, ",\\s*")[[1]] %>% str_trim()
    dat3 <- dat3 %>%
      filter(!(metric_name == m_name & !coverage %in% keep_locs))
  }
}

final_unique <- dat3 %>%
  filter(metric_name != "estimated_avoidable_deaths") %>% 
  pull(metric_name) %>% 
  unique()

registry_names <- unique(final_unique)

# This creates the wide-format table where each column is a unique metric
dat4 <- dat3 %>%
  
# Combine Site and Metric name to create unique column headers
mutate(column_label = paste0(coverage, "_", metric_name)) %>%
select(operational_day, column_label, daily_value) %>%
pivot_wider(names_from = column_label, values_from = daily_value)

# Ensure the dates are in chronological order
dat4 <- dat4 %>% arrange(operational_day)

# renaming outcome metric back to estimated_avoidable_deaths
dat4 <- dat4 %>%
  rename(estimated_avoidable_deaths = "NHS Bristol, North Somerset, South Gloucestershire Integrated Care Board_estimated_avoidable_deaths")

to_registry <- function(col_names) {
  vapply(col_names, function(nm) {
    hit <- registry_names[vapply(registry_names, function(rn)
      endsWith(nm, rn), logical(1))]
    if (length(hit) >= 1) hit[which.max(nchar(hit))] else nm 
  }, character(1), USE.NAMES = FALSE)
}

# find the longest continuous gap length in data
find_longest_NA_string <- function(x) {
  
  col_index <- which(!is.na(x))[1]
  current_string <- 0
  longest_string <- 0
  
  while(col_index <= length(x)) {
    if(!is.na(x[col_index])) {
      current_string <- 0
    }
    else {
      current_string <- current_string + 1
      if(current_string > longest_string) {
        longest_string <- current_string
      }
    }
    col_index <- col_index + 1
  }
  return(longest_string)
}
largest_gaps <- lapply(dat4, find_longest_NA_string)

valid_train_amnt <- function(col) {
  train_rows <- dat4[dat4$operational_day <= train_cutoff, col]
  return(sum(!is.na(train_rows)))
}
sufficient_train_dat <- sapply(names(dat4), valid_train_amnt)

weeks_of_data <- sapply(names(dat4), function(col) nrow(dat4[which(!is.na(dat4[[col]]))[1]:nrow(dat4), col]) / 7)
days_of_data <- sapply(names(dat4), function(col) length(which(!is.na(dat4[[col]]))))
avg_data_per_week <- days_of_data / weeks_of_data

keep_cols <- (largest_gaps < 10) & (avg_data_per_week > 6) & (sufficient_train_dat >= 365)
dat5 <- dat4[, keep_cols]

dat5_dropped <- dat4[, !keep_cols]
col_names <- names(dat5_dropped)

dat5_dropped$operational_day <- dat5$operational_day

# --- covariate registry accounting for the gap step ---
dropped_column_names <- setdiff(names(dat4), names(dat5))
dropped_column_names <- setdiff(dropped_column_names,
                                c("operational_day", "estimated_avoidable_deaths"))

kept_cols_cov  <- setdiff(names(dat5), c("operational_day", "estimated_avoidable_deaths"))

registry_dropped_raw <- unique(to_registry(dropped_column_names))
registry_kept        <- unique(to_registry(kept_cols_cov))

# a covariate is only *truly* dropped if NONE of its columns survived:
registry_truly_dropped <- setdiff(registry_dropped_raw, registry_kept)

# helper function to linearily interpolate interior/training NA
interpolate_non_leading <- function(x) {
  first_valid <- which(!is.na(x))[1]
  x_interp <- na_interpolation(x, option = "linear")
  if (!is.na(first_valid) && first_valid > 1) {
    x_interp[1:(first_valid - 1)] <- NA
  }
  return(x_interp)
}

# impute NAs
dat6 <- dat5 %>%
  # Linear interpolation on interior/trailing NAs only; leading NAs preserved
  mutate(across(where(is.numeric) & !estimated_avoidable_deaths, 
                ~interpolate_non_leading(.x))) %>%
  # Handle target variable
  mutate(estimated_avoidable_deaths = na_locf(estimated_avoidable_deaths, na_remaining = "rev")) %>%
  mutate(estimated_avoidable_deaths = pmax(0, estimated_avoidable_deaths))

remove_gam_outliers_wide <- function(dat, col_name) {
  
  x   <- dat[[col_name]]
  day <- as.numeric(dat$operational_day)
  
  valid <- !is.na(x)
  if (sd(x, na.rm = TRUE) == 0 || sum(valid) < 10) return(x)
  
  mu <- mean(x, na.rm = TRUE)
  s  <- sd(x, na.rm = TRUE)
  z  <- (x - mu) / s
  
  # Fit only on non-NA rows so leading-NA covariates don't misalign
  # predict()'s output against the full-length z/day vectors
  z_valid   <- z[valid]
  day_valid <- day[valid]
  
  gam_model <- gam(z_valid ~ s(day_valid, k = 45))
  pred    <- predict(gam_model, se.fit = TRUE)
  trend   <- pred$fit
  se_pred <- sqrt(gam_model$sig2 + pred$se.fit^2)
  
  z_score  <- qnorm(1 - (1 - 0.9997) / 2)   # ~3.43
  lower_ci <- trend - z_score * se_pred
  upper_ci <- trend + z_score * se_pred
  
  is_outlier_valid <- (z_valid < lower_ci) | (z_valid > upper_ci)
  is_outlier_valid[is.na(is_outlier_valid)] <- FALSE
  
  # Map outlier flags back onto the full-length vector
  is_outlier <- rep(FALSE, length(x))
  is_outlier[valid] <- is_outlier_valid
  
  # revert to original scale for plotting
  trend    <- trend    * s + mu
  lower_ci <- lower_ci * s + mu
  upper_ci <- upper_ci * s + mu
  
  x[is_outlier] <- NA
  train <- dat$operational_day <= train_cutoff
  
  x[train]  <- na_interpolation(x[train], option = "linear")
  x <- na.locf(x)
  
  # na_interpolation() extrapolates leading NAs to the first valid value;
  # restore them so leading gaps stay NA until the post-detrend fill step
  first_valid <- which(valid)[1]
  if (!is.na(first_valid) && first_valid > 1) {
    x[1:(first_valid - 1)] <- NA
  }
  return(x)
}

skip_cols <- c("operational_day", "estimated_avoidable_deaths")
for (col in setdiff(names(dat6), skip_cols)) {
  dat6[[col]] <- remove_gam_outliers_wide(dat6, col)
}

dat6$wday <- factor(lubridate::wday(dat6$operational_day, label = TRUE, abbr = TRUE))
dat6$mon <- factor(lubridate::month(dat6$operational_day, label = TRUE, abbr = TRUE))
contrasts(dat6$wday) <- contr.sum(7)
contrasts(dat6$mon) <- contr.sum(12)
seasonal_df <- model.matrix(~ wday + mon - 1, data = dat6) %>% as.data.frame()
clean_names <- c(levels(dat6$wday), levels(dat6$mon)[-12])
colnames(seasonal_df) <- clean_names
dat6 <- cbind(dat6, seasonal_df)
seasonal_cols <- colnames(seasonal_df)
covar_cols <- setdiff(colnames(dat6), c("estimated_avoidable_deaths", "operational_day", "wday", "mon", seasonal_cols))
pre_regression <- dat6 %>% select(operational_day, all_of(covar_cols))

for(column in covar_cols) {
  start_date <- dat6$operational_day[which(!is.na(dat6[[column]]))[1]]
  
  formula_str <- paste0("`", column, "` ~ ", paste(seasonal_cols, collapse = " + "))
  model <- lm(as.formula(formula_str), data = dat6 %>% filter(operational_day >= start_date))
  
  na_coefs <- names(which(is.na(coef(model))))
  if(length(na_coefs) > 0) {
    current_seasonal_cols <- setdiff(seasonal_cols, na_coefs)
    formula_str <- paste0("`", column, "` ~ ", paste(current_seasonal_cols, collapse = " + "))
    model <- lm(as.formula(formula_str), data = dat6 %>% filter(operational_day >= start_date))
  }
  
  dat6[[column]] <- dat6[[column]] - predict(model, newdata = dat6)
}

dat6 <- dat6 %>% select(-wday, -mon, -any_of(seasonal_cols))

dat6 <- dat6 %>%
  mutate(across(all_of(covar_cols), ~ {
    x <- .x
    first_valid <- which(!is.na(x))[1]
    if (!is.na(first_valid) && first_valid > 1) {
      x[1:(first_valid - 1)] <- mean(x[operational_day <= train_cutoff], na.rm = TRUE)
    }
    x
  })) %>%
  # Currently data leakage, to be handled later
  mutate(across(all_of(covar_cols), ~ (.x - mean(.x[operational_day <= train_cutoff], na.rm = TRUE)) /
                  sd(.x[operational_day <= train_cutoff], na.rm = TRUE)))

skip_cols <- c("operational_day", "estimated_avoidable_deaths")
for (col in setdiff(names(dat6), skip_cols)) {
  dat6[[col]] <- remove_gam_outliers_wide(dat6, col)
}

dat6_copy <- dat6 %>%
  mutate(across(everything(), ~ replace_na(.x, 0)))

covar_cols <- colnames(dat6_copy %>% select(-operational_day, -estimated_avoidable_deaths))

m <- length(covar_cols)
covariate_corr <- matrix(0, nrow = m, ncol = m, dimnames = list(covar_cols, covar_cols))

for (r in 1:m) {
  for (c in seq_len(r - 1)) {
    val <- max(abs(ccf(dat6_copy[[covar_cols[r]]], dat6_copy[[covar_cols[c]]],
                       lag.max = 13, plot = FALSE)$acf))
    covariate_corr[r, c] <- val
    covariate_corr[c, r] <- val
  }
}
diag(covariate_corr) <- 1

to_drop <- character(0)
for(r in 1:m) {
  for(c in seq_len(r - 1)) {
    if(covariate_corr[r, c] > 0.5) {
      
      tlcc_r <- ccf(dat6_copy[[covar_cols[r]]], dat6_copy[["estimated_avoidable_deaths"]], 
                    lag.max = 13, plot = FALSE)
      
      tlcc_c <- ccf(dat6_copy[[covar_cols[c]]], dat6_copy[["estimated_avoidable_deaths"]], 
                    lag.max = 13, plot = FALSE)
      
      lag_idx_r <- which(tlcc_r$lag >= -13 & tlcc_r$lag <= 0)
      lag_idx_c <- which(tlcc_c$lag >= -13 & tlcc_c$lag <= 0)
      
      strength_r <- mean(abs(tlcc_r$acf[lag_idx_r]))
      strength_c <- mean(abs(tlcc_c$acf[lag_idx_c]))
      
      loser <- if(strength_r >= strength_c) covar_cols[c] else covar_cols[r]
      to_drop <- union(to_drop, loser)
    }
  }
}
covar_cols_final <- setdiff(covar_cols, to_drop)
dat6 <- dat6 %>% select(all_of(c("operational_day", "estimated_avoidable_deaths", covar_cols_final)))

registry_final   <- unique(to_registry(covar_cols_final))
registry_pre_corr <- unique(to_registry(kept_cols_cov))   # what entered the corr step
registry_dropped_corr <- setdiff(registry_pre_corr, registry_final)

lag_cols <- setdiff(names(dat6), c("estimated_avoidable_deaths", "operational_day"))
lag_list <- shift(as.data.frame(dat6)[lag_cols], n = 1:13, type = "lag", give.names = TRUE)
dat_lagged <- cbind(as.data.frame(dat6), lag_list)

dat_lagged <- dat_lagged %>% select(-operational_day)
stopifnot(!any(duplicated(make.names(names(dat_lagged)))))

colnames(dat_lagged) <- make.names(colnames(dat_lagged))
rf <- ranger(estimated_avoidable_deaths ~ ., data = dat_lagged, importance = "permutation", num.trees = 2000, seed = 48)
rf_selection <- sort(ranger::importance(rf), decreasing = FALSE)

base_covariate <- sub("_lag_[0-9]+$", "", names(rf_selection))
importance_indices <- seq_along(rf_selection)
covariate_scores <- tapply(importance_indices, base_covariate, sum)
covariate_scores <- sort(covariate_scores, decreasing = TRUE)

n_keep <- 10
top_covariates <- names(covariate_scores)[1:n_keep]
name_map <- setNames(names(dat6), make.names(names(dat6)))
top_covariates_original <- name_map[top_covariates]
dat6 <- dat6 %>% select(all_of(c("operational_day", "estimated_avoidable_deaths", top_covariates_original)))

write.csv(dat6, here("data", "processed", "modeldat.csv"), row.names = FALSE)



import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
import pytensor.tensor as pt
import matplotlib.pyplot as plt
from pygam import LinearGAM, s
from datetime import datetime, timedelta
import patsy
import os
from pyprojroot import here


# Model settings go here
# ~~~~~~~~~~~~~~~~~~~~~~

n_test = 10
n_data_lag = 3
training_end_date = datetime(2025, 9, 30)
M = 4
M_prime = 4

# derived from settings
last_training_data_date = training_end_date - timedelta(days=n_data_lag)


# Get data
# ~~~~~~~~

data_path = here("data/processed/modeldat.csv")
data_dir = os.path.dirname(data_path)
data_root = os.path.dirname(data_dir)
project_root = os.path.dirname(data_root)

os.makedirs(os.path.join(project_root, "results", "figures"), exist_ok=True) 
output_dir = os.path.join(project_root, "results", "figures")

df = pd.read_csv(data_path)


# Prepare data for pyMC
# ~~~~~~~~~~~~~~~~~~~~~

# Construct outcome variable, time index, day index
y_df = df[['operational_day']]
y_df = y_df.rename(columns={'operational_day': 'dt'})
y_df['dt'] = pd.to_datetime(y_df['dt'])
y_df['t'] = np.arange(len(y_df))
y_df['day'] = y_df['dt'].dt.day_of_week
y_df['y_obs'] = df['estimated_avoidable_deaths']
y_df['log_y_obs'] = np.log1p(y_df['y_obs'])

X_scaled = df.drop(columns=['operational_day', 'estimated_avoidable_deaths'])
X_scaled_values = X_scaled.values

lag_list = []
for i in range(n_test+n_data_lag+1):
    # Shift the data down by i days 
    shifted_X = X_scaled.shift(i).add_suffix(f'_lag{i}')
    lag_list.append(shifted_X)
X_final = np.stack([df.values for df in lag_list], axis=1)

index = pd.MultiIndex.from_product(
    [y_df['t'], -np.arange(n_test+n_data_lag+1), X_scaled.columns],
    names=["t", "lag", "covariate_name"]
)
X_df = pd.DataFrame({"covariate_value": X_final.ravel()}, index=index)

# Drop the first n_test + n_data_lag timesteps in both dataframes
bad_times = X_df["covariate_value"].isna().groupby(level="t").any() # detect times containing at least one NaN
bad_times = bad_times[bad_times].index # Keep only times where NaNs were found
X_df = X_df.drop(index=bad_times, level="t")
X_df = X_df.reset_index()
y_df = y_df[~y_df['t'].isin(bad_times.values)]
X_df['dt'] = X_df['t'].map(dict(zip(y_df['t'], y_df['dt'])))

# Estimate seasonal amplitude directly from the data
mu0_est = np.median(y_df['log_y_obs'])
A_est = np.quantile(y_df['log_y_obs'], q=0.85) - np.median(y_df['log_y_obs'])

# Split the dataframes using dates
y_df_train = y_df[y_df['dt'] <= last_training_data_date]
X_df_train = X_df[X_df['dt'] <= last_training_data_date] 

# Convert to numpy arrays for the pyMC model
t_train = y_df_train['t'].values
dt_train = y_df_train['dt'].values
d_train = y_df_train['day'].values
y_train = y_df_train['log_y_obs'].values
X_train = X_df_train['covariate_value'].values.reshape(len(X_df_train['t'].unique()), len(X_df_train['lag'].unique()), len(X_df_train['covariate_name'].unique()))


# Build the spline cross-basis for the DLNM
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

n_times, n_lags, n_covs = X_final.shape
bins = np.arange(-5, 5, 0.1)
n_bins = len(bins)

K = np.digitize(X_final, bins) - 1
K_train = np.digitize(X_train, bins) - 1

N = np.sort(np.abs(X_df_train['lag'].unique()))
B = patsy.dmatrix("bs(N, df=M, degree=3, include_intercept=True) - 1", {"N": N})
B = np.asarray(B)

N_prime = bins
B_prime = patsy.dmatrix("bs(N_prime, df=M_prime, degree=3, include_intercept=True) - 1", {"N_prime": N_prime})
B_prime = np.asarray(B_prime)


# Build the pyMC model
# ~~~~~~~~~~~~~~~~~~~~

coords = { 
    "date": dt_train,
    "day": ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
    "covariates": X_df['covariate_name'].unique(),
    "b_functions_covariates": np.arange(M_prime),
    "lags": X_df['lag'].unique(),
    "b_functions_lags": np.arange(M),
    "bins": bins, 
        }

with pm.Model(coords=coords) as nhs_model:

    # Data containers for swapping
    t_shared = pm.Data("t_shared", t_train, dims="date")
    d_shared = pm.Data("d_shared", d_train, dims="date")
    X_shared = pm.Data("X_shared", X_train, dims=("date", "lags", "covariates"))
    K_shared = pm.Data("K_shared", K_train, dims=("date", "lags", "covariates"))
    y_shared = pm.Data("y_shared", y_train, dims="date")

    # Long-term seasonal model
    phi = 5.2
    mu0 = pm.Normal("mu0", mu=mu0_est, sigma=1/3)
    A = pm.LogNormal("A", mu=A_est, sigma=1/3)
    mu_seasonal = pm.Deterministic("mu_seasonal", mu0 + A * pm.math.cos((2 * np.pi * t_shared / 365.25) - phi), dims="date")

    # DLNM
    B_prime_K = pt.as_tensor(B_prime)[K_shared]
    gamma = pm.Laplace("gamma", mu=0, b=(0.1/3)/(np.sqrt(2)), shape=(M, M_prime, n_covs))
    mu_covariates = pm.Deterministic("mu_covariates", pt.einsum('nm,tncp,mpc->t', B, B_prime_K, gamma))

    # Day-of-week RE
    phi_d = pm.Normal("phi_d", mu=0, sigma=0.05/3, dims="day") 
    day_of_week_effect = phi_d[d_shared]

    # Likelihood
    mu_total = pm.Deterministic("mu_total", mu_seasonal + mu_covariates + day_of_week_effect, dims="date")
    sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.15/3)
    obs = pm.Normal("obs", mu=mu_total, sigma=sigma_obs, observed=y_shared, dims="date")


# Sample the pyMC model
# ~~~~~~~~~~~~~~~~~~~~~

with nhs_model:
    trace = pm.sample(draws=150, tune=150, chains=3, progressbar=True, target_accept=0.8,  
                      init='adapt_diag',                                                              
                      initvals=3*[{'mu0': mu0_est, 'A': A_est, 'sigma_obs': 0.15},])  
    train = pm.sample_posterior_predictive(trace)

# Save traces
os.makedirs(os.path.join(project_root, "results", "model"), exist_ok=True)   
trace.to_netcdf(os.path.join(project_root, "results", "model", "pymc_trace.nc"))

# Save traceplots
output_dir = os.path.join(project_root, "results", "figures", "traces")
os.makedirs(output_dir, exist_ok=True)
for var in ["mu0", "A", "gamma", "sigma_obs"]:
    az.plot_trace_dist(trace, var_names=[var], compact=True, combined=False, kind='kde')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'trace-{var}.pdf'))
    plt.close()

# Summary table of convergence diagnostics
summary = az.summary(trace, var_names=["mu0", "A", "gamma", "sigma_obs"])
summary.to_csv(os.path.join(output_dir, 'trace-summary.csv'), index=True)

# Training goodness-of-fit plot
post_pred = train.posterior_predictive["obs"].median(dim=['chain','draw'])
lower = train.posterior_predictive["obs"].quantile(q=0.025, dim=['chain', 'draw'])
upper = train.posterior_predictive["obs"].quantile(q=0.975, dim=['chain', 'draw'])

plt.figure(figsize=(14, 6))

plt.plot(y_df_train['dt'], y_df_train['log_y_obs'], label="Actual (Log-Deaths)", color="black", alpha=0.6, linewidth=1)
plt.plot(post_pred.coords['date'], post_pred, label="Model Median", color="blue", linewidth=1.5)
plt.fill_between(lower.coords['date'], lower, upper, color="blue", alpha=0.15, label="95% HDI")
plt.plot(post_pred.coords['date'], trace.posterior["mu_seasonal"].median(dim=['chain', 'draw']), color='red', linewidth=1.5, label="Seasonal median")

plt.xlabel("Date")
plt.ylabel("ln1p(Avoidable Deaths)")
plt.title("NHS Deaths: Model Fit (March 2023 - Sept 2025)")
plt.legend()

plt.gcf().autofmt_xdate() 

plt.savefig(os.path.join(output_dir, 'goodness-of-fit_training.pdf'))
plt.close()


# Visualise the lag-exposure surfaces
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

for c in range(n_covs):
    save_dir = os.path.join(project_root, "results", "figures", "Lag-Exposure Surfaces")
    os.makedirs(save_dir, exist_ok=True)
    
    gamma_mean = trace.posterior["gamma"].mean(("chain","draw")).values
    surface = B @ gamma_mean[:,:,c] @ B_prime.T
    
    plot_lags, plot_bins = np.meshgrid(N, bins, indexing='ij')
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(plot_bins, plot_lags, surface, cmap='coolwarm', edgecolor='k', linewidth=0.2)
    
    ax.set_xlabel('covariate values')
    ax.set_ylabel('lags')
    ax.set_zlabel('values')

    plt.title(f"Surface for {X_scaled.columns[c]}")
    file_name = f"Lag_Exposure_Surface_for_{X_scaled.columns[c]}.pdf"
    plt.savefig(os.path.join(save_dir, file_name), bbox_inches='tight')
    plt.close()
    
for c in range(n_covs):
    save_dir = os.path.join(project_root, "results", "figures", "Lag-Exposure Surfaces 2D")
    os.makedirs(save_dir, exist_ok=True)

    gamma_mean = trace.posterior["gamma"].mean(("chain","draw")).values
    surface = B @ gamma_mean[:,:,c] @ B_prime.T

    plt.imshow(surface, extent=[-5,5,0,surface.shape[0]],cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.xlabel('covariate values')
    plt.ylabel('lags')
    plt.title(f"Surface for {X_scaled.columns[c]}")
    
    file_name = f"Lag_Exposure_Surface_for_{X_scaled.columns[c]}_2D.pdf"
    plt.savefig(os.path.join(save_dir, file_name), bbox_inches='tight')
    plt.close()
    
# (n, m) @ (m, m_prime) @ (m_prime, n_prime) --> (n, n_prime)
for c in range(n_covs):
    for b in range(n_lags):
        save_dir = os.path.join(project_root, "results", "figures", "1D Exposure Curves")
        os.makedirs(save_dir, exist_ok=True)
        gamma_mean = trace.posterior["gamma"].mean(("chain","draw")).values
        gamma_quantiles = trace.posterior["gamma"].quantile([0.025, 0.5, 0.975], dim=("chain", "draw")).values
        gamma_lower, gamma_median, gamma_upper = gamma_quantiles
        
        surface = B @ gamma_mean[:,:,c] @ B_prime.T
        lower_surface = B @ gamma_lower[:,:,c] @ B_prime.T
        upper_surface = B @ gamma_upper[:,:,c] @ B_prime.T
        
        single_lag_slice = surface[b,:]
        single_lower_lag_slice = lower_surface[b,:]
        single_upper_lag_slice = upper_surface[b,:]

        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Plot Bands
        ax.fill_between(bins, single_lower_lag_slice, single_upper_lag_slice, color="pink", lw=0, zorder=1)
        
        # Plot Mean
        ax.plot(bins, single_lag_slice, color="black", marker="o", zorder=5)

        ax.set_xlabel('covariate value')
        ax.set_ylabel('value')
    
        plt.title(f"1D Curve for {X_scaled.columns[c]} on Lag {b+1}")
        file_name = f"Lag_Exposure_Surface_for_{X_scaled.columns[c]}_lag_{b+1}.pdf"
        plt.savefig(os.path.join(save_dir, file_name), bbox_inches='tight')
        plt.close()
        
# (n, m) @ (m, m_prime) @ (m_prime, n_prime) --> (n, n_prime)
lags = abs(X_df['lag'].unique())
for c in range(n_covs):
    for a in range(n_bins):
        save_dir = os.path.join(project_root, "results", "figures", "1D Lag Curves")
        os.makedirs(save_dir, exist_ok=True)
        gamma_mean = trace.posterior["gamma"].mean(("chain","draw")).values
        gamma_quantiles = trace.posterior["gamma"].quantile([0.025, 0.5, 0.975], dim=("chain", "draw")).values
        gamma_lower, gamma_median, gamma_upper = gamma_quantiles
        
        surface = B @ gamma_mean[:,:,c] @ B_prime.T
        lower_surface = B @ gamma_lower[:,:,c] @ B_prime.T
        upper_surface = B @ gamma_upper[:,:,c] @ B_prime.T
        
        single_bin_slice = surface[:,a]
        single_lower_bin_slice = lower_surface[:,a]
        single_upper_bin_slice = upper_surface[:,a]

        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Plot Bands
        ax.fill_between(lags, single_lower_bin_slice, single_upper_bin_slice, color="yellow", lw=0, zorder=1)
        
        # Plot Mean
        ax.plot(lags, single_bin_slice, color="black", marker="o", zorder=5)

        ax.set_xlabel('covariate value')
        ax.set_ylabel('value')
    
        plt.title(f"1D Curve for {X_scaled.columns[c]} on Bin {a+1}")
        file_name = f"Lag_Exposure_Surface_for_{X_scaled.columns[c]}_bin_{a+1}.pdf"
        plt.savefig(os.path.join(save_dir, file_name), bbox_inches='tight')
        plt.close()


# Assess out-of-sample forecasting accuracy
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Evaluate forecast accuracy out-of-sample
def plot_window_trajectory(train, forecast, y_df_test, dt_test, anchor_date, null_forecast, null_score=None, model_score=None):
    save_dir = os.path.join(project_root, "results", "figures", "forecasts")
    os.makedirs(save_dir, exist_ok=True)
    
    show_n_tr = 60
    dates_train = train.posterior_predictive.coords['date'].values
    dates_test = forecast.posterior_predictive.coords['date'].values

    plt.figure(figsize=(14, 6))

    # Historical Training Data & Fit
    plt.plot(dates_train[-show_n_tr:], np.expm1(train.observed_data['obs'].values[-show_n_tr:]), color="black", linewidth=2.5)
    plt.plot(dates_train[-show_n_tr:], np.expm1(train.posterior_predictive['obs'].median(dim=['chain','draw']).values[-show_n_tr:]), color="black", linewidth=0.5)
    
    plt.fill_between(dates_train[-show_n_tr:],
                     np.expm1(train.posterior_predictive['obs'].quantile(q=0.025, dim=['chain','draw']).values[-show_n_tr:]),
                     np.expm1(train.posterior_predictive['obs'].quantile(q=0.975, dim=['chain','draw']).values[-show_n_tr:]),
                     color="black", alpha=0.05)
    plt.fill_between(dates_train[-show_n_tr:],
                     np.expm1(train.posterior_predictive['obs'].quantile(q=0.25, dim=['chain','draw']).values[-show_n_tr:]),
                     np.expm1(train.posterior_predictive['obs'].quantile(q=0.75, dim=['chain','draw']).values[-show_n_tr:]),
                     color="black", alpha=0.15)

    # Data Gap Connection
    gap_dates = (y_df['dt'] >= dates_train[-1]) & (y_df['dt'] <= dates_test[0])
    plt.plot(y_df.loc[gap_dates, 'dt'], np.expm1(y_df.loc[gap_dates, 'log_y_obs']), color="blue", linewidth=2.5)

    # Out-of-Sample Test Evaluation
    plt.plot(dates_test, np.expm1(y_df_test['log_y_obs'].values), linestyle=':', color="black", linewidth=2.5)
    plt.plot(dates_test, np.expm1(forecast.posterior_predictive['obs'].median(dim=['chain','draw'])), color="red", linewidth=1.5)
    
    plt.fill_between(dates_test,
                     np.expm1(forecast.posterior_predictive['obs'].quantile(q=0.025, dim=['chain','draw'])),
                     np.expm1(forecast.posterior_predictive['obs'].quantile(q=0.975, dim=['chain','draw'])),
                     color="red", alpha=0.05)
    plt.fill_between(dates_test,
                     np.expm1(forecast.posterior_predictive['obs'].quantile(q=0.25, dim=['chain','draw'])),
                     np.expm1(forecast.posterior_predictive['obs'].quantile(q=0.75, dim=['chain','draw'])),
                     color="red", alpha=0.15)

    # Baseline Model
    plt.plot(dates_test, null_forecast, color='green', linewidth=3)

    # Vertical Markers
    plt.axvline(x=dt_test[0], color='black', linestyle='--') 
    plt.axvline(x=dt_test[n_data_lag+1], color='black', linestyle='--') 

    title_str = f"Out-of-Sample Performance: Actual vs. Predicted Avoidable Deaths ({anchor_date.strftime('%Y-%m-%d')})"
    if null_score is not None:
        title_str += f" | Horizon null MSE: {null_score:.4f}"
    if model_score is not None:
        title_str += f" | Horizon model MSE: {model_score:.4f}"
        
    plt.title(title_str)
    plt.ylabel("Avoidable Deaths")
    plt.xlabel("Date")
    plt.grid(alpha=0.3)
    
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    
    file_name = f"goodness-of-fit_forecast_start_{anchor_date.strftime('%Y-%m-%d')}.pdf"
    plt.savefig(os.path.join(save_dir, file_name), bbox_inches='tight')
    plt.close()
    
def evaluate_forecasts(start_date, train_end, num_forecasts, horizon, train_trace):

    max_date = y_df['dt'].max()
    assert start_date >= train_end, "Evaluation start date must be after training window ends."
    assert (max_date - (start_date + timedelta(days=num_forecasts - 1))).days >= horizon, "Insufficient data for final horizon."   
    
    mse_records_1_5 = []
    mse_records_6_10 = []
    gam_mse_records_1_5 = []
    gam_mse_records_6_10 = []
    date_labels = []
    forecast_container = []

    for step in range(num_forecasts):
        
        D_zero = start_date + timedelta(days=step)
        D_min3 = D_zero - timedelta(days=n_data_lag)
        D_plus10 = D_zero + timedelta(days=horizon)
        forecast_eval_dates = y_df[(y_df['dt'] > D_zero) & (y_df['dt'] <= D_plus10)]['dt'].values

        x_train_GAM = y_df[y_df['dt'] <= D_min3]['t']
        y_train_GAM = y_df[y_df['dt'] <= D_min3]['y_obs']
        x_predict_GAM = y_df[y_df['dt'] <= D_plus10]['t']
        gam = LinearGAM(s(0), fit_intercept=True).fit(x_train_GAM, y_train_GAM)
        gam_trend = pd.DataFrame(index=y_df[y_df['dt'] <= D_plus10]['dt'], data=gam.predict(x_predict_GAM), columns=['gam_trend',])

        flat_value = gam_trend.loc[D_min3, "gam_trend"]
        gam_trend.loc[D_min3:, "gam_trend"] = flat_value 

        gam_trend = gam_trend.reset_index()
        y_pred_gam = gam_trend[gam_trend['dt'].isin(forecast_eval_dates)]['gam_trend'].values
        y_vis_gam = gam_trend[((gam_trend['dt'] >= D_min3) & (gam_trend['dt'] <= D_plus10))]['gam_trend'].values

        y_slice = y_df[(y_df['dt'] >= D_min3) & (y_df['dt'] <= D_plus10)]
        X_slice = X_df[(X_df['dt'] >= D_min3) & (X_df['dt'] <= D_plus10)] 

        t_val = y_slice['t'].values
        dt_val = y_slice['dt'].values
        d_val = y_slice['day'].values
        y_val = y_slice['log_y_obs'].values
        y_val = np.concatenate((np.array([y_slice.iloc[0]['log_y_obs']]), np.full(len(y_val)-1, np.nan)), axis=0)   # retain only D-3 (other values NaN)
        y_actual = y_df[y_df['dt'].isin(forecast_eval_dates)]['y_obs'].values # NOT THE LOG

        unique_lags = sorted(X_slice['lag'].unique(), reverse=True)
        n_times = len(sorted(X_slice['t'].unique()))
        n_lags = len(unique_lags) 
        n_vars = len(X_slice['covariate_name'].unique())
        X_matrix = X_slice['covariate_value'].values.reshape(n_times, n_lags, n_vars)
        X_test = np.copy(X_matrix)
        K_test = np.digitize(X_test, bins) - 1

        for step_index in range(n_data_lag + 1, n_times):
            horizon_distance = step_index - n_data_lag - 1
            masked_lags = np.arange(n_lags) > horizon_distance
            X_test[step_index, ~masked_lags, :] = 0.0

        t_val = np.asarray(t_val, dtype=np.int64)
        d_val = np.asarray(d_val, dtype=np.int64)
        y_val = np.asarray(y_val, dtype=np.float64)
        X_test = np.asarray(X_test, dtype=np.float64)
        K_test = np.asarray(K_test, dtype=np.float64)

        if hasattr(t_val, 'filled'): t_val = t_val.filled(0)
        if hasattr(d_val, 'filled'): d_val = d_val.filled(0)
        if hasattr(y_val, 'filled'): y_val = y_val.filled(0.0)
        if hasattr(X_test, 'filled'): X_test = X_test.filled(0.0)

        t_val = np.nan_to_num(t_val, nan=0).astype(np.int64)
        d_val = np.nan_to_num(d_val, nan=0).astype(np.int64)
        y_val = np.nan_to_num(y_val, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)
        K_test = np.nan_to_num(K_test, nan=0).astype(np.int64)
        
        # Forecast
        print(f"Window Number {step}")
        with nhs_model:
            pm.set_data({"t_shared": t_val, "d_shared": d_val, "y_shared": y_val, "X_shared": X_test, "K_shared": K_test}, coords={"date": dt_val})
            forecast = pm.sample_posterior_predictive(trace, sample_vars=["obs"], progressbar=False)

        # Compute forecast MSE
        y_pred = np.expm1(forecast["posterior_predictive/obs"].median(dim=['chain', 'draw']).sel(date=forecast_eval_dates)).values
        
        mse_model_1_5 = (1 / 5) * np.sum((y_actual[0:5] - y_pred[0:5])**2)
        mse_model_6_10 = (1 / 5) * np.sum((y_actual[5:10] - y_pred[5:10])**2)
        mse_records_1_5.append(mse_model_1_5)
        mse_records_6_10.append(mse_model_6_10)

        # Compute null model MSE
        mse_null_1_5 = (1 / 5) * np.sum((y_actual[0:5] - y_pred_gam[0:5])**2)
        mse_null_6_10 = (1 / 5) * np.sum((y_actual[5:10] - y_pred_gam[5:10])**2)
        gam_mse_records_1_5.append(mse_null_1_5)
        gam_mse_records_6_10.append(mse_null_6_10)

        date_labels.append(D_zero.strftime('%B %d, %Y'))
        
        mse_model_total = (1 / len(y_actual)) * np.sum((y_actual - y_pred)**2)
        mse_null_total = (1 / len(y_actual)) * np.sum((y_actual - y_pred_gam)**2)

        plot_window_trajectory(
            train=train_trace,
            forecast=forecast,
            y_df_test=y_slice,
            dt_test=dt_val,
            anchor_date=D_zero,
            null_forecast=y_vis_gam,
            null_score=mse_null_total,
            model_score=mse_model_total
        )
        forecast_container.append(y_pred) # This keeps a collection of the forecasts to create pred_matrix
    
    pred_matrix = pd.DataFrame({
        "forecast_id": np.arange(1, num_forecasts + 1)
    } | {
        f"day_{x}": [forecast_container[i][x-1] for i in range(num_forecasts)]
    for x in range(1, 11)
    })
    csv_out = os.path.join(project_root, "results", "tables", "pred_matrix.csv")
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    pred_matrix.to_csv(csv_out, index=False)

    mse_summary = pd.DataFrame({
        "forecast_id": np.arange(1, num_forecasts + 1),
        "mse_1_5": mse_records_1_5,
        "mse_6_10": mse_records_6_10
    })
    csv_out = os.path.join(project_root, "results", "tables", "mse_summary.csv")
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    mse_summary.to_csv(csv_out, index=False)

    bayesian_and_gam = pd.DataFrame({
        "Evaluation_Date": date_labels, 
        "MSE_model_1_5": mse_records_1_5, 
        "MSE_model_6_10": mse_records_6_10, 
        "MSE_null_1_5": gam_mse_records_1_5,
        "MSE_null_6_10": gam_mse_records_6_10
    })
    csv_out = os.path.join(project_root, "results", "tables", "bayesian_and_gam.csv")
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    bayesian_and_gam.to_csv(csv_out, index=False)

    print(f'Mean MSE for Bayesian model (Days 1-5): {np.mean(mse_records_1_5)}')
    print(f'Mean MSE for null model (Days 1-5): {np.mean(gam_mse_records_1_5)}')
    print(f'MSE ratio (Bayesian/null) (Days 1-5): {np.mean(mse_records_1_5)/np.mean(gam_mse_records_1_5)}')
    print(f'Mean MSE for Bayesian model (Days 6-10): {np.mean(mse_records_6_10)}')
    print(f'Mean MSE for null model (Days 6-10): {np.mean(gam_mse_records_6_10)}')
    print(f'MSE ratio (Bayesian/null) (Days 6-10): {np.mean(mse_records_6_10)/np.mean(gam_mse_records_6_10)}')

    return pred_matrix, mse_summary, bayesian_and_gam, np.mean(mse_records_1_5), np.mean(mse_records_6_10)

evaluate_forecasts(datetime(2025, 9, 30), training_end_date, 131, n_test, train)
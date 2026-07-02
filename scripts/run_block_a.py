# Block A: leakage-controlled tabular benchmark on Ames Housing (De Cock, 2011).
# Reproduces Tables 1-2 and Figures 1-5 of the manuscript. CPU-only, ~5 min.
# Usage: pip install -r ../requirements.txt && python run_block_a.py
import numpy as np, pandas as pd, os, json, time, warnings, platform
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
from sklearn.model_selection import train_test_split, KFold, cross_validate, ParameterSampler, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor, DMatrix
from scipy import stats as sp_stats
import sklearn, xgboost

SEED, TARGET, N_BOOT = 42, 'SalePrice', 2000
OUT = 'results_rerun'
os.makedirs(OUT, exist_ok=True)

# ---------- data ----------
import rdatasets  # bundles openintro/ames (2930 x 82); alt: fetch_openml(data_id=42165)
df = rdatasets.data('openintro', 'ames')
df = df.drop(columns=[c for c in ['rownames', 'Order', 'PID'] if c in df.columns])
df = df.rename(columns={'price': TARGET})
df.columns = [str(c).strip().replace(' ', '') for c in df.columns]
df[TARGET] = pd.to_numeric(df[TARGET], errors='coerce')
df = df[df[TARGET].notna()].drop_duplicates().reset_index(drop=True)
print(f'Ames Housing: {df.shape[0]} rows, {df.shape[1]-1} features')

X_raw, y_raw = df.drop(columns=[TARGET]), df[TARGET]
y_q = pd.qcut(y_raw, q=5, labels=False, duplicates='drop')
X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2,
                                                    random_state=SEED, stratify=y_q)
y_train_log, y_test_log = np.log1p(y_train), np.log1p(y_test)
num_feats = X_train.select_dtypes(include='number').columns.tolist()
cat_feats = X_train.select_dtypes(exclude='number').columns.tolist()

class QuantileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lo=0.01, hi=0.99): self.lo, self.hi = lo, hi
    def fit(self, X, y=None):
        X = np.array(X, dtype=float)
        self.lower_ = np.nanquantile(X, self.lo, axis=0)
        self.upper_ = np.nanquantile(X, self.hi, axis=0); return self
    def transform(self, X, y=None):
        X = np.array(X, dtype=float).copy()
        for i in range(X.shape[1]): X[:, i] = np.clip(X[:, i], self.lower_[i], self.upper_[i])
        return X
    def get_feature_names_out(self, input_features=None):
        return np.array(input_features, dtype=object) if input_features is not None \
            else np.arange(len(self.lower_)).astype(str)

prep = ColumnTransformer([
    ('num', Pipeline([('clip', QuantileClipper()), ('imp', SimpleImputer(strategy='median'))]), num_feats),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                      ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False, min_frequency=5))]), cat_feats)
], remainder='drop')
Xtr = prep.fit_transform(X_train)           # statistics from train only
Xte = prep.transform(X_test)
feat_names = [f.replace('num__', '').replace('cat__', '') for f in prep.get_feature_names_out()]
cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
ytr = y_train_log.values
y_true = np.expm1(y_test_log.values)

# ---------- baseline CV ----------
rows = []
for name, mdl in {'Dummy (mean)': DummyRegressor(strategy='mean'),
                  'LinearRegression': LinearRegression(), 'Ridge': Ridge(alpha=1.0)}.items():
    sc = cross_validate(mdl, Xtr, ytr, cv=cv, scoring=['neg_root_mean_squared_error', 'r2'], n_jobs=-1)
    rows.append({'Model': name, 'CV_RMSE_log': -sc['test_neg_root_mean_squared_error'].mean(),
                 'CV_RMSE_log_std': sc['test_neg_root_mean_squared_error'].std(),
                 'CV_R2': sc['test_r2'].mean(), 'CV_R2_std': sc['test_r2'].std()})
    print(rows[-1])
pd.DataFrame(rows).to_csv(f'{OUT}/baseline_cv.csv', index=False)

# ---------- XGBoost randomised search (20 configs x 5-fold) ----------
param_dist = {'n_estimators': [600, 900, 1200, 1500], 'max_depth': [4, 5, 6, 7],
              'learning_rate': [0.01, 0.02, 0.03, 0.05], 'subsample': [0.75, 0.85, 0.95],
              'colsample_bytree': [0.7, 0.8, 0.9], 'min_child_weight': [1, 2, 4],
              'gamma': [0.0, 0.05, 0.1], 'reg_lambda': [0.8, 1.2, 2.0], 'reg_alpha': [0.0, 0.02, 0.08]}
folds = list(cv.split(Xtr))
results, t0 = [], time.perf_counter()
for p in ParameterSampler(param_dist, n_iter=20, random_state=SEED):
    rm = []
    for tr, va in folds:
        m = XGBRegressor(objective='reg:squarederror', tree_method='hist',
                         random_state=SEED, n_jobs=-1, verbosity=0, **p)
        m.fit(Xtr[tr], ytr[tr])
        rm.append(np.sqrt(mean_squared_error(ytr[va], m.predict(Xtr[va]))))
    results.append({'params': p, 'cv_rmse_log': float(np.mean(rm))})
best = min(results, key=lambda r: r['cv_rmse_log'])
tune_time = time.perf_counter() - t0
print(f"best cv rmse_log={best['cv_rmse_log']:.5f} ({tune_time:.0f}s) {best['params']}")

def fit_xgb():
    m = XGBRegressor(objective='reg:squarederror', tree_method='hist',
                     random_state=SEED, n_jobs=-1, verbosity=0, **best['params'])
    m.fit(Xtr, ytr); return m

# ---------- test comparison with bootstrap CI + Wilcoxon ----------
models = {'Dummy (mean)': DummyRegressor(strategy='mean'), 'LinearRegression': LinearRegression(),
          'Ridge': Ridge(alpha=1.0),
          'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=8, random_state=SEED, n_jobs=-1),
          'GradientBoosting': GradientBoostingRegressor(n_estimators=700, learning_rate=0.03, max_depth=3,
                                                        subsample=0.9, random_state=SEED),
          'XGBoost (tuned)': None}
rng_b = np.random.default_rng(123)
idx_mat = rng_b.integers(0, len(y_true), (N_BOOT, len(y_true)))
comp, preds = [], {}
for name, mdl in models.items():
    t0 = time.perf_counter()
    m = fit_xgb() if mdl is None else clone(mdl).fit(Xtr, ytr)
    ft = time.perf_counter() - t0
    yp = np.expm1(m.predict(Xte)); preds[name] = yp
    r2b = np.array([r2_score(y_true[i], yp[i]) for i in idx_mat])
    rmb = np.array([np.sqrt(mean_squared_error(y_true[i], yp[i])) for i in idx_mat])
    comp.append({'Model': name, 'R2': round(r2_score(y_true, yp), 4),
                 'R2_lo': round(np.percentile(r2b, 2.5), 4), 'R2_hi': round(np.percentile(r2b, 97.5), 4),
                 'RMSE': int(np.sqrt(mean_squared_error(y_true, yp))),
                 'RMSE_lo': int(np.percentile(rmb, 2.5)), 'RMSE_hi': int(np.percentile(rmb, 97.5)),
                 'MAE': int(mean_absolute_error(y_true, yp)),
                 'MAPE': round(float(np.mean(np.abs((y_true-yp)/y_true))*100), 2),
                 'fit_time_s': round(ft, 2)})
    print(comp[-1])
pd.DataFrame(comp).to_csv(f'{OUT}/test_comparison_A.csv', index=False)
w, pval = sp_stats.wilcoxon(np.abs(y_true - preds['XGBoost (tuned)']),
                            np.abs(y_true - preds['LinearRegression']), alternative='less')
print(f'Wilcoxon (XGB < LR): W={w:.0f}, p={pval:.4f}')

# ---------- SHAP (exact TreeSHAP via XGBoost pred_contribs) ----------
xgb_m = fit_xgb()
contribs = xgb_m.get_booster().predict(DMatrix(Xte), pred_contribs=True)
sv = contribs[:, :-1]
mean_abs = pd.Series(np.abs(sv).mean(0), index=feat_names).sort_values(ascending=False)
mean_abs.head(20).to_csv(f'{OUT}/shap_top20.csv')
try:
    import shap
    plt.figure(figsize=(9, 6))
    shap.summary_plot(sv, Xte, feature_names=[f.replace('.', ' ') for f in feat_names],
                      max_display=15, show=False)
    plt.tight_layout(); plt.savefig(f'{OUT}/shap_summary.png', dpi=300, bbox_inches='tight'); plt.close()
except ImportError:
    print('shap not installed - skipping summary plot (values already saved)')

# ---------- OOF error meta-model ----------
oof_log = cross_val_predict(Pipeline([('m', XGBRegressor(objective='reg:squarederror', tree_method='hist',
                                                          random_state=SEED, n_jobs=-1, verbosity=0,
                                                          **best['params']))]),
                            Xtr, ytr, cv=cv, n_jobs=-1)
oof_err = np.abs(np.expm1(oof_log) - np.expm1(ytr))
X_e = SimpleImputer(strategy='median').fit_transform(X_train[num_feats])
gbr = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=SEED)
cv_r2_e = float(cross_val_score(gbr, X_e, oof_err, cv=cv, scoring='r2').mean())
print(f'OOF mean abs error: {oof_err.mean():,.0f} USD | error-predictor CV R2 = {cv_r2_e:.4f}')

# ---------- summary ----------
json.dump({'dataset': {'n': int(len(df)), 'n_train': int(len(X_train)), 'n_test': int(len(X_test)),
                       'n_features_after_prep': int(Xtr.shape[1])},
           'baselines_cv': rows, 'xgb_best_params': best['params'],
           'xgb_best_cv_rmse_log': best['cv_rmse_log'], 'xgb_tuning_time_s': round(tune_time, 1),
           'wilcoxon': {'W': float(w), 'p': float(pval)}, 'test_comparison': comp,
           'shap_top10': mean_abs.head(10).round(4).to_dict(),
           'oof': {'mean_abs_err_usd': float(oof_err.mean()), 'error_predictor_cv_r2': cv_r2_e},
           'seed': SEED, 'n_boot': N_BOOT,
           'env': {'python': platform.python_version(), 'sklearn': sklearn.__version__,
                   'xgboost': xgboost.__version__, 'numpy': np.__version__, 'pandas': pd.__version__}},
          open(f'{OUT}/block_a_summary.json', 'w'), indent=2, default=float)
print('BLOCK A COMPLETE ->', OUT)

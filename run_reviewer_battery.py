"""Reviewer-defusing battery: decomposition, placebo change-point,
findings-field RWE artifact, boilerplate-vs-content length.
PHI-safe: only aggregates written. Reuses run_core loaders/canonicaliser."""
import json, numpy as np, pandas as pd
from run_core import (load, build_sample, split_groups, fit_text, metrics,
                      canonicalise, LEX, RES, RNG)
from sklearn.linear_model import LogisticRegression
np.random.seed(RNG)
df = load(); OUT = {}

# ---------- 1 STRUCTURE-ONLY vs CONTENT-ONLY DECOMPOSITION ----------
# content-only  = clinical-lexicon indicator vector (medicine, zero formatting)
# structure-only = length/missingness/punct/digit/marker counts (zero clinical words)
def decomp(years_pre, years_post, tag):
    s = build_sample(df, years_pre, years_post); tr, te = split_groups(s)
    # content-only features: presence of each finding category in impression
    def content_X(d):
        return np.column_stack([d['diag_s'].str.contains(p, regex=True).astype(float)
                                for p in LEX.values()])
    # structure-only features: NO clinical words, only form
    def struct_X(d):
        digits = d['diag_s'].str.count(r'[0-9]')
        punct  = d['diag_s'].str.count(r'[，。；：、？！,.;:?!（）()【】]')
        return np.column_stack([
            d['miss_find'].values, d['emerg'].values,
            np.log1p(d['dlen'].values), np.log1p(d['flen'].values),
            digits.values, punct.values, d['report_hour'].values])
    res = {}
    for name, fx in [('content_only', content_X), ('structure_only', struct_X)]:
        Xtr, Xte = fx(tr), fx(te)
        mu, sd = Xtr.mean(0), Xtr.std(0)+1e-9
        clf = LogisticRegression(class_weight='balanced', max_iter=500, random_state=RNG)
        clf.fit((Xtr-mu)/sd, tr['y'])
        p = clf.predict_proba((Xte-mu)/sd)[:, 1]
        res[name] = metrics(te['y'].values, p, groups=te['reg'].values)
    return res
OUT['decomposition'] = decomp([2018, 2019], [2020, 2021], 'main')

# ---------- 2 PLACEBO CHANGE-POINT ----------
# real event = Dec 2019. Test fake splits that avoid straddling the real one.
def placebo(pre_years, post_years):
    s = build_sample(df, pre_years, post_years, n_each=8000)
    tr, te = split_groups(s)
    m, _, _ = fit_text(tr['diag_s'], tr['y'], te['diag_s'], te['y'], te['reg'].values)
    return m['roc_auc'], m['roc_auc_95ci']
OUT['placebo'] = {
  'real_2018-19_vs_2020-21': dict(zip(['auc', 'ci'], placebo([2018, 2019], [2020, 2021]))),
  'placebo_2016-17_vs_2018-19_all_pre': dict(zip(['auc', 'ci'], placebo([2016, 2017], [2018, 2019]))),
  'placebo_2021-22_vs_2023-24_all_post': dict(zip(['auc', 'ci'], placebo([2021, 2022], [2023, 2024]))),
}

# ---------- 3 FINDINGS-FIELD RWE ARTIFACT ----------
# hemorrhage prevalence queried from FINDINGS field only, by year -> craters at Dec2019
d = df[df.year.between(2016, 2025)]
g = d.groupby('year')
find_heme = g.apply(lambda x: 100*d['findings_s'].loc[x.index].str.contains(LEX['hemorrhage'], regex=True).mean())
diag_heme = g.apply(lambda x: 100*d['diag_s'].loc[x.index].str.contains(LEX['hemorrhage'], regex=True).mean())
OUT['rwe_artifact'] = dict(
  findings_field_hemorrhage_pct_by_year={int(y): round(float(v), 2) for y, v in find_heme.items()},
  impression_field_hemorrhage_pct_by_year={int(y): round(float(v), 2) for y, v in diag_heme.items()},
  note='Querying the findings field alone shows hemorrhage vanishing in 2020 (pure artifact); impression is stable.')

# ---------- 4 BOILERPLATE vs CONTENT LENGTH ----------
raw_len = g.apply(lambda x: float(d['dlen'].loc[x.index].median()))
can = canonicalise(d['diag_s'])
can_len = g.apply(lambda x: float(can.loc[x.index].str.len().median()))
OUT['length_test'] = dict(
  raw_median_by_year={int(y): round(float(v), 1) for y, v in raw_len.items()},
  canonical_median_by_year={int(y): round(float(v), 1) for y, v in can_len.items()},
  raw_growth_2016_2025=float(raw_len.loc[2025]-raw_len.loc[2016]),
  canonical_growth_2016_2025=float(can_len.loc[2025]-can_len.loc[2016]))

json.dump(OUT, open(RES/'reviewer_battery.json', 'w'), ensure_ascii=False, indent=2)
print('DECOMPOSITION  content-only AUC {:.3f} {} | structure-only AUC {:.3f} {}'.format(
    OUT['decomposition']['content_only']['roc_auc'], OUT['decomposition']['content_only']['roc_auc_95ci'],
    OUT['decomposition']['structure_only']['roc_auc'], OUT['decomposition']['structure_only']['roc_auc_95ci']))
print('PLACEBO  real {:.3f} | placebo-pre {:.3f} | placebo-post {:.3f}'.format(
    OUT['placebo']['real_2018-19_vs_2020-21']['auc'],
    OUT['placebo']['placebo_2016-17_vs_2018-19_all_pre']['auc'],
    OUT['placebo']['placebo_2021-22_vs_2023-24_all_post']['auc']))
print('RWE  findings-field hemorrhage 2018/2019/2020/2021: {}/{}/{}/{}'.format(
    OUT['rwe_artifact']['findings_field_hemorrhage_pct_by_year'][2018],
    OUT['rwe_artifact']['findings_field_hemorrhage_pct_by_year'][2019],
    OUT['rwe_artifact']['findings_field_hemorrhage_pct_by_year'][2020],
    OUT['rwe_artifact']['findings_field_hemorrhage_pct_by_year'][2021]))
print('LENGTH  raw growth {:+.0f} ch | canonical growth {:+.0f} ch'.format(
    OUT['length_test']['raw_growth_2016_2025'], OUT['length_test']['canonical_growth_2016_2025']))

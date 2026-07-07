#!/usr/bin/env python3
"""
RadAudit-CN v2 analysis engine — part 2 (downstream consequences + remedies).

  8  downstream-harm: deployed-QA alert-rate transportability
  9  unsupervised drift monitor (PSI), label-free
  10 longitudinal copy-forward sub-study

Imports the part-1 module so the data loader, lexicon and canonicaliser are
shared. PHI-safe: writes only aggregate rates / distributions.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run_core import (load, empty, canonicalise, LEX, HEAD_TERMS,
                      NONHEAD_TERMS, POINTER, RES, RNG)

np.random.seed(RNG)


# ============================================================ 8 DOWNSTREAM HARM
def sec_downstream(df):
    """Two standard, pre-specified QA rules whose alert rate explodes at the
    migration even though the *true* clinical/error content is stable."""
    d = df[df.year.between(2016, 2025)].copy()
    d["nonhead"] = d["both"].str.contains(NONHEAD_TERMS, regex=True)
    d["head"] = d["both"].str.contains(HEAD_TERMS, regex=True)
    d["anatomy_mismatch"] = (d["nonhead"] & ~d["head"]).astype(int)  # head CT, no head term, has body term
    d["cross_ptr"] = d["both"].str.contains(POINTER, regex=True).astype(int)
    g = d.groupby("year")

    rule_completeness = (g["miss_find"].mean()*100).round(2)          # "findings section present?"
    rule_anatomy = (g["anatomy_mismatch"].mean()*100).round(2)        # "anatomy matches order?"
    rule_pointer = (g["cross_ptr"].mean()*100).round(2)               # "no external pointer?"

    # true clinical content (hemorrhage prevalence) for contrast
    heme = d["both"].str.contains(LEX["hemorrhage"], regex=True)
    rule_heme = (g.apply(lambda x: 100*heme.loc[x.index].mean())).round(2)

    def expl(series):
        pre = series.loc[2018:2019].mean(); post = series.loc[2020:2021].mean()
        return dict(pre_2018_19=float(pre), post_2020_21=float(post),
                    fold_change=float(post/pre) if pre > 0 else None,
                    by_year={int(y): float(v) for y, v in series.items()})

    return dict(
        completeness_missing_findings_rule=expl(rule_completeness),
        anatomy_mismatch_rule=expl(rule_anatomy),
        external_pointer_rule=expl(rule_pointer),
        true_hemorrhage_prevalence=expl(rule_heme),
        note=("Completeness and anatomy-mismatch alert rates jump several-fold "
              "at the Dec-2019 migration while hemorrhage prevalence is flat; "
              "the alerts reflect documentation structure, not true errors."))


# ============================================================ 9 DRIFT MONITOR
def psi(ref, cur, bins):
    r, _ = np.histogram(ref, bins=bins); c, _ = np.histogram(cur, bins=bins)
    r = r/ max(r.sum(), 1); c = c/ max(c.sum(), 1)
    r = np.clip(r, 1e-4, None); c = np.clip(c, 1e-4, None)
    return float(np.sum((c-r)*np.log(c/r)))


def sec_drift_monitor(df):
    """Label-free monitor: monthly Population Stability Index of report-length
    and findings-missingness against a fixed 2018 reference window. Crosses the
    standard PSI>0.25 'major shift' line exactly at the migration."""
    d = df[df.apply_t_dt.notna()].copy()
    ref = d[(d.month >= "2018-01-01") & (d.month <= "2018-12-31")]
    len_bins = np.quantile(ref["dlen"], np.linspace(0, 1, 11))
    len_bins[0], len_bins[-1] = -np.inf, np.inf
    miss_bins = np.array([-np.inf, 0.5, np.inf])  # 0 vs 1 missingness

    rows = []
    for m, sub in d[(d.month >= "2018-01-01") & (d.month <= "2021-12-31")
                    ].groupby("month"):
        rows.append(dict(
            month=str(m.date()),
            psi_length=psi(ref["dlen"], sub["dlen"], len_bins),
            psi_missing=psi(ref["miss_find"], sub["miss_find"], miss_bins),
            n=int(len(sub))))
    mon = pd.DataFrame(rows)
    mon["psi_combined"] = mon[["psi_length", "psi_missing"]].max(axis=1)
    first_alert = mon.loc[mon.psi_combined > 0.25, "month"]
    return dict(
        monthly=mon.round(4).to_dict(orient="records"),
        alert_threshold=0.25,
        first_month_over_threshold=(str(first_alert.iloc[0])
                                    if len(first_alert) else None),
        max_psi=float(mon.psi_combined.max()),
        note=("PSI>0.25 is the conventional 'major population shift' alert "
              "level used in model monitoring."))


# ============================================================ 10 COPY-FORWARD
def char_ngrams(s, n=3):
    s = s or ""
    return set(s[i:i+n] for i in range(max(len(s)-n+1, 0)))


def cosine_set(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / np.sqrt(len(a)*len(b))


def sec_copyforward(df, n_sample=10000):
    """Longitudinal near-duplication in repeat-scan patients: a QA opportunity
    (interval-change reasoning) and a risk (copy-forward)."""
    d = df.copy()
    d["sort_t"] = d["exam_end_dt"].fillna(d["apply_t_dt"])
    d = d[d["sort_t"].notna()]
    counts = d.groupby("reg").size()
    multi = counts[counts >= 2]
    dist = counts.value_counts().sort_index()
    dd = d[d.reg.isin(multi.index)].sort_values(["reg", "sort_t"])

    pairs = []  # (reg, days, textA, textB)
    for reg, sub in dd.groupby("reg"):
        sub = sub.reset_index()
        for i in range(len(sub)-1):
            dt = (sub.loc[i+1, "sort_t"]-sub.loc[i, "sort_t"]).days
            pairs.append((sub.loc[i, "diag_s"], sub.loc[i+1, "diag_s"], dt))
    rng = np.random.default_rng(RNG)
    idx = rng.choice(len(pairs), min(n_sample, len(pairs)), replace=False)
    sims, intervals, exact = [], [], 0
    for j in idx:
        a, b, dt = pairs[j]
        if a == b and a != "":
            exact += 1
        sims.append(cosine_set(char_ngrams(a), char_ngrams(b)))
        intervals.append(dt)
    sims = np.array(sims); intervals = np.array(intervals)
    # short (<=30d) vs long (>180d) interval near-duplicates
    short = sims[intervals <= 30]; long = sims[intervals > 180]
    all_exact = sum(1 for a, b, _ in pairs if a == b and a != "")
    return dict(
        identifiers_with_multiple=int(len(multi)),
        max_records_one_id=int(counts.max()),
        records_per_id_distribution={int(k): int(v) for k, v in dist.items()},
        n_consecutive_pairs=int(len(pairs)),
        median_interval_days=float(np.median([p[2] for p in pairs])),
        sampled=int(len(idx)),
        median_similarity=float(np.median(sims)),
        p90_similarity=float(np.percentile(sims, 90)),
        prop_ge_080=float((sims >= 0.80).mean()),
        prop_ge_090=float((sims >= 0.90).mean()),
        exact_repeat_pairs_all=int(all_exact),
        median_sim_short_interval=float(np.median(short)) if len(short) else None,
        median_sim_long_interval=float(np.median(long)) if len(long) else None,
        prop_ge_090_short=float((short >= .9).mean()) if len(short) else None,
        prop_ge_090_long=float((long >= .9).mean()) if len(long) else None)


if __name__ == "__main__":
    print("loading…"); df = load()
    R = {}
    print("8 downstream harm"); R["downstream_harm"] = sec_downstream(df)
    print("9 drift monitor"); R["drift_monitor"] = sec_drift_monitor(df)
    print("10 copy-forward"); R["copy_forward"] = sec_copyforward(df)
    json.dump(R, open(RES/"consequence_results.json", "w"),
              ensure_ascii=False, indent=2)
    print("wrote", RES/"consequence_results.json")
    h = R["downstream_harm"]
    print("  completeness rule pre/post:",
          round(h["completeness_missing_findings_rule"]["pre_2018_19"], 1),
          "->", round(h["completeness_missing_findings_rule"]["post_2020_21"], 1), "%")
    print("  anatomy-mismatch rule pre/post:",
          round(h["anatomy_mismatch_rule"]["pre_2018_19"], 1), "->",
          round(h["anatomy_mismatch_rule"]["post_2020_21"], 1), "%",
          "(", round(h["anatomy_mismatch_rule"]["fold_change"], 1), "x )")
    print("  hemorrhage prevalence pre/post:",
          round(h["true_hemorrhage_prevalence"]["pre_2018_19"], 1), "->",
          round(h["true_hemorrhage_prevalence"]["post_2020_21"], 1), "% (stable)")
    print("  drift monitor first alert month:",
          R["drift_monitor"]["first_month_over_threshold"],
          "| max PSI", round(R["drift_monitor"]["max_psi"], 2))
    print("  copy-forward >=0.90 short vs long interval:",
          round(R["copy_forward"]["prop_ge_090_short"], 3), "vs",
          round(R["copy_forward"]["prop_ge_090_long"], 3))

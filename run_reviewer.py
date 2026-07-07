#!/usr/bin/env python3
"""Reviewer-defense experiments for RadAudit-CN v2.
  R1 structure-only vs content-only variance decomposition
  R2 placebo change-point specificity (AUC vs candidate split)
  R3 findings-field real-world-evidence artifact (terms crater to 0%)
  R4 boilerplate-vs-content length (raw vs canonical length over time)
PHI-safe: aggregates only."""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from run_core import load, canonicalise, LEX, RES, RNG

np.random.seed(RNG)
PUNCT = r"[，。；：、？！,.;:?!（）()【】\[\]<>《》\"'/\\\-—~·…]"


def grouped_auc(X, y, groups, seed=RNG):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    tr, te = next(gss.split(X, groups=groups))
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    clf = LogisticRegression(class_weight="balanced", max_iter=500, random_state=seed)
    clf.fit((X[tr]-mu)/sd, y[tr])
    p = clf.predict_proba((X[te]-mu)/sd)[:, 1]
    return float(roc_auc_score(y[te], p))


def balanced(df, yrs_pre, yrs_post, n=12000, seed=RNG):
    pre = df[df.year.isin(yrs_pre)]; post = df[df.year.isin(yrs_post)]
    if len(pre) == 0 or len(post) == 0:
        return None
    rng = np.random.default_rng(seed)
    pi = rng.choice(pre.index, min(n, len(pre)), replace=False)
    qi = rng.choice(post.index, min(n, len(post)), replace=False)
    s = df.loc[np.concatenate([pi, qi])].copy()
    s["y"] = s.year.isin(yrs_post).astype(int)
    return s


def content_features(s):
    return np.column_stack([s["diag_s"].str.contains(p, regex=True).astype(float).values
                            for p in LEX.values()])


def structure_features(s):
    punct = s["diag_s"].str.count(PUNCT).astype(float).values
    digit = s["diag_s"].str.count(r"[0-9]").astype(float).values
    return np.column_stack([
        s["miss_find"].values.astype(float), s["emerg"].values.astype(float),
        np.log1p(s["dlen"].values), np.log1p(s["flen"].values),
        s["report_hour"].values, punct, digit])


def r1_decomposition(df):
    s = balanced(df, [2018, 2019], [2020, 2021])
    y, g = s["y"].values, s["reg"].values
    out = dict(
        content_only=grouped_auc(content_features(s), y, g),
        structure_only=grouped_auc(structure_features(s), y, g),
        combined=grouped_auc(np.column_stack([content_features(s),
                                              structure_features(s)]), y, g),
        n=int(len(s)), n_content_feats=content_features(s).shape[1],
        n_structure_feats=structure_features(s).shape[1])
    return out


def r2_placebo(df):
    rows = []
    for Y in range(2017, 2025):
        s = balanced(df, [Y-2, Y-1], [Y, Y+1])
        if s is None or s["y"].nunique() < 2:
            continue
        auc = grouped_auc(structure_features(s), s["y"].values, s["reg"].values)
        rows.append(dict(split_year=int(Y), auc=round(auc, 4),
                         is_true_transition=(Y == 2020)))
    return dict(by_split=rows,
                true_transition_auc=next(r["auc"] for r in rows if r["is_true_transition"]),
                max_placebo_auc=max(r["auc"] for r in rows if not r["is_true_transition"]))


def r3_rwe_artifact(df):
    d = df[df.year.between(2016, 2025)].copy()
    g = d.groupby("year")
    # descriptive imaging terms that lived in the FINDINGS field pre-2020
    terms = {"low_density 低密度": r"低密度", "density 密度": r"密度",
             "ventricle 脑室": r"脑室", "midline 中线": r"中线",
             "sulci/fissure 脑沟/脑裂": r"脑沟|脑裂"}
    res = {}
    for name, pat in terms.items():
        in_find = g.apply(lambda x: 100*d["findings_s"].loc[x.index].str.contains(pat, regex=True).mean())
        in_union = g.apply(lambda x: 100*d["both"].loc[x.index].str.contains(pat, regex=True).mean())
        res[name] = dict(findings_field={int(y): round(v, 2) for y, v in in_find.items()},
                         union={int(y): round(v, 2) for y, v in in_union.items()})
    return dict(terms=res,
                note=("An analyst querying the field literally named 'imaging "
                      "findings' would see every descriptor collapse to 0% in "
                      "Dec 2019 while the same term persists in the union text."))


def r4_length(df):
    d = df[df.year.between(2016, 2025)].copy()
    can = canonicalise(d["diag_s"])
    d = d.assign(canlen=can.str.len())
    g = d.groupby("year")
    raw = g["dlen"].median(); canon = g["canlen"].median()
    out = {int(y): dict(raw_median=float(raw[y]), canonical_median=float(canon[y]))
           for y in raw.index}
    pre = [2016, 2017, 2018, 2019]; post = [2022, 2023, 2024, 2025]
    raw_growth = float(raw.loc[post].mean()-raw.loc[pre].mean())
    can_growth = float(canon.loc[post].mean()-canon.loc[pre].mean())
    return dict(by_year=out, raw_growth_pre_to_post=raw_growth,
                canonical_growth_pre_to_post=can_growth,
                boilerplate_fraction_of_growth=round(1-can_growth/raw_growth, 3) if raw_growth else None)


if __name__ == "__main__":
    df = load()
    R = {}
    print("R1 decomposition"); R["decomposition"] = r1_decomposition(df)
    print("R2 placebo"); R["placebo_changepoint"] = r2_placebo(df)
    print("R3 rwe artifact"); R["rwe_artifact"] = r3_rwe_artifact(df)
    print("R4 length"); R["length_boilerplate"] = r4_length(df)
    json.dump(R, open(RES/"reviewer_results.json", "w"), ensure_ascii=False, indent=2)
    d = R["decomposition"]
    print(f"  content-only AUC {d['content_only']:.3f} | structure-only {d['structure_only']:.3f} | combined {d['combined']:.3f}")
    p = R["placebo_changepoint"]
    print(f"  placebo: true-transition AUC {p['true_transition_auc']:.3f} vs max placebo {p['max_placebo_auc']:.3f}")
    print("  placebo curve:", [(r['split_year'], r['auc']) for r in p['by_split']])
    l = R["length_boilerplate"]
    print(f"  length raw growth {l['raw_growth_pre_to_post']:.1f} ch vs canonical {l['canonical_growth_pre_to_post']:.1f} ch; boilerplate frac {l['boilerplate_fraction_of_growth']}")
    print("  RWE artifact (low-density in findings field by year):")
    print("   ", R["rwe_artifact"]["terms"]["low_density 低密度"]["findings_field"])

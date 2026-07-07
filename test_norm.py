import numpy as np, pandas as pd, re
from run_core import load, build_sample, split_groups, fit_text
df = load()
s = build_sample(df,[2018,2019],[2020,2021]); tr,te = split_groups(s)

variants = {
 'raw': lambda x: x,
 'strip_markers': lambda x: x.str.replace(r'急诊报告|急诊|见[^。]{0,8}(报告|检查|片)|报告',' ',regex=True),
 'markers+digits+punct': lambda x: x.str.replace(r'急诊报告|急诊|见[^。]{0,8}(报告|检查|片)|报告|[0-9]+|[，。；：、？！,.;:?!（）()【】\[\]<>《》"\'/\\-]',' ',regex=True),
}
for name,fn in variants.items():
    m,_,_ = fit_text(fn(tr['diag_s']), tr['y'], fn(te['diag_s']), te['y'], te['reg'].values)
    print(f'  {name:24s} AUC {m["roc_auc"]:.3f}  balacc {m["balanced_accuracy"]:.3f}')
# confirm clinical terms survive aggressive normalisation
can = df['both'].str.replace(r'急诊报告|急诊|见[^。]{0,8}(报告|检查|片)|报告|[0-9]+|[，。；：、？！,.;:?!（）()【】\[\]<>《》"\'/\\-]',' ',regex=True)
for pat in [r'出血',r'梗死',r'占位',r'骨折']:
    print(f'   term {pat}: before {df["both"].str.contains(pat).mean()*100:.1f}% after {can.str.contains(pat).mean()*100:.1f}%')

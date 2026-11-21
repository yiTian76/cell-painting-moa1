"""Generate Chinese report figures from the released dataset and prediction tables."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


TEAL = "#116B6B"
RUST = "#AE542E"
INK = "#203744"
GRAY = "#DDE5E5"


def style(font_path):
    if font_path:
        font_manager.fontManager.addfont(font_path)
        names = [font_manager.FontProperties(fname=font_path).get_name()]
    else:
        names = ["Microsoft YaHei", "Noto Sans CJK SC", "SimHei", "Arial Unicode MS"]
    plt.rcParams.update({"font.family": ["sans-serif"], "font.sans-serif": names + ["DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10.5,
                         "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
                         "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "axes.edgecolor": GRAY, "axes.axisbelow": True})


def finish(fig, out, name):
    fig.savefig(out / (name + ".png"), dpi=220, facecolor="white", bbox_inches="tight", pad_inches=.12)
    plt.close(fig)


def compare(results, out):
    primary = pd.read_csv(results / "metrics.csv").set_index("method")
    extra = pd.read_csv(results / "supplement_metrics.csv").set_index("model")
    rows = [(s, primary.loc[c]) for s,c in [
        ("多数类", "majority"), ("全特征类别均值", "flat_mean_cosine"),
        ("通道等权主模型", "channel_balanced"), ("余弦最近邻", "cosine_1nn")]]
    rows.append(("类别平衡岭分类器", extra.loc["ridge_balanced_10uM"]))
    fig, ax = plt.subplots(figsize=(8.0, 3.45), layout="constrained")
    pos = np.arange(len(rows))
    for dx, key, clr, label in [(-.18,"accuracy",TEAL,"首选命中率"),(.18,"top3_accuracy","#849DB0","前三命中率")]:
        values = np.array([r[key]*100 for _,r in rows])
        ax.barh(pos+dx,values,height=.31,color=clr,label=label)
        for y,x in zip(pos+dx,values):
            ax.text(x+.27,y,f"{x:.1f}%",va="center",fontsize=9)
    ax.set(yticks=pos, yticklabels=[r[0] for r in rows], xlabel="折外化合物命中率（%）",xlim=(0,30))
    ax.invert_yaxis()
    ax.tick_params(axis="y",length=0)
    ax.grid(axis="x",color=GRAY)
    ax.legend(loc="lower left", bbox_to_anchor=(0,1.01), ncol=2, frameon=False)
    finish(fig, out, "model_comparison")


def ablation(results, out):
    labels = {"AGP":"AGP（肌动蛋白、高尔基体、质膜）", "DNA":"DNA（细胞核）",
              "ER":"ER（内质网）","Mito":"Mito（线粒体）","RNA":"RNA（核仁及细胞质 RNA）",
              "cross_channel":"跨通道测量","shape_other":"形状及其他测量"}
    data = pd.read_csv(results / "ablation.csv").sort_values("macro_f1_full_minus_omitted",ascending=False)
    x = 100*data.macro_f1_full_minus_omitted
    fig, ax = plt.subplots(figsize=(8.2, 2.78), layout="constrained")
    pos = np.arange(len(data))
    ax.barh(pos, x, height=.49, color=[TEAL if v>=0 else RUST for v in x])
    for y,v in zip(pos,x):
        ax.text(v + (.035 if v>=0 else -.035), y, f"{v:+.2f}", va="center",
                ha="left" if v>=0 else "right", fontsize=10)
    ax.axvline(0, color=INK, lw=.8)
    ax.set(yticks=pos, yticklabels=[labels[s] for s in data.omitted_block],
           xlabel="完整模型减去移除该组后的宏 F1（百分点）",xlim=(-1.48,1.68))
    ax.tick_params(axis="y",length=0,pad=8)
    ax.grid(axis="x",color=GRAY,linewidth=.65)
    ax.invert_yaxis()
    finish(fig,out,"channel_ablation")


def profiles(data, results, out):
    groups = [("CDK 抑制剂","CDK inhibitor"),("MEK 抑制剂","MEK inhibitor"),
              ("糖皮质激素\n受体激动剂","glucocorticoid receptor agonist"),
              ("磷酸二酯酶\n抑制剂","phosphodiesterase inhibitor")]
    features = [("细胞核面积","Nuclei_AreaShape_Area"),("细胞质线粒体通道平均强度","Cytoplasm_Intensity_MeanIntensity_Mito")]
    cols = ["compound_id","moa"]+[f for _,f in features]
    d = pd.read_csv(data,usecols=cols).set_index("compound_id")
    folds = pd.read_csv(results / "folds.csv")
    d = d.loc[folds.compound_id]
    fig, axes = plt.subplots(1,2,figsize=(8.35,3.28),layout="constrained")
    for ax,(title,feat) in zip(axes,features):
        rng = np.random.default_rng(42)
        for i,(_,grp) in enumerate(groups):
            values = d.loc[d.moa==grp, feat].dropna()
            xx = i+rng.uniform(-.16,.16,len(values))
            ax.scatter(xx,values,s=13,alpha=.76,color="#5D9B99",edgecolors="none")
            ax.hlines(values.median(),i-.23,i+.23,color=INK,linewidth=2.1)
        ax.axhline(0,color="#76848B",ls=":",lw=.9)
        ax.set(xticks=range(4),xticklabels=[f"{title_}\n(n={int((d.moa==grp).sum())})" for title_,grp in groups],
               xlim=(-.55,3.55))
        ax.set_title(title,fontsize=12,loc="left",fontweight="bold")
        ax.tick_params(axis="x",length=0,labelsize=9)
        ax.grid(axis="y",color=GRAY,linewidth=.65)
    axes[0].set_ylabel("相对同板 DMSO 的稳健标准化值")
    finish(fig,out,"class_morphology")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,required=True)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--font",help="Optional local Chinese font file")
    a=parser.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    style(a.font)
    compare(a.results,a.out)
    ablation(a.results,a.out)
    profiles(a.data,a.results,a.out)

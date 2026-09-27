"""Export the dashboard data (reanalysis/web/data/netflix.json) and the README figures.

    python reanalysis/python/build_web.py      # after run_analysis.py
Only derived numbers leave the machine: no rating, user or date from the Netflix files is exported.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW, WORK = ROOT / "data" / "raw", ROOT / "data" / "work"
RES, WEB, FIG = ROOT / "reanalysis" / "results", ROOT / "reanalysis" / "web" / "data", ROOT / "reanalysis" / "docs"
MIN_RATINGS, TOP = 1000, 10          # films in the "similar films" app, neighbours per film


def titles() -> pd.DataFrame:
    rows = []
    for line in (RAW / "movie_titles.csv").read_bytes().decode("latin-1").splitlines():
        mid, year, title = line.split(",", 2)
        rows.append((int(mid), None if year == "NULL" else int(year), title.strip()))
    return pd.DataFrame(rows, columns=["id", "year", "title"]).set_index("id")


def main() -> None:
    WEB.mkdir(parents=True, exist_ok=True); FIG.mkdir(exist_ok=True)
    m = json.loads((RES / "metrics.json").read_text())
    f = np.load(WORK / "item_factors.npz")
    t = titles()
    keep = np.flatnonzero(f["n"] >= MIN_RATINGS)
    V = f["Q"][keep]
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    S = V @ V.T
    np.fill_diagonal(S, -1)
    nb = np.argsort(-S, axis=1)[:, :TOP]
    films = {"t": [t.at[i, "title"] for i in keep], "y": [None if pd.isna(t.at[i, "year"]) else int(t.at[i, "year"]) for i in keep],
             "n": f["n"][keep].astype(int).tolist(), "m": f["mean"][keep].round(2).tolist(),
             "nb": [[[int(j), round(float(S[a, j]), 3)] for j in row] for a, row in enumerate(nb)]}
    out = {"generated": pd.Timestamp.now().strftime("%Y-%m-%d"), "metrics": m, "films": films,
           "curve": pd.read_csv(RES / "learning_curve.csv").to_dict("records"),
           "activity": pd.read_csv(RES / "activity.csv").to_dict("records"),
           "monthly": pd.read_csv(RES / "monthly.csv").values.tolist()}
    (WEB / "netflix.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print("netflix.json", round((WEB / "netflix.json").stat().st_size / 1e6, 2), "MB,", len(keep), "films")

    # README figure: the same models on the two splits, against the Prize's own yardsticks
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    r = pd.DataFrame(m["results"]).set_index("split")
    models = [("global_mean", "Global mean"), ("movie_mean", "Movie mean"), ("biases", "User + movie biases"), ("mf", "Matrix factorisation (k=50)")]
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    y = np.arange(len(models))
    ax.barh(y - 0.19, [r.at["probe", k] for k, _ in models], height=0.36, color="#b0272f", label="official probe (1.41 M newest ratings)")
    ax.barh(y + 0.19, [r.at["random", k] for k, _ in models], height=0.36, color="#c9b8b0", label="random 20 % of all ratings (2025 design)")
    for k, (key, _) in enumerate(models):
        for dy, s in ((-0.19, "probe"), (0.19, "random")):
            ax.text(r.at[s, key] + 0.004, k + dy, f"{r.at[s, key]:.3f}", va="center", fontsize=8.5)
    ref = m["reference"]
    for v, lab, ha in ((ref["cinematch_quiz"], "Cinematch 0.951 (quiz)", "left"), (ref["grand_prize_test"], "Grand Prize 0.857 (test)", "right")):
        ax.axvline(v, color="#333", lw=0.8, ls="--")
        ax.text(v + (0.004 if ha == "left" else -0.004), -0.72, lab, fontsize=8, ha=ha, va="center")
    ax.set_yticks(y, [l for _, l in models]); ax.set_ylim(len(models) - 0.45, -0.95)
    ax.set_xlim(0.75, 1.2); ax.set_xlabel("RMSE (lower is better)")
    fig.legend(frameon=False, fontsize=8.5, loc="lower center", ncol=2)
    ax.set_title("Same models, two splits: the random split flatters all four", loc="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(FIG / "rmse_two_splits.png", dpi=160); plt.close(fig)

    mo = pd.read_csv(RES / "monthly.csv")
    mo = mo[mo["n"] >= 10_000]
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    ax.plot(pd.to_datetime(mo["month"]), mo["mean_rating"], color="#b0272f", lw=1.8)
    ax.set_ylabel("mean rating"); ax.set_title("Monthly mean rating (months with ≥ 10,000 ratings)", loc="left", fontsize=10.5)
    fig.tight_layout(); fig.savefig(FIG / "monthly_mean.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()

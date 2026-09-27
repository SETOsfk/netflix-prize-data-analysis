"""How good is "0.81 RMSE"? Every model on the official probe split and on the random 80/20 split the 2025 team used.

    python reanalysis/python/run_analysis.py      # after prepare.py; ~25 min (two 20-epoch SGD runs)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

ROOT = Path(__file__).resolve().parents[2]
WORK, OUT = ROOT / "data" / "work", ROOT / "reanalysis" / "results"
OUT.mkdir(exist_ok=True)
K, LR, REG, REG_B, EPOCHS = 50, 0.005, 0.02, 0.02, 20     # textbook defaults (Funk / Surprise SVD); never tuned on a test set
LAMBDA_I, LAMBDA_U = 25, 10                               # shrinkage of the bias baseline (Koren 2008)
BUCKETS = [(1, 10), (11, 50), (51, 200), (201, 1000), (1001, 10**6)]


@njit(cache=True)
def sgd_epoch(u, i, r, mu, bu, bi, P, Q):
    sse = 0.0
    for t in range(len(r)):
        a, b = u[t], i[t]
        pred = mu + bu[a] + bi[b]
        for f in range(P.shape[1]):
            pred += P[a, f] * Q[b, f]
        e = np.float64(r[t]) - pred
        sse += e * e
        bu[a] += LR * (e - REG_B * bu[a])
        bi[b] += LR * (e - REG_B * bi[b])
        for f in range(P.shape[1]):
            pf, qf = P[a, f], Q[b, f]
            P[a, f] += LR * (e * qf - REG * pf)
            Q[b, f] += LR * (e * pf - REG * qf)
    return np.sqrt(sse / len(r))


@njit(cache=True)
def predict(u, i, mu, bu, bi, P, Q):
    out = np.empty(len(u))
    for t in range(len(u)):
        pred = mu + bu[u[t]] + bi[i[t]]
        for f in range(P.shape[1]):
            pred += P[u[t], f] * Q[i[t], f]
        out[t] = min(5.0, max(1.0, pred))
    return out


def load_bin(name):                                      # kept compact (int32 / int16 / int8): ~0.7 GB per 100 M ratings
    return (np.fromfile(WORK / f"{name}_u.bin", "<i4"), np.fromfile(WORK / f"{name}_i.bin", "<i2"),
            np.fromfile(WORK / f"{name}_r.bin", "<i1"))


rmse = lambda y, p: float(np.sqrt(np.mean((y - p) ** 2)))


def baselines(u, i, r, n_users):
    """Global mean, movie mean and the shrunken bias baseline mu + b_u + b_i, fitted on training arrays."""
    mu = r.mean(dtype=np.float64)
    ni = np.bincount(i, minlength=17771)
    movie_mean = np.where(ni > 0, np.bincount(i, r.astype(np.float64), 17771) / np.maximum(ni, 1), mu)
    bi = np.bincount(i, r - mu, 17771) / (LAMBDA_I + ni)
    nu = np.bincount(u, minlength=n_users)
    bu = np.bincount(u, r - mu - bi[i], n_users) / (LAMBDA_U + nu)
    return mu, movie_mean, bi, bu, nu


def run_split(name: str, n_users: int, last_train, test_dates, curve):
    u, i, r = load_bin(f"{name}_train")
    tu, ti, tr = load_bin(f"{name}_test")
    mu, movie_mean, bi0, bu0, nu = baselines(u, i, r, n_users)
    base = np.clip(mu + bu0[tu] + bi0[ti], 1, 5)
    res = {"split": name, "n_train": int(len(r)), "n_test": int(len(tr)),
           "global_mean": rmse(tr, mu), "movie_mean": rmse(tr, movie_mean[ti]), "biases": rmse(tr, base)}
    P = np.fromfile(WORK / "P0.bin", "<f8").reshape(n_users, K)
    Q = np.fromfile(WORK / "Q0.bin", "<f8").reshape(17771, K)
    bu, bi = np.zeros(n_users), np.zeros(17771)
    for ep in range(1, EPOCHS + 1):
        t = time.time()
        train_rmse = sgd_epoch(u, i, r, mu, bu, bi, P, Q)
        test_rmse = rmse(tr, predict(tu, ti, mu, bu, bi, P, Q))
        curve.append({"split": name, "epoch": ep, "train_rmse": round(train_rmse, 5), "test_rmse": round(test_rmse, 5)})
        print(f"{name} epoch {ep:2d}: train {train_rmse:.4f} test {test_rmse:.4f} ({time.time() - t:.0f}s)", flush=True)
    pred = predict(tu, ti, mu, bu, bi, P, Q)
    res["mf"] = rmse(tr, pred)
    # who is being tested: how active is the user, and is the test rating newer than all of their training ratings?
    res["test_user_median_train_ratings"] = float(np.median(nu[tu]))
    res["test_share_users_le_50"] = float(np.mean(nu[tu] <= 50))
    res["test_share_after_last_train_rating"] = float(np.mean(test_dates > last_train[tu]))
    act = []
    for lo, hi in BUCKETS:
        m = (nu[tu] >= lo) & (nu[tu] <= hi)
        act.append({"split": name, "bucket": f"{lo}-{hi}" if hi < 10**6 else f"{lo}+", "n_test": int(m.sum()),
                    "biases": round(rmse(tr[m], base[m]), 5), "mf": round(rmse(tr[m], pred[m]), 5)})
    res = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in res.items()}
    return res, act, (mu, bu, bi, P, Q)


def main() -> None:
    d = np.load(WORK / "ratings.npz")
    info = eval((WORK / "prepare_info.txt").read_text())
    n_users = info["n_users"]
    user, rating, date = d["user"], d["rating"], d["date"]
    # monthly volume and mean rating (the early-2004 jump is a known quirk of this data)
    month = (date.astype(np.int64) + np.datetime64("1998-01-01", "D")).astype("datetime64[M]")
    mm, inv = np.unique(month, return_inverse=True)
    monthly = pd.DataFrame({"month": mm.astype(str), "n": np.bincount(inv),
                            "mean_rating": np.bincount(inv, rating) / np.bincount(inv)}).round(4)
    monthly.to_csv(OUT / "monthly.csv", index=False)
    dist = np.bincount(rating, minlength=6)[1:]

    curve, rows, acts = [], [], []
    for name, mask in (("probe", d["probe"]), ("random", d["rand_test"])):
        last = np.full(n_users, -32768, np.int32)
        np.maximum.at(last, user[~mask], date[~mask])
        res, act, model = run_split(name, n_users, last, date[mask], curve)
        rows.append(res); acts += act
        if name == "probe":                                   # item factors for the "similar movies" app
            mu, bu, bi, P, Q = model
            np.savez(WORK / "item_factors.npz", Q=Q, bi=bi, mu=mu,
                     n=np.bincount(d["movie"][~mask], minlength=17771),
                     mean=np.bincount(d["movie"][~mask], rating[~mask], 17771) / np.maximum(np.bincount(d["movie"][~mask], minlength=17771), 1))
        pd.DataFrame(rows).to_csv(OUT / "rmse.csv", index=False)       # write as we go: a long run
    pd.DataFrame(curve).to_csv(OUT / "learning_curve.csv", index=False)
    pd.DataFrame(acts).to_csv(OUT / "activity.csv", index=False)
    metrics = {**info, "rating_share": (dist / dist.sum()).round(4).tolist(),
               "first_date": str(np.datetime64("1998-01-01") + int(date.min())), "last_date": str(np.datetime64("1998-01-01") + int(date.max())),
               "hyper": {"k": K, "lr": LR, "reg": REG, "reg_bias": REG_B, "epochs": EPOCHS, "lambda_i": LAMBDA_I, "lambda_u": LAMBDA_U},
               "results": rows,
               "reference": {"movie_mean_quiz": 1.0540, "cinematch_quiz": 0.9514, "cinematch_test": 0.9525, "grand_prize_test": 0.8567}}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()

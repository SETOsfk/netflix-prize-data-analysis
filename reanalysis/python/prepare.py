"""Parse the Netflix Prize files once and build both evaluation splits.

    python reanalysis/python/prepare.py      # ~3 min, ~4 GB RAM; reads data/raw/, writes data/work/

Input (Kaggle netflix-inc/netflix-prize-data, .txt or .txt.gz): combined_data_1..4, probe.txt, movie_titles.csv.
Output: data/work/ratings.npz and, per split, the training/test arrays in the exact order both languages train on
(little-endian int32 user, int16 movie, int8 rating) plus the shared initial factors — so R/run_analysis.R
replays the very same SGD.
"""
from __future__ import annotations

import gzip
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[2]
RAW, WORK = ROOT / "data" / "raw", ROOT / "data" / "work"
SEED, K = 20260927, 50
N_RATINGS = 100_480_507


def read_bytes(stem: str) -> np.ndarray:
    for p in (RAW / stem, RAW / (stem + ".gz")):
        if p.exists():
            opener = gzip.open if p.suffix == ".gz" else open
            with opener(p, "rb") as f:
                return np.frombuffer(f.read(), dtype=np.uint8)
    sys.exit(f"missing {RAW / stem} — see data/README.md")


@njit(cache=True)
def days(y, m, d):                                     # days since 1998-01-01 (civil-from-days algorithm)
    y -= m <= 2
    era = y // 400
    yoe = y - era * 400
    doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    return era * 146097 + yoe * 365 + yoe // 4 - yoe // 100 + doy - 719468 - 10227


@njit(cache=True)
def parse(buf, users, movies, ratings, dates, n):
    i, movie = 0, 0
    while i < len(buf):
        v = 0
        while i < len(buf) and 48 <= buf[i] <= 57:
            v = v * 10 + buf[i] - 48
            i += 1
        if i >= len(buf):
            break
        if buf[i] == 58:                               # "123:" starts a movie block
            movie = v
            i += 1
        else:                                          # "user,rating,YYYY-MM-DD"
            ratings[n] = buf[i + 1] - 48
            i += 3
            y = (buf[i] - 48) * 1000 + (buf[i + 1] - 48) * 100 + (buf[i + 2] - 48) * 10 + buf[i + 3] - 48
            dates[n] = days(y, (buf[i + 5] - 48) * 10 + buf[i + 6] - 48, (buf[i + 8] - 48) * 10 + buf[i + 9] - 48)
            users[n], movies[n] = v, movie
            n += 1
            i += 10
        while i < len(buf) and (buf[i] == 10 or buf[i] == 13):
            i += 1
    return n


@njit(cache=True)
def parse_probe(buf, movies, users):
    i, movie, n = 0, 0, 0
    while i < len(buf):
        v = 0
        while i < len(buf) and 48 <= buf[i] <= 57:
            v = v * 10 + buf[i] - 48
            i += 1
        if i < len(buf) and buf[i] == 58:
            movie = v
            i += 1
        elif v > 0:
            movies[n], users[n] = movie, v
            n += 1
        while i < len(buf) and (buf[i] == 10 or buf[i] == 13):
            i += 1
    return n


def export(name: str, idx: np.ndarray, u, m, r) -> None:
    for arr, tag in ((u[idx], "u"), (m[idx], "i"), (r[idx], "r")):
        arr.astype(arr.dtype.newbyteorder("<")).tofile(WORK / f"{name}_{tag}.bin")


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    users = np.empty(N_RATINGS + 16, np.int32)
    movies, ratings, dates = np.empty_like(users, np.int16), np.empty_like(users, np.int8), np.empty_like(users, np.int16)
    n = 0
    for k in range(1, 5):
        n = parse(read_bytes(f"combined_data_{k}.txt"), users, movies, ratings, dates, n)
        print(f"combined_data_{k}: {n:,} ratings ({time.time() - t0:.0f}s)", flush=True)
    users, movies, ratings, dates = users[:n], movies[:n], ratings[:n], dates[:n]

    raw_ids, uidx = np.unique(users, return_inverse=True)            # 480,189 users → 0..U-1
    uidx = uidx.astype(np.int32)
    key = movies.astype(np.int64) * 3_000_000 + users
    order = np.argsort(key, kind="stable")
    key = key[order]                                                  # sorted keys
    dup = int((np.diff(key) == 0).sum())
    pb = read_bytes("probe.txt")
    pm, pu = np.empty(2_000_000, np.int64), np.empty(2_000_000, np.int64)
    k = parse_probe(pb, pm, pu)
    pkey = pm[:k] * 3_000_000 + pu[:k]
    pos = np.searchsorted(key, pkey)
    found = key[np.minimum(pos, n - 1)] == pkey
    probe = np.zeros(n, bool)
    probe[order[pos[found]]] = True
    del key, order

    rng = np.random.default_rng(SEED)
    rand_test = rng.random(n) < 0.2                                  # the 2025 team design: random 80/20 of all ratings
    for name, test in (("probe", probe), ("random", rand_test)):
        tr = np.flatnonzero(~test)
        export(f"{name}_train", rng.permutation(tr), uidx, movies, ratings)   # SGD order, shared with R
        export(f"{name}_test", np.flatnonzero(test), uidx, movies, ratings)
    rng.normal(0, 0.1, (len(raw_ids), K)).astype("<f8").tofile(WORK / "P0.bin")
    rng.normal(0, 0.1, (17771, K)).astype("<f8").tofile(WORK / "Q0.bin")

    np.savez(WORK / "ratings.npz", user=uidx, movie=movies, rating=ratings, date=dates, probe=probe, rand_test=rand_test)
    info = {"n_ratings": int(n), "n_users": int(len(raw_ids)), "n_movies": int(np.unique(movies).size),
            "duplicate_user_movie": dup, "probe_listed": int(k), "probe_matched": int(found.sum())}
    (WORK / "prepare_info.txt").write_text(repr(info))
    print(info, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

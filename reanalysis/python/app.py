"""Streamlit version of the re-analysis page.   streamlit run reanalysis/python/app.py
Reads reanalysis/web/data/netflix.json (built by build_web.py): aggregates and film neighbours only."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="Netflix Prize, revisited", page_icon="🎬", layout="wide")


@st.cache_data
def load():
    return json.loads((ROOT / "web" / "data" / "netflix.json").read_text())


D = load()
tr = st.sidebar.radio("Dil / Language", ["Türkçe", "English"]) == "Türkçe"
T = (lambda a, b: a if tr else b)
res = pd.DataFrame(D["metrics"]["results"]).set_index("split")
F = D["films"]

st.title(T("\"0,81 RMSE\" gerçekte ne kadar iyiydi?", "How good was \"0.81 RMSE\" really?"))
c1, c2, c3 = st.columns(3)
c1.metric(T("Resmî probe (MF, k=50)", "Official probe (MF, k=50)"), f"{res.at['probe', 'mf']:.3f}", T("Cinematch 0,9514 · ödül 0,8567", "Cinematch 0.9514 · prize 0.8567"), delta_color="off", delta_arrow="off")
c2.metric(T("Aynı model, rastgele %20", "Same model, random 20%"), f"{res.at['random', 'mf']:.3f}", T("2025'teki 0,81'in tasarımı", "the design behind 2025's 0.81"), delta_color="off", delta_arrow="off")
c3.metric(T("Puan / kullanıcı / film", "Ratings / users / films"), f"{D['metrics']['n_ratings'] / 1e6:.2f} M", f"{D['metrics']['n_users']:,} · {D['metrics']['n_movies']:,}", delta_color="off", delta_arrow="off")

tab1, tab2 = st.tabs([T("Benzer filmler", "Similar films"), T("Sonuçlar", "Results")])
with tab1:
    order = sorted(range(len(F["t"])), key=lambda i: -F["n"][i])
    default = next((k for k, i in enumerate(order) if "Fellowship of the Ring" in F["t"][i]), 0)
    i = st.selectbox(T("Film", "Film"), order, index=default, format_func=lambda i: f"{F['t'][i]} ({F['y'][i]})")
    st.caption(T(f"{F['n'][i]:,} puan · ortalama {F['m'][i]:.2f}", f"{F['n'][i]:,} ratings · mean {F['m'][i]:.2f}"))
    st.dataframe([{T("Film", "Film"): F["t"][j], T("Yıl", "Year"): F["y"][j], T("Benzerlik", "Similarity"): f"{s:.0%}",
                   T("Puan sayısı", "Ratings"): F["n"][j], T("Ortalama", "Mean"): F["m"][j]} for j, s in F["nb"][i]], hide_index=True, width="stretch")
with tab2:
    names = {"global_mean": T("Genel ortalama", "Global mean"), "movie_mean": T("Film ortalaması", "Movie mean"),
             "biases": T("Kullanıcı + film sapmaları", "User + movie biases"), "mf": T("Matris ayrıştırma (k=50)", "Matrix factorisation (k=50)")}
    tbl = res[list(names)].T.rename(index=names).rename(columns={"probe": T("probe (resmî)", "probe (official)"), "random": T("rastgele %20", "random 20%")})
    st.dataframe(tbl, width="stretch")
    st.bar_chart(tbl, horizontal=True, stack=False)
    cur = pd.DataFrame(D["curve"]).pivot(index="epoch", columns="split", values="test_rmse")
    st.markdown("#### " + T("Öğrenme eğrisi (test RMSE)", "Learning curve (test RMSE)"))
    st.line_chart(cur)
    mo = pd.DataFrame(D["monthly"], columns=["month", "n", "mean"]).query("n >= 10000").set_index("month")
    st.markdown("#### " + T("Aylık ortalama puan", "Monthly mean rating"))
    st.line_chart(mo["mean"])
st.caption(T("Netflix Prize verisi: yalnız araştırma amaçlı, yeniden dağıtılamaz. Bu uygulama yalnız toplu sonuçları gösterir; Netflix ile bağı yoktur.",
             "Netflix Prize data: research use only, no redistribution. This app shows aggregates only; not affiliated with Netflix."))

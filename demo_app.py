"""Storefront demonstration of item cold-start recommendation.

Both models score every product in the catalogue once per customer, computed
ahead of time. Every view here filters and sorts those stored scores, so the
rankings are produced by the models rather than arranged in advance.

Newly listed products carry no rating and no reviews because their interaction
history is withheld by construction.
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

DATA = Path(__file__).parent / "demo" / "demo_data.json"

st.set_page_config(page_title="voltix.", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
  #MainMenu, footer, header {visibility:hidden;}
  .stApp {background:#eef2f6;}
  .block-container {padding-top:1rem; max-width:1500px;}

  .nav {background:linear-gradient(100deg,#0b1220,#1e293b);border-radius:14px;
        padding:1rem 1.5rem;margin-bottom:1.1rem;display:flex;
        justify-content:space-between;align-items:center;}
  .nav .logo{color:#fff;font-size:1.5rem;font-weight:700;letter-spacing:-.6px;}
  .nav .logo span{color:#38bdf8;}
  .nav .acct{color:#cbd5e1;font-size:.83rem;}

  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(185px,1fr));
        gap:.9rem;margin:.4rem 0 .3rem;}
  .card{background:#fff;border-radius:12px;overflow:hidden;border:1px solid #e2e8f0;
        display:flex;flex-direction:column;position:relative;}
  .card.new{border:1px solid #10b981;box-shadow:0 0 0 3px rgba(16,185,129,.10);}
  .rank{position:absolute;top:.5rem;left:.5rem;background:#0f172a;color:#fff;
        font-size:.65rem;font-weight:700;padding:.18rem .42rem;border-radius:5px;}
  .imgwrap{height:150px;display:flex;align-items:center;justify-content:center;
           padding:.8rem;background:#fff;}
  .imgwrap img{max-height:100%;max-width:100%;object-fit:contain;}
  .body{padding:.6rem .75rem .9rem;flex:1;display:flex;flex-direction:column;gap:.25rem;}
  .name{font-size:.78rem;line-height:1.35;color:#0f172a;font-weight:500;
        display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;
        overflow:hidden;}
  .cat{font-size:.67rem;color:#94a3b8;text-transform:capitalize;}
  .stars{color:#f59e0b;font-size:.78rem;}
  .rate{color:#475569;font-size:.73rem;}
  .norate{color:#94a3b8;font-size:.71rem;font-style:italic;}
  .badge{display:inline-block;background:#10b981;color:#fff;font-size:.58rem;
         font-weight:700;padding:.15rem .42rem;border-radius:4px;letter-spacing:.6px;
         width:fit-content;}
  .score{font-size:.68rem;color:#2563eb;font-weight:600;}

  .pdp{background:#fff;border-radius:14px;padding:1.5rem;border:1px solid #e2e8f0;}
  .pdp h2{font-size:1.3rem;color:#0f172a;margin:.4rem 0 .5rem;font-weight:600;}
  .pdp .desc{color:#475569;font-size:.85rem;line-height:1.6;margin-top:.8rem;}
  .imgbox{background:#fff;border:1px solid #e2e8f0;border-radius:12px;height:330px;
          display:flex;align-items:center;justify-content:center;padding:1rem;}
  .imgbox img{max-height:100%;max-width:100%;object-fit:contain;}

  .ok{background:#ecfdf5;border-left:3px solid #10b981;padding:.85rem 1rem;
      border-radius:7px;color:#065f46;font-size:.83rem;margin-top:.9rem;}
  .bad{background:#fef2f2;border-left:3px solid #ef4444;padding:.85rem 1rem;
       border-radius:7px;color:#991b1b;font-size:.83rem;margin-top:.9rem;}
  .sect{font-size:1.08rem;font-weight:600;color:#0f172a;margin:1.3rem 0 .2rem;}
  .sub{color:#64748b;font-size:.82rem;margin-bottom:.6rem;}
  .stat{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:.85rem 1rem;}
  .stat .n{font-size:1.45rem;font-weight:700;color:#0f172a;}
  .stat .l{font-size:.73rem;color:#64748b;}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load():
    return json.loads(DATA.read_text())


data = load()
catalogue = data["catalogue"]

st.session_state.setdefault("product", None)
st.session_state.setdefault("customer", 0)


def stars(item):
    if item["is_new"] or item["rating"] is None:
        return '<div class="norate">No ratings yet</div>'
    full = int(item["rating"])
    half = 1 if item["rating"] - full >= 0.5 else 0
    return (f'<div><span class="stars">{"★"*full}{"⯨"*half}'
            f'{"☆"*(5-full-half)}</span> <span class="rate">{item["rating"]} '
            f'({item["reviews"]:,})</span></div>')


def card(item_id, rank=None, score=None):
    item = catalogue[str(item_id)]
    image = (f'<img src="{item["image"]}">' if item["image"]
             else '<span class="norate">no image</span>')
    return (f'<div class="card{" new" if item["is_new"] else ""}">'
            f'{f"<div class=rank>#{rank}</div>" if rank else ""}'
            f'<div class="imgwrap">{image}</div><div class="body">'
            f'{"<span class=badge>NEW LISTING</span>" if item["is_new"] else ""}'
            f'<div class="name">{item["title"]}</div>{stars(item)}'
            f'<div class="cat">{item["domain"]}</div>'
            f'{f"<div class=score>score {score}</div>" if score is not None else ""}'
            f'</div></div>')


def grid(ids, scores=None, ranked=False, per_row=6, tag="g"):
    """Cards render as HTML; a parallel row of buttons handles selection."""
    for start in range(0, len(ids), per_row):
        chunk = ids[start:start + per_row]
        st.markdown('<div class="grid">' + "".join(
            card(i, start + n + 1 if ranked else None,
                 scores[start + n] if scores else None)
            for n, i in enumerate(chunk)) + '</div>', unsafe_allow_html=True)
        for slot, item_id in zip(st.columns(per_row), chunk):
            with slot:
                if st.button("View", key=f"{tag}_{item_id}_{start}",
                             use_container_width=True):
                    st.session_state.product = item_id
                    st.rerun()


def rank_by(model_key, pool):
    """Sort a pool of item ids by the stored score from one model."""
    scores = customer[model_key]
    return sorted(pool, key=lambda i: -scores.get(str(i), -1e9))


customer = data["customers"][st.session_state.customer]
all_ids = [int(k) for k in catalogue]

st.markdown(
    f'<div class="nav"><div class="logo">voltix<span>.</span></div>'
    f'<div class="acct">{customer["label"]} · '
    f'{len(customer["history"])} orders</div></div>', unsafe_allow_html=True)

bar = st.columns([3, 2, 1])
with bar[0]:
    query = st.text_input("s", placeholder="Search the store…",
                          label_visibility="collapsed")
with bar[1]:
    picked = st.selectbox("c", range(len(data["customers"])),
                          format_func=lambda i: data["customers"][i]["label"],
                          index=st.session_state.customer,
                          label_visibility="collapsed")
    if picked != st.session_state.customer:
        st.session_state.customer = picked
        st.session_state.product = None
        st.rerun()
with bar[2]:
    if st.session_state.product is not None and st.button("← Back",
                                                          use_container_width=True):
        st.session_state.product = None
        st.rerun()

model = st.radio("Ranking model", ["Multimodal", "Collaborative filtering"],
                 horizontal=True, label_visibility="collapsed")
key = "scores_multimodal" if model == "Multimodal" else "scores_collaborative"

if query:
    pool = [i for i in all_ids if query.lower() in catalogue[str(i)]["title"].lower()]
    ranked = rank_by(key, pool)[:18]
    new_in_top = sum(catalogue[str(i)]["is_new"] for i in ranked)
    st.markdown(f'<div class="sect">{len(pool)} results for “{query}”</div>'
                f'<div class="sub">Ranked for {customer["label"]} by the '
                f'{model.lower()} model · {new_in_top} new listings in the top '
                f'{len(ranked)}</div>', unsafe_allow_html=True)
    if ranked:
        grid(ranked, scores=[customer[key][str(i)] for i in ranked],
             ranked=True, tag="search")
    else:
        st.info("Nothing matched that search.")

elif st.session_state.product is not None:
    item = catalogue[str(st.session_state.product)]
    idx = str(st.session_state.product)
    left, right = st.columns([1, 1.3])

    with left:
        image = f'<img src="{item["image"]}">' if item["image"] else ""
        st.markdown(f'<div class="pdp"><div class="imgbox">{image}</div></div>',
                    unsafe_allow_html=True)

    with right:
        if item["is_new"]:
            note = ('<div class="ok"><b>Listed today.</b> No purchases, no ratings, '
                    'no reviews. Collaborative filtering has nothing to work from; '
                    'this product is ranked from its photograph and description '
                    'alone.</div>')
        else:
            note = (f'<div class="ok"><b>{item["purchases"]:,} purchases</b> in the '
                    f'training data. Both models can rank this product.</div>')
        st.markdown(
            f'<div class="pdp">'
            f'{"<span class=badge>NEW LISTING</span>" if item["is_new"] else ""}'
            f'<h2>{item["title"]}</h2>{stars(item)}'
            f'<div class="cat">{item["domain"]}</div>'
            f'<div class="desc">{item["description"] or "No description available."}'
            f'</div>{note}</div>', unsafe_allow_html=True)

        # ranks rather than raw scores: the two models produce values on
        # different scales, so only position within a model is comparable
        a, b = st.columns(2)
        a.markdown(f'<div class="stat"><div class="n">'
                   f'#{customer["rank_multimodal"][idx]:,}</div>'
                   f'<div class="l">multimodal rank for this customer, '
                   f'of {data["n_total"]:,}</div></div>', unsafe_allow_html=True)
        b.markdown(f'<div class="stat"><div class="n">'
                   f'#{customer["rank_collaborative"][idx]:,}</div>'
                   f'<div class="l">collaborative rank, '
                   f'of {data["n_total"]:,}</div></div>', unsafe_allow_html=True)

    similar = [i for i in item.get("similar", []) if str(i) in catalogue]
    ranked_similar = rank_by(key, similar)[:6]
    st.markdown(
        f'<div class="sect">Similar products</div>'
        f'<div class="sub">Products matching this one by image and text content, '
        f'then ordered for {customer["label"]} by the {model.lower()} model</div>',
        unsafe_allow_html=True)
    if ranked_similar:
        grid(ranked_similar, ranked=True,
             scores=[f'#{customer["rank_" + key.split("_")[1]][str(i)]:,}'
                     for i in ranked_similar],
             per_row=6, tag="sim")
    else:
        st.info("No similar products available for this item.")

else:
    tabs = st.tabs(["Recommended for you", "New arrivals", "Your orders",
                    "The experiment"])

    with tabs[0]:
        ranked = rank_by(key, all_ids)[:18]
        new_in_top = sum(catalogue[str(i)]["is_new"] for i in ranked)
        st.markdown(
            f'<div class="sect">Recommended for you</div>'
            f'<div class="sub">Every product in the store, ranked by the '
            f'{model.lower()} model · <b>{new_in_top}</b> new listings in the top '
            f'{len(ranked)}</div>', unsafe_allow_html=True)
        if model == "Multimodal":
            st.markdown(
                f'<div class="ok">Best new listing sits at rank '
                f'<b>{customer["best_new_rank_multimodal"]:,}</b> of '
                f'{data["n_total"]:,} products.</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="bad">Best new listing sits at rank '
                f'<b>{customer["best_new_rank_collaborative"]:,}</b> of '
                f'{data["n_total"]:,} products. All {data["n_new"]:,} new listings '
                f'receive <b>{customer["distinct_collaborative"]}</b> distinct '
                f'score.</div>', unsafe_allow_html=True)
        grid(ranked, scores=[customer[key][str(i)] for i in ranked],
             ranked=True, tag="rec")

    with tabs[1]:
        new_only = [i for i in all_ids if catalogue[str(i)]["is_new"]]
        ranked = rank_by(key, new_only)[:18]
        st.markdown(
            f'<div class="sect">New arrivals</div>'
            f'<div class="sub">{data["n_new"]:,} products listed today. None has '
            f'been purchased, so none carries a rating or a review. Ranked by the '
            f'{model.lower()} model.</div>', unsafe_allow_html=True)
        if model != "Multimodal":
            st.markdown('<div class="bad">Every one of these receives an identical '
                        'score, so this ordering is arbitrary.</div>',
                        unsafe_allow_html=True)
        grid(ranked, scores=[customer[key][str(i)] for i in ranked],
             ranked=True, tag="new")

    with tabs[2]:
        st.markdown(f'<div class="sect">Your orders</div>'
                    f'<div class="sub">{len(customer["history"])} purchases · '
                    f'{customer["domain_share"]:.0%} in '
                    f'{customer["dominant_domain"]}</div>', unsafe_allow_html=True)
        grid(customer["history"][:12], tag="hist")

    with tabs[3]:
        st.markdown('<div class="sect">What this demonstrates</div>',
                    unsafe_allow_html=True)
        columns = st.columns(3)
        columns[0].markdown(
            f'<div class="stat"><div class="n">'
            f'{customer["best_new_rank_multimodal"]:,}</div><div class="l">'
            f'best new listing, multimodal rank</div></div>', unsafe_allow_html=True)
        columns[1].markdown(
            f'<div class="stat"><div class="n">'
            f'{customer["best_new_rank_collaborative"]:,}</div><div class="l">'
            f'best new listing, collaborative rank</div></div>',
            unsafe_allow_html=True)
        columns[2].markdown(
            f'<div class="stat"><div class="n">'
            f'{customer["distinct_collaborative"]}</div><div class="l">'
            f'distinct collaborative scores across {data["n_new"]:,} new listings'
            f'</div></div>', unsafe_allow_html=True)

        st.markdown(f"""
Both models scored every one of the {data['n_total']:,} products in the
catalogue for this customer. Each view above filters and sorts those scores; the
orderings are produced by the models.

{data['n_new']:,} of those products were withheld from training entirely. Neither
model has seen a single purchase of any of them, which is why their pages show no
rating and no reviews.

The collaborative model assigns **{customer['distinct_collaborative']} distinct
score** across all {data['n_new']:,} new listings. Its item representations are
learned from purchases, and these products have none, so it cannot tell them
apart — the ordering among them is arbitrary. Its best new listing sits at rank
**{customer['best_new_rank_collaborative']:,}**.

The multimodal model ranks them from the photograph and description, placing its
best new listing at rank **{customer['best_new_rank_multimodal']:,}**.

Across the full evaluation — 25,656 held-out interactions, five training seeds —
the collaborative approaches score exactly zero on withheld products, while the
content-based models reach roughly three times a random baseline.
""")
        st.caption(f"Multimodal: {data['multimodal_model']} · Collaborative: "
                   f"{data['collaborative_model']} · seed {data['seed']} · "
                   f"generated {data['generated']}")
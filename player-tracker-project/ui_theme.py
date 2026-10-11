"""Look and feel: theme CSS and small HTML components used by the app.

Every function that builds HTML returns a string with no blank lines or indentation, because
Streamlit's markdown would otherwise treat indented lines as code.
"""
import html

import streamlit as st

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{--bg:#0b1220;--surface:#111a2e;--card:#15213a;--line:#233456;--text:#e6edf7;--muted:#8fa1c0;
--accent:#22c55e;--accent-dark:#16a34a;--gold:#f5b942;--silver:#cbd5e1;--bronze:#d08a4e;--amber:#f59e0b;--sky:#38bdf8;}
.stApp, .stApp :not([data-testid="stIconMaterial"]):not(.material-icons){font-family:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif;}
.stApp{background:radial-gradient(1100px 520px at 12% -8%, rgba(34,197,94,.13), transparent 60%),
radial-gradient(900px 420px at 100% 0%, rgba(56,189,248,.08), transparent 60%), var(--bg);}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stToolbar"], .stAppDeployButton, footer, #MainMenu{display:none !important;}
.block-container{max-width:1180px;padding-top:1.4rem;padding-bottom:4rem;}
h1,h2,h3,h4,h5{letter-spacing:-.02em;color:var(--text);}
h3{margin-top:1.8rem;font-weight:700;}
h5{margin:.2rem 0 .6rem 0;font-weight:700;color:var(--muted);text-transform:uppercase;font-size:.78rem;letter-spacing:.08em;}
p, li, label{color:var(--text);}
[data-testid="stCaptionContainer"]{color:var(--muted);}

/* hero */
.hero{position:relative;overflow:hidden;border-radius:22px;padding:34px 38px 32px 38px;margin-bottom:18px;
border:1px solid #1d4b34;background:linear-gradient(135deg,#0e3321 0%,#0d2a30 48%,#0b1220 100%);}
.hero:before{content:"";position:absolute;inset:0;opacity:.35;pointer-events:none;
background:
radial-gradient(circle at 86% 50%, transparent 0 78px, rgba(255,255,255,.55) 79px 81px, transparent 82px),
linear-gradient(90deg, transparent 0 calc(86% - 1px), rgba(255,255,255,.55) calc(86% - 1px) calc(86% + 1px), transparent calc(86% + 1px)),
repeating-linear-gradient(90deg, rgba(255,255,255,.03) 0 60px, transparent 60px 120px);}
.hero h1{position:relative;font-size:2.35rem;line-height:1.15;margin:.5rem 0 .6rem 0;font-weight:800;color:#fff;padding:0;}
.hero h1 .accent{color:var(--accent);}
.hero p{position:relative;max-width:640px;color:#b9c8e0;font-size:1.02rem;margin:0;}
.badges{position:relative;display:flex;gap:8px;flex-wrap:wrap;}
.badge{font-size:.7rem;font-weight:700;letter-spacing:.08em;padding:5px 11px;border-radius:999px;
background:rgba(245,185,66,.16);color:var(--gold);border:1px solid rgba(245,185,66,.4);}
.badge.ghost{background:rgba(255,255,255,.06);color:#c9d6ec;border-color:rgba(255,255,255,.14);}

/* tabs (role-based selectors, so they survive Streamlit's internal class changes) */
[data-testid="stTabs"] [role="tablist"], .stTabs [data-baseweb="tab-list"]{gap:6px !important;background:var(--surface) !important;
padding:6px !important;border-radius:14px !important;border:1px solid var(--line) !important;width:fit-content !important;
margin-bottom:.8rem !important;height:auto !important;box-shadow:none !important;}
[data-testid="stTabs"] [role="tab"], .stTabs [data-baseweb="tab"]{height:42px !important;padding:0 20px !important;border-radius:10px !important;
color:var(--muted) !important;font-weight:600 !important;border:0 !important;box-shadow:none !important;margin:0 !important;}
[data-testid="stTabs"] [role="tab"][aria-selected="true"], .stTabs [data-baseweb="tab"][aria-selected="true"]{
background:rgba(34,197,94,.18) !important;color:#eafff2 !important;}
[data-testid="stTabs"] [role="tab"]:hover{color:#eafff2 !important;}
[data-testid="stTabs"] [role="tablist"]::after, [data-testid="stTabs"] [role="tablist"]::before{display:none !important;}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"]{display:none !important;}

/* step tracker */
.steps{display:flex;gap:10px;margin:.4rem 0 1.1rem 0;flex-wrap:wrap;}
.step{flex:1;min-width:150px;display:flex;align-items:center;gap:10px;padding:11px 14px;border-radius:12px;
background:var(--surface);border:1px solid var(--line);color:var(--muted);font-weight:600;font-size:.9rem;}
.step span{display:inline-flex;align-items:center;justify-content:center;width:26px;height:26px;border-radius:50%;
background:#1b2a47;color:#9fb1d1;font-size:.8rem;font-weight:700;flex:none;}
.step.active{border-color:#2f7d4f;color:#eafff2;background:rgba(34,197,94,.08);}
.step.active span{background:var(--accent);color:#04130a;}
.step.done{color:#bfe9cf;}
.step.done span{background:#14532d;color:#86efac;}

/* cards, inputs, buttons */
[data-testid="stVerticalBlockBorderWrapper"]{background:var(--surface);border:1px solid var(--line) !important;border-radius:18px;}
[data-testid="stExpander"]{border:1px solid var(--line);border-radius:14px;background:var(--card);}
[data-testid="stExpander"] summary{font-weight:600;}
[data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-baseweb="base-input"]{background:var(--card) !important;border-color:var(--line) !important;border-radius:10px !important;}
[data-testid="stFileUploaderDropzone"]{background:linear-gradient(180deg,rgba(34,197,94,.08),rgba(34,197,94,.02));
border:2px dashed #2f7d4f;border-radius:16px;padding:1.5rem 1.4rem;}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--accent);background:rgba(34,197,94,.1);}
button[data-testid="stBaseButton-primary"]{background:linear-gradient(135deg,#22c55e,#16a34a);border:0;color:#04130a;
font-weight:700;border-radius:12px;padding:.62rem 1.5rem;box-shadow:0 8px 22px rgba(34,197,94,.22);transition:all .15s;}
button[data-testid="stBaseButton-primary"]:hover{filter:brightness(1.08);transform:translateY(-1px);color:#04130a;}
button[data-testid="stBaseButton-primary"]:disabled{background:#1b2742;color:#6b7c9c;box-shadow:none;transform:none;}
button[data-testid="stBaseButton-secondary"]{border-radius:12px;border:1px solid var(--line);background:var(--card);font-weight:600;}
button[data-testid="stBaseButton-secondary"]:hover{border-color:var(--accent);color:#eafff2;}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:14px;overflow:hidden;}
[data-testid="stAlert"]{border-radius:14px;}
[data-testid="stVideo"] video{border-radius:14px;border:1px solid var(--line);}

/* stat tiles */
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:.2rem 0 1rem 0;}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:16px 18px;}
.tile b{display:block;font-size:1.9rem;font-weight:800;color:#fff;letter-spacing:-.03em;line-height:1.1;}
.tile small{color:var(--muted);font-weight:500;font-size:.82rem;}

/* top picks */
.picks{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:.4rem 0 1.2rem 0;}
.pick{position:relative;background:linear-gradient(180deg,#16233f,#121c33);border:1px solid var(--line);border-radius:18px;padding:16px;
display:flex;gap:14px;align-items:flex-start;}
.pick.r1{border-color:rgba(245,185,66,.7);box-shadow:0 10px 34px rgba(245,185,66,.12);}
.pick .rank{position:absolute;top:-10px;left:14px;width:30px;height:30px;border-radius:50%;display:flex;align-items:center;
justify-content:center;font-weight:800;font-size:.9rem;color:#1a1200;background:var(--silver);border:2px solid var(--bg);}
.pick.r1 .rank{background:var(--gold);} .pick.r2 .rank{background:var(--silver);} .pick.r3 .rank{background:var(--bronze);}
.photo{width:74px;height:112px;border-radius:12px;object-fit:cover;background:#0e182c;border:1px solid var(--line);flex:none;}
.photo.ph{display:flex;align-items:center;justify-content:center;font-weight:800;color:#6f86ad;font-size:1.3rem;}
.photo.big{width:100%;max-width:120px;height:auto;aspect-ratio:1/1.8;object-fit:cover;}
.pick .body{min-width:0;flex:1;}
.pick .name{font-weight:700;font-size:1.05rem;color:#fff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.pick .sub{color:var(--muted);font-size:.8rem;margin:2px 0 8px 0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.dot{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;vertical-align:baseline;border:1px solid rgba(255,255,255,.35);}
.score{display:flex;align-items:baseline;gap:6px;}
.score .num{font-size:1.9rem;font-weight:800;color:var(--accent);letter-spacing:-.03em;line-height:1;}
.pick.r1 .score .num{color:var(--gold);}
.score .den{color:var(--muted);font-size:.75rem;font-weight:600;}
.bar{height:6px;border-radius:99px;background:#1c2a47;margin:7px 0 10px 0;overflow:hidden;}
.bar > div{height:100%;border-radius:99px;background:linear-gradient(90deg,#16a34a,#22c55e);}
.pick.r1 .bar > div{background:linear-gradient(90deg,#d99a1c,#f5b942);}
.mini{display:flex;gap:14px;}
.mini div{line-height:1.15;} .mini b{display:block;color:#fff;font-size:.95rem;font-weight:700;}
.mini small{color:var(--muted);font-size:.7rem;font-weight:500;}

/* report */
.rhead{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;margin-bottom:6px;}
.rname{font-size:1.5rem;font-weight:800;color:#fff;letter-spacing:-.02em;}
.rsub{color:var(--muted);font-size:.9rem;margin-top:2px;}
.rscore{text-align:right;} .rscore b{font-size:2.3rem;font-weight:800;color:var(--accent);letter-spacing:-.03em;line-height:1;}
.rscore small{display:block;color:var(--muted);font-size:.72rem;font-weight:600;letter-spacing:.06em;text-transform:uppercase;}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0 6px 0;}
.chip{font-size:.8rem;font-weight:600;padding:5px 11px;border-radius:999px;border:1px solid;}
.chip.good{color:#86efac;background:rgba(34,197,94,.12);border-color:rgba(34,197,94,.4);}
.chip.warn{color:#fcd34d;background:rgba(245,158,11,.12);border-color:rgba(245,158,11,.4);}
.chip.neutral{color:#c9d6ec;background:rgba(255,255,255,.05);border-color:rgba(255,255,255,.15);}
.mrow{display:grid;grid-template-columns:230px 1fr 90px;gap:14px;align-items:center;padding:7px 0;}
.mlabel{color:var(--text);font-weight:600;font-size:.88rem;} .mlabel small{display:block;color:var(--muted);font-weight:500;font-size:.72rem;}
.mbar{height:10px;border-radius:99px;background:#1c2a47;overflow:hidden;}
.mbar > div{height:100%;border-radius:99px;}
.mval{text-align:right;color:var(--muted);font-size:.82rem;font-weight:600;} .mval b{color:#fff;}
.swatches{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:.1rem 0 .6rem 0;}
.swatches .lead{color:var(--muted);font-size:.85rem;margin-right:4px;}
.swatches .sw{display:inline-flex;align-items:center;gap:7px;padding:4px 11px;border-radius:999px;background:var(--card);
border:1px solid var(--line);font-size:.82rem;font-weight:600;white-space:nowrap;}
.swatches .sw i{display:inline-block;width:12px;height:12px;border-radius:3px;border:1px solid rgba(255,255,255,.35);}
.how{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:1.2rem 0 .4rem 0;}
.how > div{background:linear-gradient(180deg,#16233f,#121c33);border:1px solid var(--line);border-radius:18px;padding:20px 22px;}
.how .n{display:inline-flex;align-items:center;justify-content:center;width:30px;height:30px;border-radius:50%;
background:rgba(34,197,94,.16);color:var(--accent);font-weight:800;margin-bottom:10px;}
.how h4{margin:0 0 6px 0;font-size:1.05rem;font-weight:700;color:#fff;}
.how p{margin:0;color:var(--muted);font-size:.9rem;line-height:1.5;}
.legend{color:var(--muted);font-size:.9rem;line-height:1.55;}
.legend .key{display:inline-block;width:14px;height:14px;border-radius:3px;vertical-align:-2px;margin-right:8px;}
.note{border-left:3px solid var(--gold);background:rgba(245,185,66,.07);border-radius:0 12px 12px 0;padding:10px 14px;color:#e8d9b0;font-size:.88rem;margin:.4rem 0 1rem 0;}
@media (max-width:900px){.how{grid-template-columns:1fr;}.tiles{grid-template-columns:repeat(2,1fr);}.picks{grid-template-columns:1fr;}.mrow{grid-template-columns:1fr;gap:4px;}.hero h1{font-size:1.7rem;}}
"""


def inject_css():
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def render(markup):
    st.markdown(markup, unsafe_allow_html=True)


def esc(value):
    return html.escape("" if value is None else str(value))


def hero():
    return (
        '<div class="hero">'
        '<div class="badges"><span class="badge">PROTOTYPE</span>'
        '<span class="badge ghost">VIDEO AI</span><span class="badge ghost">MATCH DATA</span></div>'
        '<h1>Find the right player,<br><span class="accent">straight from match video</span></h1>'
        '<p>Upload highlights, say what kind of player you need, and get a ranked shortlist with photos, '
        'movement estimates and a replay that highlights the best fits.</p>'
        '</div>'
    )


def stepper(active, labels):
    """active is the 0-based index of the current step; earlier steps show as done."""
    parts = []
    for i, label in enumerate(labels):
        state = "done" if i < active else ("active" if i == active else "")
        mark = "✓" if i < active else str(i + 1)
        parts.append(f'<div class="step {state}"><span>{mark}</span>{esc(label)}</div>')
    return '<div class="steps">' + "".join(parts) + "</div>"


def tiles(items):
    """items: list of (value, label)."""
    return '<div class="tiles">' + "".join(
        f'<div class="tile"><b>{esc(v)}</b><small>{esc(label)}</small></div>' for v, label in items) + "</div>"


def photo(uri, fallback, big=False):
    size = " big" if big else ""
    if uri:
        return f'<img class="photo{size}" src="{uri}" alt="{esc(fallback)}"/>'
    return f'<div class="photo ph{size}">{esc(fallback)}</div>'


def initials(name):
    parts = [p for p in str(name).replace(".", " ").split() if p]
    return "".join(p[0] for p in parts[:2]).upper() or "?"


def picks(cards):
    """cards: list of dicts with title, sub, swatch, photo_uri, fallback, score, stats [(value, label)]."""
    out = []
    for i, c in enumerate(cards, start=1):
        dot = f'<span class="dot" style="background:{esc(c["swatch"])}"></span>' if c.get("swatch") else ""
        stats = "".join(f'<div><b>{esc(v)}</b><small>{esc(label)}</small></div>' for v, label in c["stats"])
        out.append(
            f'<div class="pick r{i}"><div class="rank">{i}</div>{photo(c.get("photo_uri"), c.get("fallback", "?"))}'
            f'<div class="body"><div class="name">{esc(c["title"])}</div>'
            f'<div class="sub">{dot}{esc(c["sub"])}</div>'
            f'<div class="score"><span class="num">{c["score"]:.0f}</span><span class="den">/ 100 FIT</span></div>'
            f'<div class="bar"><div style="width:{max(2, min(100, c["score"])):.0f}%"></div></div>'
            f'<div class="mini">{stats}</div></div></div>')
    return '<div class="picks">' + "".join(out) + "</div>"


def report_header(title, sub, score, swatch=None):
    dot = f'<span class="dot" style="background:{esc(swatch)}"></span>' if swatch else ""
    return (
        f'<div class="rhead"><div><div class="rname">{esc(title)}</div><div class="rsub">{dot}{esc(sub)}</div></div>'
        f'<div class="rscore"><b>{score:.0f}</b><small>Fit score</small></div></div>'
    )


def chips(items, kind):
    return '<div class="chips">' + "".join(f'<span class="chip {kind}">{esc(i)}</span>' for i in items) + "</div>"


def bars(rows):
    """rows: list of (label, percentile 0-100, value text, importance text)."""
    out = []
    for label, pct, value, weight in rows:
        colour = "#22c55e" if pct >= 70 else ("#38bdf8" if pct >= 40 else "#f59e0b")
        out.append(
            f'<div class="mrow"><div class="mlabel">{esc(label)}<small>{esc(weight)}</small></div>'
            f'<div class="mbar"><div style="width:{max(2, min(100, pct)):.0f}%;background:{colour}"></div></div>'
            f'<div class="mval"><b>{pct:.0f}</b> · {esc(value)}</div></div>')
    return "".join(out)


def swatches(items, lead):
    """items: list of (team name, hex colour)."""
    chips_ = "".join(f'<span class="sw"><i style="background:{esc(c)}"></i>{esc(t)}</span>' for t, c in items)
    return f'<div class="swatches"><span class="lead">{esc(lead)}</span>{chips_}</div>'


def how_it_works():
    items = [
        ("Detect and track", "The AI finds every player on the pitch and follows them through camera pans and cuts."),
        ("Measure movement", "Speed, distance, high-intensity running and pressing are estimated for each player."),
        ("Rank and highlight", "Players are scored against your requirement, and the best fits are boxed in gold in a replay."),
    ]
    return '<div class="how">' + "".join(
        f'<div><span class="n">{i}</span><h4>{esc(t)}</h4><p>{esc(d)}</p></div>'
        for i, (t, d) in enumerate(items, start=1)) + "</div>"


def note(text):
    return f'<div class="note">{esc(text)}</div>'

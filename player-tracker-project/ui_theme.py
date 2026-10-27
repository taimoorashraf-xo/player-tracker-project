"""Presentation layer: stylesheet plus the small HTML and SVG pieces the app draws with.

Charts are hand-built SVG rather than a plotting library, because what matters here is a
movement map and a speed trace drawn to a true metre scale, which general-purpose chart
widgets do not do well. Every builder returns a single-line string: Streamlit's markdown
treats indented lines as code blocks, so the HTML must not be indented.
"""
import html

import numpy as np
import streamlit as st

ACCENT = "#7dd87d"
SPEED_STOPS = [(0.0, "#3d5a80"), (0.45, "#6fb3c9"), (0.7, "#e8c468"), (1.0, "#f0784b")]

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');
:root{--bg:#0d1117;--panel:#141b24;--panel-2:#1a232e;--line:#273340;--line-soft:#1e2832;
--ink:#e6edf3;--ink-dim:#8b9bb0;--ink-faint:#5d6b7d;--accent:#7dd87d;--warn:#e8b339;--bad:#e8785b;}
.stApp{background:var(--bg);}
.stApp, .stApp :not([data-testid="stIconMaterial"]):not(.material-icons){
font-family:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif;}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stToolbar"], .stAppDeployButton, footer, #MainMenu{display:none !important;}
.block-container{max-width:1240px;padding-top:1.2rem;padding-bottom:5rem;}
h1,h2,h3,h4{color:var(--ink);letter-spacing:-.02em;}
p,li,label,span{color:var(--ink);}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p{color:var(--ink-faint);font-size:.82rem;}
.mono{font-family:'JetBrains Mono',ui-monospace,monospace;}

/* masthead */
.mast{display:flex;align-items:baseline;gap:14px;border-bottom:1px solid var(--line);
padding-bottom:12px;margin-bottom:18px;flex-wrap:wrap;}
.mast h1{font-size:1.35rem;font-weight:700;margin:0;}
.mast .sub{color:var(--ink-faint);font-size:.88rem;}
.mast .right{margin-left:auto;color:var(--ink-faint);font-size:.76rem;
font-family:'JetBrains Mono',monospace;text-transform:uppercase;letter-spacing:.06em;}

/* section heading */
.sec{display:flex;align-items:center;gap:12px;margin:26px 0 12px 0;}
.sec h3{font-size:.82rem;font-weight:600;margin:0;text-transform:uppercase;letter-spacing:.1em;
color:var(--ink-dim);white-space:nowrap;}
.sec:after{content:"";flex:1;height:1px;background:var(--line);}

/* tabs */
[data-testid="stTabs"] [role="tablist"]{gap:4px !important;border-bottom:1px solid var(--line) !important;
padding:0 !important;background:transparent !important;border-radius:0 !important;}
[data-testid="stTabs"] [role="tab"]{height:38px !important;padding:0 16px !important;color:var(--ink-faint) !important;
font-weight:600 !important;font-size:.9rem !important;border-radius:0 !important;border-bottom:2px solid transparent !important;}
[data-testid="stTabs"] [role="tab"][aria-selected="true"]{color:var(--ink) !important;
border-bottom-color:var(--accent) !important;background:transparent !important;}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none !important;}

/* inputs */
[data-testid="stVerticalBlockBorderWrapper"]{background:var(--panel);border:1px solid var(--line) !important;border-radius:10px;}
[data-testid="stExpander"]{border:1px solid var(--line);border-radius:10px;background:var(--panel);}
[data-baseweb="select"]>div,[data-baseweb="input"]>div,[data-baseweb="base-input"]{
background:var(--panel-2) !important;border-color:var(--line) !important;border-radius:7px !important;}
[data-testid="stFileUploaderDropzone"]{background:var(--panel-2);border:1px dashed var(--line);border-radius:10px;}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--accent);}
button[data-testid="stBaseButton-primary"]{background:var(--accent);border:0;color:#0c1a0c;font-weight:700;
border-radius:7px;padding:.5rem 1.3rem;}
button[data-testid="stBaseButton-primary"]:hover{filter:brightness(1.1);color:#0c1a0c;}
button[data-testid="stBaseButton-primary"]:disabled{background:#223040;color:var(--ink-faint);}
button[data-testid="stBaseButton-secondary"]{border-radius:7px;border:1px solid var(--line);
background:var(--panel-2);font-weight:500;}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:8px;overflow:hidden;}
[data-testid="stAlert"]{border-radius:8px;}
[data-testid="stVideo"] video{border-radius:8px;border:1px solid var(--line);}

/* ranked player rows */
.rows{border:1px solid var(--line);border-radius:10px;overflow:hidden;background:var(--panel);}
.row{display:grid;grid-template-columns:34px 46px 1fr 140px 62px;gap:12px;align-items:center;
padding:9px 14px;border-top:1px solid var(--line-soft);}
.row:first-child{border-top:0;}
.row.pick{background:rgba(125,216,125,.07);}
.row .pos{font-family:'JetBrains Mono',monospace;color:var(--ink-faint);font-size:.82rem;text-align:right;}
.row.pick .pos{color:var(--accent);}
.row img.face{width:34px;height:44px;object-fit:cover;border-radius:5px;background:var(--panel-2);border:1px solid var(--line);}
.row .noface{width:34px;height:44px;border-radius:5px;background:var(--panel-2);border:1px solid var(--line);
display:flex;align-items:center;justify-content:center;color:var(--ink-faint);font-size:.72rem;font-weight:700;}
.row .who{min-width:0;}
.row .nm{font-weight:600;font-size:.94rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.row .meta{color:var(--ink-faint);font-size:.76rem;font-family:'JetBrains Mono',monospace;margin-top:1px;}
.row .kit{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:6px;vertical-align:baseline;
border:1px solid rgba(255,255,255,.3);}
.row .track{height:5px;border-radius:3px;background:#1f2a36;overflow:hidden;}
.row .track>i{display:block;height:100%;background:var(--accent);border-radius:3px;}
.row .sc{font-family:'JetBrains Mono',monospace;font-weight:600;font-size:1.0rem;text-align:right;color:var(--ink);}
.row.pick .sc{color:var(--accent);}

/* measurement bars */
.m{display:grid;grid-template-columns:150px 1fr 104px;gap:12px;align-items:center;padding:5px 0;}
.m .lb{font-size:.86rem;font-weight:500;}
.m .lb small{display:block;color:var(--ink-faint);font-size:.7rem;font-weight:400;
font-family:'JetBrains Mono',monospace;}
.m .bar{height:7px;border-radius:4px;background:#1f2a36;overflow:hidden;}
.m .bar>i{display:block;height:100%;border-radius:4px;}
.m .vl{text-align:right;font-family:'JetBrains Mono',monospace;font-size:.82rem;color:var(--ink-faint);}
.m .vl b{color:var(--ink);font-weight:600;}

/* chips */
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0;}
.chip{font-size:.76rem;font-weight:500;padding:3px 9px;border-radius:5px;border:1px solid;
font-family:'JetBrains Mono',monospace;}
.chip.up{color:#9fe89f;background:rgba(125,216,125,.1);border-color:rgba(125,216,125,.3);}
.chip.down{color:#f0c66b;background:rgba(232,179,57,.1);border-color:rgba(232,179,57,.3);}
.chip.flat{color:var(--ink-dim);background:rgba(255,255,255,.04);border-color:var(--line);}

/* report header */
.rh{display:flex;align-items:flex-end;justify-content:space-between;gap:16px;flex-wrap:wrap;}
.rh .nm{font-size:1.3rem;font-weight:700;}
.rh .meta{color:var(--ink-faint);font-size:.82rem;font-family:'JetBrains Mono',monospace;margin-top:3px;}
.rh .sc{text-align:right;}
.rh .sc b{font-family:'JetBrains Mono',monospace;font-size:1.9rem;font-weight:600;color:var(--accent);line-height:1;}
.rh .sc small{display:block;color:var(--ink-faint);font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;margin-top:3px;}

/* key numbers */
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:1px;background:var(--line);
border:1px solid var(--line);border-radius:8px;overflow:hidden;margin:10px 0;}
.kpis>div{background:var(--panel);padding:10px 12px;}
.kpis b{display:block;font-family:'JetBrains Mono',monospace;font-size:1.25rem;font-weight:600;color:var(--ink);}
.kpis b em{font-style:normal;font-size:.72rem;color:var(--ink-faint);margin-left:2px;}
.kpis small{color:var(--ink-faint);font-size:.72rem;}

/* charts */
.chart{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px;}
.chart .cap{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px;}
.chart .cap h4{font-size:.78rem;font-weight:600;margin:0;text-transform:uppercase;letter-spacing:.08em;color:var(--ink-dim);}
.chart .cap span{color:var(--ink-faint);font-size:.72rem;font-family:'JetBrains Mono',monospace;}
.chart svg{display:block;width:100%;height:auto;}
.legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--ink-faint);font-size:.74rem;
font-family:'JetBrains Mono',monospace;margin-top:8px;}
.legend i{display:inline-block;width:18px;height:3px;border-radius:2px;vertical-align:middle;margin-right:5px;}

/* caveat */
.caveat{border-left:2px solid var(--warn);background:rgba(232,179,57,.06);border-radius:0 7px 7px 0;
padding:9px 13px;color:#e4d5ad;font-size:.84rem;margin:10px 0;}
@media (max-width:820px){.row{grid-template-columns:28px 1fr 56px;}.row img.face,.row .noface,.row .track{display:none;}
.m{grid-template-columns:1fr;gap:3px;}}
"""


def inject_css():
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def render(markup):
    st.markdown(markup, unsafe_allow_html=True)


def esc(value):
    return html.escape("" if value is None else str(value))


def masthead(title, subtitle, right=""):
    return (f'<div class="mast"><h1>{esc(title)}</h1><span class="sub">{esc(subtitle)}</span>'
            f'<span class="right">{esc(right)}</span></div>')


def section(title):
    return f'<div class="sec"><h3>{esc(title)}</h3></div>'


def caveat(text):
    return f'<div class="caveat">{esc(text)}</div>'


def initials(name):
    parts = [p for p in str(name).replace(".", " ").split() if p]
    return "".join(p[0] for p in parts[:2]).upper() or "?"


def kpis(items):
    """items: list of (value, unit, label)."""
    cells = "".join(f'<div><b>{esc(v)}<em>{esc(u)}</em></b><small>{esc(label)}</small></div>'
                    for v, u, label in items)
    return f'<div class="kpis">{cells}</div>'


def chips(items, kind):
    return '<div class="chips">' + "".join(
        f'<span class="chip {kind}">{esc(i)}</span>' for i in items) + "</div>"


def _speed_colour(fraction):
    """Blend the speed palette at a position from 0 to 1."""
    fraction = max(0.0, min(1.0, float(fraction)))
    for (p0, c0), (p1, c1) in zip(SPEED_STOPS, SPEED_STOPS[1:]):
        if fraction <= p1:
            span = (fraction - p0) / (p1 - p0) if p1 > p0 else 0
            a = [int(c0[i:i + 2], 16) for i in (1, 3, 5)]
            b = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
            return "#" + "".join(f"{round(x + (y - x) * span):02x}" for x, y in zip(a, b))
    return SPEED_STOPS[-1][1]


def ranked_rows(entries):
    """entries: list of dicts with rank, name, meta, kit, photo, score, share, picked."""
    out = []
    for e in entries:
        face = (f'<img class="face" src="{e["photo"]}" alt=""/>' if e.get("photo")
                else f'<div class="noface">{esc(initials(e["name"]))}</div>')
        kit = (f'<span class="kit" style="background:{esc(e["kit"])}"></span>' if e.get("kit") else "")
        out.append(
            f'<div class="row{" pick" if e.get("picked") else ""}">'
            f'<div class="pos">{e["rank"]}</div>{face}'
            f'<div class="who"><div class="nm">{esc(e["name"])}</div>'
            f'<div class="meta">{kit}{esc(e["meta"])}</div></div>'
            f'<div class="track"><i style="width:{max(2, min(100, e["score"])):.0f}%"></i></div>'
            f'<div class="sc">{e["score"]:.0f}</div></div>')
    return f'<div class="rows">{"".join(out)}</div>'


def measure_bars(rows):
    """rows: list of (label, note, percentile 0-100, value text)."""
    out = []
    for label, note, pct, value in rows:
        colour = ACCENT if pct >= 70 else ("#6fb3c9" if pct >= 40 else "#e8b339")
        out.append(
            f'<div class="m"><div class="lb">{esc(label)}<small>{esc(note)}</small></div>'
            f'<div class="bar"><i style="width:{max(2, min(100, pct)):.0f}%;background:{colour}"></i></div>'
            f'<div class="vl"><b>{esc(value)}</b> · p{pct:.0f}</div></div>')
    return "".join(out)


def report_header(name, meta, score):
    return (f'<div class="rh"><div><div class="nm">{esc(name)}</div>'
            f'<div class="meta">{esc(meta)}</div></div>'
            f'<div class="sc"><b>{score:.0f}</b><small>fit score</small></div></div>')


# =====================================================
# MOVEMENT MAP
# =====================================================
def movement_map(paths, focus=None, width=620, grid_m=10):
    """Every tracked path drawn to a true metre scale, the chosen player picked out.

    `paths` maps a player name to a frame with x and y in metres and a speed column. The
    vertical axis is distance from the camera, drawn with further away at the top so the
    picture matches what the viewer saw.
    """
    usable = {k: p for k, p in paths.items() if p is not None and len(p) >= 2}
    if not usable:
        return '<div class="chart"><div class="cap"><h4>Movement map</h4></div>' \
               '<p style="color:#5d6b7d;font-size:.84rem">No movement long enough to draw.</p></div>'

    xs = [v for p in usable.values() for v in p["x"]]
    ys = [v for p in usable.values() for v in p["y"]]
    pad = 4.0
    x0, x1 = min(xs) - pad, max(xs) + pad
    y0, y1 = min(ys) - pad, max(ys) + pad
    span_x, span_y = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    # The picture takes its shape from the ground it covers, so a wide passage of play does
    # not leave half the panel empty, and metres stay square in both directions
    height = float(np.clip(width * span_y / span_x, 230, 430))
    scale = min((width - 16) / span_x, (height - 16) / span_y)
    off_x = (width - span_x * scale) / 2
    off_y = (height - span_y * scale) / 2

    def place(x, y):
        # y is distance from the camera: flip it so the far side is at the top
        return off_x + (x - x0) * scale, off_y + (y1 - y) * scale

    parts = [f'<rect width="{width}" height="{height:.0f}" fill="#101821"/>']
    start = int(x0 // grid_m) * grid_m
    for gx in range(start, int(x1) + grid_m, grid_m):
        px, _ = place(gx, y0)
        parts.append(f'<line x1="{px:.1f}" y1="0" x2="{px:.1f}" y2="{height:.0f}" stroke="#1b2531"/>')
    start = int(y0 // grid_m) * grid_m
    for gy in range(start, int(y1) + grid_m, grid_m):
        _, py = place(x0, gy)
        parts.append(f'<line x1="0" y1="{py:.1f}" x2="{width}" y2="{py:.1f}" stroke="#1b2531"/>')

    fastest = max((p["speed"].max() for p in usable.values()), default=1.0) or 1.0
    for name, path in usable.items():
        picked = (focus is None) or (name == focus)
        points = [place(x, y) for x, y in zip(path["x"], path["y"])]
        if not picked:
            line = " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
            parts.append(f'<polyline points="{line}" fill="none" stroke="#263241" stroke-width="1.5"/>')
            continue
        # The chosen path is drawn segment by segment so speed can colour it
        for (ax, ay), (bx, by), speed in zip(points, points[1:], path["speed"][1:]):
            colour = _speed_colour(float(speed) / fastest)
            parts.append(f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}" '
                         f'stroke="{colour}" stroke-width="3" stroke-linecap="round"/>')
        sx, sy = points[0]
        ex, ey = points[-1]
        parts.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="4" fill="#0d1117" stroke="#8b9bb0" stroke-width="2"/>')
        parts.append(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="5" fill="{ACCENT}"/>')

    bar = 10 * scale
    parts.append(f'<line x1="14" y1="{height-14:.0f}" x2="{14+bar:.1f}" y2="{height-14:.0f}" stroke="#8b9bb0" stroke-width="2"/>')
    parts.append(f'<text x="{14+bar+7:.1f}" y="{height-10:.0f}" fill="#5d6b7d" font-size="11" '
                 f'font-family="JetBrains Mono, monospace">10 m</text>')

    gradient = " ".join(f'<stop offset="{p*100:.0f}%" stop-color="{c}"/>' for p, c in SPEED_STOPS)
    legend = (f'<div class="legend"><span><i style="background:linear-gradient(90deg,'
              f'{",".join(c for _, c in SPEED_STOPS)})"></i>slow to fast</span>'
              f'<span><i style="background:#8b9bb0"></i>start</span>'
              f'<span><i style="background:{ACCENT}"></i>end</span>'
              f'<span><i style="background:#263241"></i>other players</span></div>')
    return (f'<div class="chart"><div class="cap"><h4>Movement map</h4>'
            f'<span>metres · further from camera is higher</span></div>'
            f'<svg viewBox="0 0 {width} {height:.0f}" preserveAspectRatio="xMidYMid meet">'
            f'<defs><linearGradient id="sp">{gradient}</linearGradient></defs>'
            f'{"".join(parts)}</svg>{legend}</div>')


def speed_profile(path, sprint_kmh, high_kmh, width=620, height=170):
    """Speed against time for one player, with the sprint threshold marked."""
    if path is None or len(path) < 2:
        return ""
    times = list(path["t"])
    speeds = [float(s) * 3.6 for s in path["speed"]]
    t0, t1 = min(times), max(times)
    top = max(max(speeds), sprint_kmh) * 1.15 or 1.0
    span_t = max(t1 - t0, 1e-6)
    pad_l, pad_b = 34, 18

    def place(t, v):
        return (pad_l + (t - t0) / span_t * (width - pad_l - 8),
                (height - pad_b) - v / top * (height - pad_b - 10))

    parts = [f'<rect width="{width}" height="{height:.0f}" fill="#101821"/>']
    for value in range(0, int(top) + 1, 10):
        _, py = place(t0, value)
        parts.append(f'<line x1="{pad_l}" y1="{py:.1f}" x2="{width-8}" y2="{py:.1f}" stroke="#1b2531"/>')
        parts.append(f'<text x="{pad_l-6}" y="{py+3:.1f}" fill="#5d6b7d" font-size="10" '
                     f'text-anchor="end" font-family="JetBrains Mono, monospace">{value}</text>')
    for threshold, colour in ((high_kmh, "#e8b339"), (sprint_kmh, "#f0784b")):
        _, py = place(t0, threshold)
        parts.append(f'<line x1="{pad_l}" y1="{py:.1f}" x2="{width-8}" y2="{py:.1f}" '
                     f'stroke="{colour}" stroke-dasharray="3 3" opacity=".65"/>')

    points = [place(t, v) for t, v in zip(times, speeds)]
    area = " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
    base_y = height - pad_b
    parts.append(f'<polygon points="{points[0][0]:.1f},{base_y} {area} {points[-1][0]:.1f},{base_y}" '
                 f'fill="{ACCENT}" opacity=".10"/>')
    parts.append(f'<polyline points="{area}" fill="none" stroke="{ACCENT}" stroke-width="2" '
                 f'stroke-linejoin="round"/>')
    for (px, py), value in zip(points, speeds):
        if value >= sprint_kmh:
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.6" fill="#f0784b"/>')
    parts.append(f'<text x="{pad_l}" y="{height-4}" fill="#5d6b7d" font-size="10" '
                 f'font-family="JetBrains Mono, monospace">{t0:.1f}s</text>')
    parts.append(f'<text x="{width-8}" y="{height-4}" fill="#5d6b7d" font-size="10" text-anchor="end" '
                 f'font-family="JetBrains Mono, monospace">{t1:.1f}s</text>')

    legend = (f'<div class="legend"><span><i style="background:#e8b339"></i>high intensity '
              f'{high_kmh:.0f} km/h</span><span><i style="background:#f0784b"></i>sprint '
              f'{sprint_kmh:.0f} km/h</span></div>')
    return (f'<div class="chart"><div class="cap"><h4>Speed trace</h4><span>km/h</span></div>'
            f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMid meet">'
            f'{"".join(parts)}</svg>{legend}</div>')

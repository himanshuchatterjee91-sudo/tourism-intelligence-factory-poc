import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Tourism Intelligence Factory (Interactive POC)", page_icon="🏭", layout="wide")

# -----------------------------
# UI polish (CSS)
# -----------------------------
CUSTOM_CSS = """
<style>
/* Layout */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px; }

/* Typography */
h1, h2, h3 { letter-spacing: -0.02em; }
.smallcaps { font-variant: all-small-caps; letter-spacing: 0.08em; color: #6b7280; }

/* Cards */
.card {
  background: rgba(255,255,255,0.65);
  border: 1px solid rgba(15,23,42,0.08);
  border-radius: 18px;
  padding: 16px 16px;
  box-shadow: 0 8px 24px rgba(15,23,42,0.06);
}
.card h4 { margin: 0 0 6px 0; }
.kpi {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 10px;
}
.kpi .k {
  background: rgba(255,255,255,0.75);
  border: 1px solid rgba(15,23,42,0.08);
  border-radius: 16px;
  padding: 12px 14px;
}
.kpi .k .label { color: #6b7280; font-size: 0.85rem; }
.kpi .k .value { font-size: 1.25rem; font-weight: 650; margin-top: 2px; }

/* Pills */
.pill { display:inline-block; padding: 4px 10px; border-radius: 999px; font-size: 0.85rem;
  border: 1px solid rgba(15,23,42,0.12); background: rgba(255,255,255,0.7); margin-right: 6px;
}
.pill.good { border-color: rgba(16,185,129,0.35); }
.pill.warn { border-color: rgba(245,158,11,0.35); }
.pill.bad  { border-color: rgba(239,68,68,0.35); }

/* Sidebar */
section[data-testid="stSidebar"] { background: linear-gradient(180deg, rgba(248,250,252,1), rgba(255,255,255,1)); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------
# Sample data (POC realistic behavior)
# -----------------------------
@st.cache_data
def make_data(seed: int = 9):
    rng = np.random.default_rng(seed)
    markets = ["UK","Germany","France","Italy","Spain","USA","India","China","UAE","Kuwait","Qatar","Egypt","Turkey","Indonesia","Malaysia"]

    df_m = pd.DataFrame({
        "Market": markets,
        "Search_Growth_%": np.round(rng.normal(18, 10, len(markets)).clip(-5, 60), 1),
        "Seat_Growth_%": np.round(rng.normal(10, 8, len(markets)).clip(-10, 45), 1),
        "Visa_Friction_1to5": rng.integers(1, 6, len(markets)),
        "Arrivals_k": np.round(rng.normal(320, 160, len(markets)).clip(30, 900), 0).astype(int),
        "Avg_Outbound_Spend_$": np.round(rng.normal(2100, 900, len(markets)).clip(600, 5500), 0).astype(int),
        "Sentiment_Positive_Ratio": np.round(rng.normal(0.58, 0.12, len(markets)).clip(0.2, 0.9), 2),
    })

    campaigns = ["Visit Saudi - Summer", "Cultural Seasons", "Luxury Escapes", "Sports & Events", "City Breaks"]
    channels = ["Search", "Paid Social", "Video", "Display", "Influencers"]

    rows = []
    for c in campaigns:
        for m in markets[:10]:
            spend = int(rng.integers(350_000, 2_500_000))
            baseline = int(rng.integers(3_000, 35_000))
            mg = df_m.loc[df_m.Market == m, "Search_Growth_%"].iloc[0]
            sent = df_m.loc[df_m.Market == m, "Sentiment_Positive_Ratio"].iloc[0]
            seats = df_m.loc[df_m.Market == m, "Seat_Growth_%"].iloc[0]
            visa = df_m.loc[df_m.Market == m, "Visa_Friction_1to5"].iloc[0]

            # incremental arrivals depends on momentum + friction
            raw = baseline * (0.02 + (mg/220) + (seats/260) + (sent-0.5)/4.5) * rng.uniform(0.65, 1.25)
            raw *= (1.0 - (visa-1)*0.06)
            inc = int(max(0, raw))
            ch = rng.choice(channels, p=[0.28,0.28,0.18,0.16,0.10])
            rows.append([c,m,ch,spend,baseline,inc])

    df_c = pd.DataFrame(rows, columns=["Campaign","Market","Channel","Spend_$","Baseline_Arrivals","Incremental_Arrivals"])
    df_c["CPIT_$"] = np.where(df_c["Incremental_Arrivals"] > 0, df_c["Spend_$"] / df_c["Incremental_Arrivals"], np.nan)

    events = ["Riyadh Season", "F1 Jeddah", "AlUla Festival", "Esports World Cup"]
    erows = []
    for e in events:
        for m in markets[:10]:
            pre = int(rng.integers(45, 120))
            peak = int(pre * rng.uniform(1.15, 2.4))
            post = int(pre * rng.uniform(0.9, 1.65))
            inc_a = int(rng.integers(500, 9000))
            inc_r = int(inc_a * rng.integers(900, 2800))
            retention = float(np.round(rng.uniform(0.06, 0.38), 2))
            erows.append([e,m,pre,peak,post,inc_a,inc_r,retention])
    df_e = pd.DataFrame(erows, columns=["Event","Market","Search_Pre","Search_Event","Search_Post","Inc_Arrivals","Inc_Revenue_$","Post_Retention"])

    return df_m, df_c, df_e


df_markets, df_campaign, df_event = make_data()

# -----------------------------
# Scoring + tiering
# -----------------------------
def norm(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    if s.max() == s.min():
        return pd.Series([50]*len(s), index=s.index)
    return (s - s.min()) / (s.max() - s.min()) * 100

def tier(score: float) -> str:
    if score >= 70: return "Tier 1"
    if score >= 50: return "Tier 2"
    return "Tier 3"

def score_markets(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    s_search = norm(df["Search_Growth_%"])
    s_seats  = norm(df["Seat_Growth_%"])
    s_arr    = norm(df["Arrivals_k"])
    s_spend  = norm(df["Avg_Outbound_Spend_$"])
    s_sent   = norm(df["Sentiment_Positive_Ratio"])
    s_visa   = 100 - norm(df["Visa_Friction_1to5"])  # lower friction better

    score = (
        weights["search"]*s_search +
        weights["seats"]*s_seats +
        weights["visa"]*s_visa +
        weights["arrivals"]*s_arr +
        weights["spend"]*s_spend +
        weights["sentiment"]*s_sent
    )

    out = df.copy()
    out["Opportunity_Score"] = np.round(score, 1)
    out["Tier"] = out["Opportunity_Score"].apply(tier)

    # For explainability: show top drivers per market
    driver_df = pd.DataFrame({
        "Search": weights["search"]*s_search,
        "Seats": weights["seats"]*s_seats,
        "Visa Ease": weights["visa"]*s_visa,
        "Arrivals": weights["arrivals"]*s_arr,
        "Spend": weights["spend"]*s_spend,
        "Sentiment": weights["sentiment"]*s_sent
    })
    out["_drivers"] = driver_df.apply(lambda r: list(r.sort_values(ascending=False).head(3).index), axis=1)

    return out.sort_values("Opportunity_Score", ascending=False).reset_index(drop=True)

# -----------------------------
# Forecast + simulation (explainable)
# -----------------------------
def forecast_arrivals(df_m: pd.DataFrame, markets: list, months: int = 12, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = dt.date.today().replace(day=1)
    months_list = [base + dt.timedelta(days=31*i) for i in range(months)]
    months_list = [d.replace(day=1) for d in months_list]

    season = np.array([0.88,0.90,0.95,1.00,1.05,1.10,1.08,1.03,1.00,0.97,0.92,0.90])[:months]

    rows = []
    for m in markets:
        r = df_m[df_m.Market == m].iloc[0]
        annual = r["Arrivals_k"]*1000
        monthly_base = annual/12

        momentum = (
            (r["Search_Growth_%"]/100)*0.55 +
            (r["Seat_Growth_%"]/100)*0.45 -
            ((r["Visa_Friction_1to5"]-1)/4)*0.22 +
            (r["Sentiment_Positive_Ratio"]-0.5)*0.75
        )
        momentum = float(np.clip(momentum, -0.2, 0.55))

        for i, d in enumerate(months_list):
            noise = rng.normal(0, 0.035)
            val = monthly_base * season[i] * (1 + momentum + noise)
            rows.append([d, m, max(0, int(val))])

    return pd.DataFrame(rows, columns=["Month","Market","Arrivals_Forecast"])

# -----------------------------
# Sidebar: experience mode
# -----------------------------
st.sidebar.markdown("### 🏭 Tourism Intelligence Factory")
mode = st.sidebar.radio("Experience", ["Guided Demo", "Explore Freely"])
st.sidebar.markdown("---")

PAGES = [
    ("Factory Floor", "🏭"),
    ("Market Tiering", "🗺️"),
    ("Perception → Narrative", "🧠"),
    ("CPIT (Campaign Efficiency)", "🎯"),
    ("Event Impact", "🎟️"),
    ("Forecast + Simulator", "🔮"),
]
page_names = [p[0] for p in PAGES]
default_index = 0
page = st.sidebar.radio("Stations", page_names, index=default_index)

# Guided demo script (for presenting)
DEMO_STEPS = {
    "Factory Floor": "Set the mandate → show the factory line → pick a station.",
    "Market Tiering": "Change weights → watch tiers change → open ‘Why this market?’",
    "Perception → Narrative": "Select a market → see perception snapshot → get a narrative blueprint.",
    "CPIT (Campaign Efficiency)": "Pick a campaign → see CPIT distribution → auto ‘Scale/Optimize/Fix’ call.",
    "Event Impact": "Pick an event → see lift + retention → get an impact verdict.",
    "Forecast + Simulator": "Toggle scenarios → show forecast shifts → highlight risk flags."
}

# -----------------------------
# Header ribbon
# -----------------------------
st.markdown('<div class="smallcaps">interactive proof-of-capability • executive storytelling prototype</div>', unsafe_allow_html=True)

# -----------------------------
# Page: Factory Floor
# -----------------------------
if page == "Factory Floor":
    st.title("The Tourism Intelligence Factory")
    left, right = st.columns([1.2, 1])

    with left:
        st.markdown(
            """
<div class="card">
<h3>STA’s mandate is bigger than marketing.</h3>
<p><b>STA is the Chief Marketing & Demand Engine</b> — responsible for driving visitation and shaping Saudi’s global tourism perception.</p>
<span class="pill good">Nation brand repositioning</span>
<span class="pill warn">Multi-market complexity</span>
<span class="pill">Mega-project demand creation</span>
<hr/>
<p><b>The problem:</b> signals are fragmented (search, seats, sentiment, arrivals, events), so decisions become reactive.</p>
<p><b>The solution:</b> a continuous intelligence engine that turns signals into <b>decision outputs</b>.</p>
</div>
""",
            unsafe_allow_html=True,
        )

        if mode == "Guided Demo":
            st.info(f"Presenter cue: {DEMO_STEPS[page]}")

    with right:
        st.subheader("Factory line (visual)")
        st.graphviz_chart(
            """
digraph {
  rankdir=LR;
  node [shape=box, style="rounded,filled", color="#CBD5E1", fillcolor="#F8FAFC", fontname="Helvetica"];
  edge [color="#94A3B8"];

  inputs [label="Inputs\\nSearch • Seats • Visa\\nArrivals • Spend • Sentiment\\nEvents", fillcolor="#EEF2FF", color="#A5B4FC"];
  mkt [label="Market Tiering\\n(Tier 1/2/3)"];
  per [label="Perception Tracking\\n(Index + Heatmap)"];
  nar [label="Narrative Blueprint\\n(what to say)"];
  cpit [label="CPIT\\n(campaign efficiency)"];
  evt [label="Event Impact\\n(volume + revenue)"];
  frc [label="Forecast + Simulator\\n(12-mo outlook)"];
  out [label="Decision Outputs\\nExec briefs • scorecards\\npriorities • scenarios", fillcolor="#ECFDF5", color="#6EE7B7"];

  inputs -> mkt -> out;
  inputs -> per -> nar -> out;
  inputs -> cpit -> out;
  inputs -> evt -> out;
  inputs -> frc -> out;
}
"""
        )

        st.markdown(
            """
<div class="card">
<h4>What makes this feel “real”</h4>
<ul>
  <li>Every station: <b>input → output</b> (interactive)</li>
  <li>Explainable logic (not black-box)</li>
  <li>Executive takeaways ready to paste into briefs</li>
</ul>
</div>
""",
            unsafe_allow_html=True
        )

# -----------------------------
# Page: Market Tiering
# -----------------------------
elif page == "Market Tiering":
    st.title("Market Tiering (Cross-market intelligence synthesis)")

    top = st.container()
    with top:
        st.markdown(
            """
<div class="card">
<h3>What we do</h3>
<p>Score 10–15 source markets using a transparent model (search, seats, visa ease, arrivals, spend, sentiment) and produce Tier 1/2/3 priorities.</p>
<h3>What we solve</h3>
<p>Market decisions driven by legacy allocation or fragmented signals.</p>
<h3>Key takeaway</h3>
<p><b>“We know which markets deserve focus — and why.”</b></p>
</div>
""", unsafe_allow_html=True)

    st.markdown("### Tune weights (live)")
    c1, c2, c3 = st.columns(3)
    with c1:
        w_search = st.slider("Search growth weight", 0, 40, 25)
        w_seats = st.slider("Seat growth weight", 0, 40, 15)
    with c2:
        w_visa = st.slider("Visa ease weight", 0, 40, 15)
        w_arr = st.slider("Arrivals weight", 0, 40, 10)
    with c3:
        w_spend = st.slider("Outbound spend weight", 0, 40, 20)
        w_sent = st.slider("Sentiment weight", 0, 40, 15)

    total = w_search + w_seats + w_visa + w_arr + w_spend + w_sent
    if total == 0:
        st.error("Please set at least one weight > 0.")
        st.stop()

    weights = {
        "search": w_search/total,
        "seats": w_seats/total,
        "visa": w_visa/total,
        "arrivals": w_arr/total,
        "spend": w_spend/total,
        "sentiment": w_sent/total
    }
    scored = score_markets(df_markets, weights)

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Market Opportunity Index")
        show = scored[["Market","Opportunity_Score","Tier","Search_Growth_%","Seat_Growth_%","Visa_Friction_1to5","Arrivals_k","Avg_Outbound_Spend_$","Sentiment_Positive_Ratio"]]
        st.dataframe(show, use_container_width=True, height=420)

    with right:
        st.subheader("Tier distribution")
        tier_counts = scored["Tier"].value_counts().reset_index()
        tier_counts.columns = ["Tier","Count"]
        st.plotly_chart(px.bar(tier_counts, x="Tier", y="Count"), use_container_width=True)

        st.subheader("Why this market ranked")
        pick = st.selectbox("Pick a market", scored["Market"].tolist(), index=0)
        row = scored[scored.Market == pick].iloc[0]
        drivers = row["_drivers"]
        st.markdown(
            f"""
<div class="card">
<h4>{pick}: {row['Tier']} • Score {row['Opportunity_Score']}</h4>
<p><b>Top drivers:</b> {drivers[0]}, {drivers[1]}, {drivers[2]}</p>
<p class="smallcaps">Explainable • defensible • adjustable</p>
</div>
""", unsafe_allow_html=True
        )

    if mode == "Guided Demo":
        st.info(f"Presenter cue: {DEMO_STEPS[page]}")

# -----------------------------
# Page: Perception -> Narrative
# -----------------------------
elif page == "Perception → Narrative":
    st.title("Perception → Narrative (AI-driven perception tracking + narrative blueprint)")

    st.markdown(
        """
<div class="card">
<h3>What we do</h3>
<p>Track perception signals by market and convert them into campaign-ready narrative blueprints (positioning, themes, objections, proof points).</p>
<h3>What we solve</h3>
<p>Generic messaging and slow reaction to perception shifts.</p>
<h3>Key takeaway</h3>
<p><b>“We know what to say — backed by evidence.”</b></p>
</div>
""",
        unsafe_allow_html=True
    )

    market = st.selectbox("Select a market", df_markets["Market"].tolist(), index=0)
    r = df_markets[df_markets.Market == market].iloc[0]
    sentiment = float(r["Sentiment_Positive_Ratio"])
    spend = int(r["Avg_Outbound_Spend_$"])
    search = float(r["Search_Growth_%"])
    visa = int(r["Visa_Friction_1to5"])

    st.markdown("### Perception snapshot")
    st.markdown(
        f"""
<div class="kpi">
  <div class="k"><div class="label">Sentiment (positive ratio)</div><div class="value">{sentiment:.2f}</div></div>
  <div class="k"><div class="label">Search growth (12m)</div><div class="value">{search:.1f}%</div></div>
  <div class="k"><div class="label">Visa friction</div><div class="value">{visa}/5</div></div>
</div>
""",
        unsafe_allow_html=True
    )

    # Narrative blueprint logic (still POC, but structured and not “random”)
    if sentiment < 0.50:
        positioning = "Reassurance + openness (reduce uncertainty)"
        themes = ["Ease & confidence", "Welcoming experiences", "Clear planning support"]
        objections = ["Safety concerns", "Uncertainty about culture", "Trip planning friction"]
        proof = ["Clear travel planning paths", "Trusted partner endorsements", "Visitor stories"]
    elif spend >= 2800:
        positioning = "Premium + exclusivity (high-yield focus)"
        themes = ["Luxury escapes", "Signature experiences", "Limited-time events"]
        objections = ["Is it truly premium?", "What’s uniquely Saudi vs alternatives?"]
        proof = ["Premium stays", "Curated itineraries", "High-end experiences"]
    elif search >= 25:
        positioning = "Intent conversion (make planning irresistible)"
        themes = ["Itineraries", "Seasonal highlights", "Direct access + routes"]
        objections = ["When to go?", "How to plan?", "What to do first?"]
        proof = ["Route maps", "Event calendar", "Top 3 itineraries"]
    else:
        positioning = "Culture + discovery (differentiated identity)"
        themes = ["Heritage", "Food & culture", "Hidden gems"]
        objections = ["Is there enough to do?", "Is it for me?"]
        proof = ["Cultural landmarks", "Local experiences", "Community-led stories"]

    left, right = st.columns([1.05, 0.95])
    with left:
        st.subheader("Narrative blueprint (campaign-ready)")
        st.markdown(
            f"""
<div class="card">
<h4>Positioning emphasis</h4>
<p><b>{positioning}</b></p>
<h4>Key themes</h4>
<ul>{''.join([f"<li>{t}</li>" for t in themes])}</ul>
</div>
""", unsafe_allow_html=True
        )

    with right:
        st.subheader("Objections & proof points")
        st.markdown(
            f"""
<div class="card">
<h4>Likely objections</h4>
<ul>{''.join([f"<li>{o}</li>" for o in objections])}</ul>
<h4>Reasons to believe</h4>
<ul>{''.join([f"<li>{p}</li>" for p in proof])}</ul>
</div>
""", unsafe_allow_html=True
        )

    st.caption("POC note: In the full program, these are generated from real perception clusters and content performance, not heuristics.")
    if mode == "Guided Demo":
        st.info(f"Presenter cue: {DEMO_STEPS[page]}")

# -----------------------------
# Page: CPIT
# -----------------------------
elif page == "CPIT (Campaign Efficiency)":
    st.title("CPIT (Cost per incremental tourist)")

    st.markdown(
        """
<div class="card">
<h3>What we do</h3>
<p>Compute CPIT across major campaigns and markets using incremental arrivals vs baseline.</p>
<h3>What we solve</h3>
<p>Performance conversations dominated by clicks, impressions, or channel bias rather than efficiency.</p>
<h3>Key takeaway</h3>
<p><b>“We can compare campaigns on one executive metric.”</b></p>
</div>
""", unsafe_allow_html=True
    )

    campaign = st.selectbox("Select campaign", sorted(df_campaign["Campaign"].unique()))
    view = df_campaign[df_campaign.Campaign == campaign].copy()
    view["CPIT_$"] = view["CPIT_$"].round(2)

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("CPIT scorecard")
        st.dataframe(
            view.sort_values("CPIT_$")[["Market","Channel","Spend_$","Baseline_Arrivals","Incremental_Arrivals","CPIT_$"]],
            use_container_width=True, height=420
        )

    with right:
        st.subheader("CPIT by market (lower is better)")
        fig = px.bar(
            view.sort_values("CPIT_$"),
            x="Market", y="CPIT_$",
            hover_data=["Spend_$","Incremental_Arrivals","Channel"]
        )
        st.plotly_chart(fig, use_container_width=True)

        q = view["CPIT_$"].dropna()
        if len(q) > 0:
            p33, p66 = np.nanpercentile(q, [33, 66])
            scale = view[view["CPIT_$"] <= p33].sort_values("CPIT_$")["Market"].head(3).tolist()
            fix = view[view["CPIT_$"] > p66].sort_values("CPIT_$", ascending=False)["Market"].head(3).tolist()
            st.markdown(
                f"""
<div class="card">
<h4>Action call (auto)</h4>
<span class="pill good"><b>Scale:</b> {', '.join(scale) if scale else '—'}</span><br/>
<span class="pill warn"><b>Optimize:</b> markets in the middle band</span><br/>
<span class="pill bad"><b>Investigate/Fix:</b> {', '.join(fix) if fix else '—'}</span>
</div>
""", unsafe_allow_html=True
            )

    if mode == "Guided Demo":
        st.info(f"Presenter cue: {DEMO_STEPS[page]}")

# -----------------------------
# Page: Event Impact
# -----------------------------
elif page == "Event Impact":
    st.title("Event Impact Measurement (volume + revenue + retention)")

    st.markdown(
        """
<div class="card">
<h3>What we do</h3>
<p>Quantify event impact using incremental arrivals and revenue proxies, plus search lift and post-event retention.</p>
<h3>What we solve</h3>
<p>Events evaluated on visibility rather than measurable tourism impact.</p>
<h3>Key takeaway</h3>
<p><b>“We can prove whether events drive real tourism growth.”</b></p>
</div>
""", unsafe_allow_html=True
    )

    event = st.selectbox("Select event", sorted(df_event["Event"].unique()))
    ev = df_event[df_event.Event == event].copy()

    # Verdict logic for storytelling
    total_arr = int(ev["Inc_Arrivals"].sum())
    total_rev = int(ev["Inc_Revenue_$"].sum())
    avg_ret = float(ev["Post_Retention"].mean())
    # Simple verdict
    if total_arr > 45_000 and avg_ret > 0.20:
        verdict = ("High impact", "good")
    elif total_arr > 25_000:
        verdict = ("Medium impact", "warn")
    else:
        verdict = ("Needs redesign", "bad")

    st.markdown(
        f"""
<div class="kpi">
  <div class="k"><div class="label">Total incremental arrivals (proxy)</div><div class="value">{total_arr:,}</div></div>
  <div class="k"><div class="label">Total incremental revenue (proxy)</div><div class="value">${total_rev:,.0f}</div></div>
  <div class="k"><div class="label">Avg post-event retention</div><div class="value">{avg_ret:.2f}</div></div>
</div>
""", unsafe_allow_html=True
    )

    st.markdown(
        f"""
<div class="card">
<h4>Impact verdict</h4>
<span class="pill {verdict[1]}"><b>{verdict[0]}</b></span>
<p class="smallcaps">Use verdict to decide: scale • optimize • rethink</p>
</div>
""", unsafe_allow_html=True
    )

    left, right = st.columns([1.1, 0.9])
    with left:
        st.subheader("Market scorecard")
        st.dataframe(ev.sort_values("Inc_Arrivals", ascending=False), use_container_width=True, height=420)

    with right:
        st.subheader("Search lift (pre → event → post)")
        melt = ev.melt(
            id_vars=["Market"],
            value_vars=["Search_Pre","Search_Event","Search_Post"],
            var_name="Window", value_name="Search_Index"
        )
        st.plotly_chart(px.line(melt, x="Window", y="Search_Index", color="Market", markers=True), use_container_width=True)

    if mode == "Guided Demo":
        st.info(f"Presenter cue: {DEMO_STEPS[page]}")

# -----------------------------
# Page: Forecast + Simulator
# -----------------------------
elif page == "Forecast + Simulator":
    st.title("12-month rolling forecast + scenario simulator")

    st.markdown(
        """
<div class="card">
<h3>What we do</h3>
<p>Create a rolling 12-month arrivals forecast and simulate how key levers change the outlook (seats, visa ease, sentiment, event boost).</p>
<h3>What we solve</h3>
<p>Reactive planning and late detection of market shifts.</p>
<h3>Key takeaway</h3>
<p><b>“We reduce uncertainty before major decisions are made.”</b></p>
</div>
""", unsafe_allow_html=True
    )

    base_markets = df_markets.sort_values("Arrivals_k", ascending=False)["Market"].head(4).tolist()
    selected = st.multiselect("Select markets", df_markets["Market"].tolist(), default=base_markets)
    if not selected:
        st.warning("Select at least one market.")
        st.stop()

    st.markdown("### Scenario levers")
    a, b, c, d = st.columns(4)
    with a:
        delta_seats = st.slider("Seat growth change (%)", -20, 30, 10)
    with b:
        delta_sent = st.slider("Sentiment shift (points)", -10, 10, 2)
    with c:
        visa_improve = st.slider("Visa friction improvement (steps)", 0, 2, 1)
    with d:
        event_boost = st.slider("Event boost (uplift %)", 0, 20, 5)

    df_s = df_markets.copy()
    df_s.loc[df_s.Market.isin(selected), "Seat_Growth_%"] += delta_seats
    df_s.loc[df_s.Market.isin(selected), "Sentiment_Positive_Ratio"] = (df_s.loc[df_s.Market.isin(selected), "Sentiment_Positive_Ratio"] + delta_sent/100).clip(0.2, 0.9)
    df_s.loc[df_s.Market.isin(selected), "Visa_Friction_1to5"] = (df_s.loc[df_s.Market.isin(selected), "Visa_Friction_1to5"] - visa_improve).clip(1,5)

    fc = forecast_arrivals(df_s, selected, months=12)
    fc["Arrivals_Forecast"] = (fc["Arrivals_Forecast"] * (1 + event_boost/100)).astype(int)

    st.subheader("Forecast (scenario-adjusted)")
    st.plotly_chart(px.line(fc, x="Month", y="Arrivals_Forecast", color="Market", markers=True), use_container_width=True)

    # Risk flags (simple)
    st.subheader("Risk flags (POC)")
    risk_rows = []
    for m in selected:
        r = df_s[df_s.Market == m].iloc[0]
        flag = "Green"
        if r["Visa_Friction_1to5"] >= 4 or r["Sentiment_Positive_Ratio"] < 0.48:
            flag = "Red"
        elif r["Seat_Growth_%"] < 0 or r["Search_Growth_%"] < 5:
            flag = "Yellow"
        risk_rows.append([m, flag, r["Search_Growth_%"], r["Seat_Growth_%"], r["Visa_Friction_1to5"], r["Sentiment_Positive_Ratio"]])

    risk = pd.DataFrame(risk_rows, columns=["Market","Flag","Search_Growth_%","Seat_Growth_%","Visa_Friction_1to5","Sentiment_Positive_Ratio"])
    st.dataframe(risk, use_container_width=True)

    if mode == "Guided Demo":
        st.info(f"Presenter cue: {DEMO_STEPS[page]}")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("POC prototype: visuals + interactivity to demonstrate the Tourism Intelligence Factory concept.")

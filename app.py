import math
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

# Optional: Plotly for quick charts (Streamlit-friendly)
import plotly.express as px

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Tourism Intelligence Factory (POC)",
    page_icon="🏭",
    layout="wide",
)

# -----------------------------
# Helper functions + sample data
# -----------------------------
@st.cache_data
def make_sample_data(seed: int = 7):
    rng = np.random.default_rng(seed)

    markets = [
        "UK", "Germany", "France", "Italy", "Spain",
        "USA", "India", "China", "UAE", "Kuwait",
        "Qatar", "Egypt", "Turkey", "Indonesia", "Malaysia"
    ]

    df = pd.DataFrame({
        "Market": markets,
        # 12-month travel search growth to Saudi (%)
        "Search_Growth_%": np.round(rng.normal(18, 10, len(markets)).clip(-5, 60), 1),
        # Airline seat capacity growth (%)
        "Seat_Growth_%": np.round(rng.normal(10, 8, len(markets)).clip(-10, 45), 1),
        # Visa friction score 1-5 (5 = high friction)
        "Visa_Friction_1to5": rng.integers(1, 6, len(markets)),
        # Current visitor volume (annual, thousands)
        "Arrivals_k": np.round(rng.normal(320, 160, len(markets)).clip(30, 900), 0).astype(int),
        # Average outbound spend ($) - proxy
        "Avg_Outbound_Spend_$": np.round(rng.normal(2100, 900, len(markets)).clip(600, 5500), 0).astype(int),
        # Sentiment score: positive ratio (0-1)
        "Sentiment_Positive_Ratio": np.round(rng.normal(0.58, 0.12, len(markets)).clip(0.2, 0.9), 2),
    })

    # Campaigns sample
    campaigns = ["Visit Saudi - Summer", "Cultural Seasons", "Luxury Escapes", "Sports & Events", "City Breaks"]
    channels = ["Search", "Paid Social", "Video", "Display", "Influencers"]

    camp_rows = []
    for c in campaigns:
        for m in markets[:10]:  # keep it lighter
            spend = rng.integers(300_000, 2_000_000)
            # baseline arrivals (during campaign period)
            baseline = rng.integers(2_500, 30_000)
            # incremental uplift influenced by search growth + sentiment
            mg = df.loc[df["Market"] == m, "Search_Growth_%"].values[0]
            sent = df.loc[df["Market"] == m, "Sentiment_Positive_Ratio"].values[0]
            uplift = int(max(0, baseline * (0.02 + (mg/200) + (sent-0.5)/5) * rng.uniform(0.6, 1.2)))
            incremental = uplift
            arrivals = baseline + incremental
            # allocate by channel (rough)
            ch = rng.choice(channels)
            camp_rows.append([c, m, ch, spend, baseline, incremental, arrivals])

    df_campaign = pd.DataFrame(
        camp_rows,
        columns=["Campaign", "Market", "Channel", "Spend_$", "Baseline_Arrivals", "Incremental_Arrivals", "Total_Arrivals"]
    )
    df_campaign["CPIT_$"] = np.round(df_campaign["Spend_$"] / df_campaign["Incremental_Arrivals"].replace(0, np.nan), 2)
    df_campaign["CPIT_$"] = df_campaign["CPIT_$"].fillna(np.inf)

    # Events sample
    events = ["Riyadh Season", "F1 Jeddah", "AlUla Festival", "Esports World Cup"]
    event_rows = []
    for e in events:
        for m in markets[:10]:
            pre_search = rng.integers(40, 120)
            event_search = int(pre_search * rng.uniform(1.1, 2.2))
            post_search = int(pre_search * rng.uniform(0.9, 1.6))
            inc_arrivals = rng.integers(400, 8000)
            inc_revenue = inc_arrivals * rng.integers(900, 2600)  # proxy
            retention = np.round(rng.uniform(0.05, 0.35), 2)  # 4-week post-event retention proxy
            event_rows.append([e, m, pre_search, event_search, post_search, inc_arrivals, inc_revenue, retention])

    df_event = pd.DataFrame(
        event_rows,
        columns=[
            "Event", "Market", "Search_Index_Pre", "Search_Index_Event", "Search_Index_Post",
            "Incremental_Arrivals", "Incremental_Revenue_$", "PostEvent_Retention_Rate"
        ]
    )

    return df, df_campaign, df_event


def tier_from_score(score: float) -> str:
    if score >= 70:
        return "Tier 1"
    elif score >= 50:
        return "Tier 2"
    return "Tier 3"


def compute_market_scores(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    # Normalize components 0-100
    def norm(series):
        s = series.astype(float)
        if s.max() == s.min():
            return pd.Series([50]*len(s), index=s.index)
        return (s - s.min()) / (s.max() - s.min()) * 100

    # Higher is better for these:
    s_search = norm(df["Search_Growth_%"])
    s_seats = norm(df["Seat_Growth_%"])
    s_arr = norm(df["Arrivals_k"])
    s_spend = norm(df["Avg_Outbound_Spend_$"])
    s_sent = norm(df["Sentiment_Positive_Ratio"])

    # Lower is better for visa friction, invert:
    s_visa = 100 - norm(df["Visa_Friction_1to5"])

    score = (
        weights["search"] * s_search +
        weights["seats"]  * s_seats  +
        weights["visa"]   * s_visa   +
        weights["arrivals"] * s_arr  +
        weights["spend"] * s_spend   +
        weights["sentiment"] * s_sent
    )

    out = df.copy()
    out["Opportunity_Score_0to100"] = np.round(score, 1)
    out["Tier"] = out["Opportunity_Score_0to100"].apply(tier_from_score)
    out = out.sort_values("Opportunity_Score_0to100", ascending=False).reset_index(drop=True)
    return out


def rolling_arrival_forecast(df_markets: pd.DataFrame, selected_markets: list, months: int = 12, seed: int = 11):
    """
    Simple, explainable forecast:
    - baseline from current arrivals
    - seasonality curve
    - momentum from search growth + seat growth - visa friction
    Returns long table: Month, Market, Forecast_Arrivals
    """
    rng = np.random.default_rng(seed)
    base_date = dt.date.today().replace(day=1)
    month_list = [base_date + dt.timedelta(days=31*i) for i in range(months)]
    month_list = [d.replace(day=1) for d in month_list]

    seasonality = np.array([0.88, 0.90, 0.95, 1.00, 1.05, 1.10, 1.08, 1.03, 1.00, 0.97, 0.92, 0.90])
    seasonality = seasonality[:months] if months <= 12 else np.pad(seasonality, (0, months-12), mode="wrap")

    rows = []
    for m in selected_markets:
        r = df_markets[df_markets["Market"] == m].iloc[0]
        annual = r["Arrivals_k"] * 1000
        monthly_base = annual / 12

        # momentum factor: explainable combination
        momentum = (
            (r["Search_Growth_%"] / 100) * 0.6 +
            (r["Seat_Growth_%"] / 100) * 0.5 -
            ((r["Visa_Friction_1to5"] - 1) / 4) * 0.25 +
            (r["Sentiment_Positive_Ratio"] - 0.5) * 0.8
        )
        momentum = max(-0.2, min(0.5, momentum))

        for i, d in enumerate(month_list):
            noise = rng.normal(0, 0.04)
            val = monthly_base * seasonality[i] * (1 + momentum + noise)
            rows.append([d, m, max(0, int(val))])

    return pd.DataFrame(rows, columns=["Month", "Market", "Forecast_Arrivals"])


# -----------------------------
# Load data
# -----------------------------
df_markets, df_campaign, df_event = make_sample_data()

# -----------------------------
# Sidebar navigation
# -----------------------------
st.sidebar.title("🏭 Tourism Intelligence Factory")
page = st.sidebar.radio(
    "Navigate",
    [
        "0) Landing (Story)",
        "1) Cross-Market Intelligence",
        "2) Perception + Narrative",
        "3) CPIT (Campaign Efficiency)",
        "4) Event Impact Measurement",
        "5) Forecast + Simulation",
        "6) Outputs (12-month deliverables)"
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("POC built for executive storytelling + interactive proof.")

# -----------------------------
# 0) Landing
# -----------------------------
if page == "0) Landing (Story)":
    st.title("🏭 The Tourism Intelligence Factory (POC)")
    st.subheader("A continuous intelligence engine powering STA’s decisions")

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown(
            """
### Setting the narrative (STA mandate)
**Think of STA as the Chief Marketing & Demand Engine of Saudi tourism.**

Key responsibilities:
- **National Destination Marketer**
- **International Market Builder**
- **Demand Creator for Mega Projects**
- **Brand Narrative Architect**

**Long term:** STA isn’t just promoting tourism — it is helping reposition Saudi Arabia’s global identity.
That’s a **nation-brand transformation mandate**.

### The tension
To deliver on this mandate, leadership needs answers that are:
- Cross-market, not siloed
- Forward-looking, not reactive
- Measurable, not anecdotal
"""
        )
    with col2:
        st.markdown("### Factory concept (simple visual)")
        st.info(
            "Inputs → Intelligence Engines → Decision Outputs\n\n"
            "**Inputs**: Search, Seats, Visa, Arrivals, Spend, Sentiment, Events\n"
            "**Engines**: Market Tiering, Perception, Narrative, CPIT, Event Impact, Forecast+Simulation\n"
            "**Outputs**: Executive briefs and decision-ready scorecards"
        )

    st.markdown("---")
    st.markdown("### What you can demo in this POC")
    st.write(
        "Use the left navigation to walk through each module. Each page is designed to feel like "
        "a real intelligence product—simple, explainable, and decision-grade."
    )

# -----------------------------
# 1) Cross-Market Intelligence
# -----------------------------
elif page == "1) Cross-Market Intelligence":
    st.title("1) Cross-Market Intelligence Synthesis")
    st.caption("Evaluate 10–15 source markets → weighted score → Tier 1/2/3 prioritization.")

    st.markdown(
        """
**What we’re doing:**  
We score each market using a structured model combining search demand growth, seat capacity growth, visa friction,
current arrivals, outbound spend, and sentiment.

**What we’re solving:**  
Market prioritization based on fragmented signals or historical bias.

**Key takeaway:**  
**“We now know, with evidence, which markets deserve focus — and why.”**
"""
    )

    st.markdown("---")
    st.subheader("Set the weights (simple + transparent)")

    w1, w2, w3 = st.columns(3)
    with w1:
        w_search = st.slider("Weight: Search growth", 0, 40, 25)
        w_seats = st.slider("Weight: Seat growth", 0, 40, 15)
    with w2:
        w_visa = st.slider("Weight: Visa ease (lower friction)", 0, 40, 15)
        w_arrivals = st.slider("Weight: Current arrivals", 0, 40, 10)
    with w3:
        w_spend = st.slider("Weight: Avg outbound spend", 0, 40, 20)
        w_sent = st.slider("Weight: Sentiment", 0, 40, 15)

    total = w_search + w_seats + w_visa + w_arrivals + w_spend + w_sent
    if total == 0:
        st.error("Set at least one weight > 0.")
        st.stop()

    weights = {
        "search": w_search / total,
        "seats": w_seats / total,
        "visa": w_visa / total,
        "arrivals": w_arrivals / total,
        "spend": w_spend / total,
        "sentiment": w_sent / total
    }

    scored = compute_market_scores(df_markets, weights)

    st.subheader("Market Opportunity Index")
    st.dataframe(scored, use_container_width=True)

    st.subheader("Tier distribution")
    tier_counts = scored["Tier"].value_counts().reset_index()
    tier_counts.columns = ["Tier", "Count"]
    fig = px.bar(tier_counts, x="Tier", y="Count")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 2) Perception + Narrative
# -----------------------------
elif page == "2) Perception + Narrative":
    st.title("2) AI-Driven Perception Tracking + Data-Backed Narrative Architecture")
    st.caption("Perception tells you what people believe; narrative tells you what to say—by market and segment.")

    st.markdown(
        """
**What we’re doing:**  
We track perception signals by market and translate them into **campaign narrative architectures**
(positioning emphasis, themes, and gaps to correct).

**What we’re solving:**  
Generic messaging and slow reaction to perception shifts.

**Key takeaway:**  
**“Our brand story is informed by data — not assumption.”**
"""
    )

    st.markdown("---")

    market = st.selectbox("Select a market", df_markets["Market"].tolist(), index=0)
    r = df_markets[df_markets["Market"] == market].iloc[0]

    # Simple narrative “recommendation” logic (POC-level, explainable)
    sentiment = r["Sentiment_Positive_Ratio"]
    spend = r["Avg_Outbound_Spend_$"]
    search_growth = r["Search_Growth_%"]

    st.subheader("Perception snapshot")
    c1, c2, c3 = st.columns(3)
    c1.metric("Sentiment (positive ratio)", f"{sentiment:.2f}")
    c2.metric("Search growth (12m)", f"{search_growth:.1f}%")
    c3.metric("Outbound spend proxy", f"${spend:,}")

    st.subheader("Narrative architecture (POC suggestion)")
    # Simple heuristic to pick a "positioning emphasis"
    if sentiment < 0.5:
        emphasis = "Trust, safety, openness, and reassurance"
        gap = "Address concerns + reduce perceived friction"
    elif spend > 2800:
        emphasis = "Premium experiences, luxury, exclusivity"
        gap = "Make premium proof points clearer"
    elif search_growth > 25:
        emphasis = "Make planning easy: itineraries, routes, experiences"
        gap = "Convert rising intent into action"
    else:
        emphasis = "Culture + discovery + signature experiences"
        gap = "Sharpen differentiation vs competitors"

    st.success(f"**Positioning emphasis:** {emphasis}")
    st.info(f"**Narrative gap to correct:** {gap}")

    st.subheader("Narrative messaging hierarchy (example)")
    st.write(
        "- **Primary promise:** What Saudi uniquely offers for this market\n"
        "- **Reasons to believe:** Proof points (destinations, events, access, safety)\n"
        "- **Conversion cue:** Booking/travel planning hooks (routes, packages, calendars)\n"
        "- **Objection handling:** Visa, climate/seasonality, cultural concerns\n"
    )

# -----------------------------
# 3) CPIT
# -----------------------------
elif page == "3) CPIT (Campaign Efficiency)":
    st.title("3) CPIT: Cost Per Incremental Tourist (Major Campaigns)")
    st.caption("Measure efficiency by campaign and market—based on incremental arrivals, not clicks.")

    st.markdown(
        """
**What we’re doing:**  
For major campaigns, we estimate incremental arrivals vs baseline and compute **CPIT**.

**What we’re solving:**  
Campaign performance assessed on vanity metrics, without a comparable efficiency standard.

**Key takeaway:**  
**“We can compare campaigns on one metric: cost per incremental tourist.”**
"""
    )

    st.markdown("---")

    campaign = st.selectbox("Select campaign", sorted(df_campaign["Campaign"].unique()))
    view = df_campaign[df_campaign["Campaign"] == campaign].copy()

    # Avoid infinite values
    view = view.replace([np.inf, -np.inf], np.nan)

    st.subheader("CPIT Scorecard")
    st.dataframe(view.sort_values("CPIT_$"), use_container_width=True)

    st.subheader("CPIT by market (lower is better)")
    fig = px.bar(
        view.sort_values("CPIT_$"),
        x="Market",
        y="CPIT_$",
        hover_data=["Spend_$", "Incremental_Arrivals", "Channel"]
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Quick decision lens (POC)")
    # Simple thresholds for storytelling
    q = view["CPIT_$"].dropna()
    if len(q) > 0:
        p33, p66 = np.nanpercentile(q, [33, 66])
        st.write(f"- **Scale candidates:** CPIT ≤ {p33:.0f}")
        st.write(f"- **Optimize:** {p33:.0f} < CPIT ≤ {p66:.0f}")
        st.write(f"- **Investigate / Fix:** CPIT > {p66:.0f}")
    else:
        st.write("Not enough CPIT data in this slice.")

# -----------------------------
# 4) Event impact
# -----------------------------
elif page == "4) Event Impact Measurement":
    st.title("4) Event Impact Measurement")
    st.caption("Quantify whether events drive real tourism growth—volume + revenue + retention.")

    st.markdown(
        """
**What we’re doing:**  
For each event, we estimate incremental arrivals and revenue proxies and track demand signals (search lift) and
post-event retention.

**What we’re solving:**  
Events evaluated on visibility rather than measurable tourism impact.

**Key takeaway:**  
**“We can quantify whether events drive real tourism growth — not just visibility.”**
"""
    )

    st.markdown("---")

    event = st.selectbox("Select event", sorted(df_event["Event"].unique()))
    ev = df_event[df_event["Event"] == event].copy()

    st.subheader("Event impact scorecard (by market)")
    st.dataframe(ev.sort_values("Incremental_Arrivals", ascending=False), use_container_width=True)

    st.subheader("Search lift (pre → event → post)")
    melt = ev.melt(
        id_vars=["Market"],
        value_vars=["Search_Index_Pre", "Search_Index_Event", "Search_Index_Post"],
        var_name="Window",
        value_name="Search_Index"
    )
    fig = px.line(melt, x="Window", y="Search_Index", color="Market", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Impact summary (POC)")
    total_arr = int(ev["Incremental_Arrivals"].sum())
    total_rev = int(ev["Incremental_Revenue_$"].sum())
    avg_ret = float(ev["PostEvent_Retention_Rate"].mean())

    c1, c2, c3 = st.columns(3)
    c1.metric("Total incremental arrivals (proxy)", f"{total_arr:,}")
    c2.metric("Total incremental revenue (proxy)", f"${total_rev:,.0f}")
    c3.metric("Avg post-event retention", f"{avg_ret:.2f}")

# -----------------------------
# 5) Forecast + Simulation
# -----------------------------
elif page == "5) Forecast + Simulation":
    st.title("5) 12-Month Rolling Arrival Forecast + Growth Simulation Engine")
    st.caption("Forecast + scenarios = leadership foresight. Explainable, not black-box.")

    st.markdown(
        """
**What we’re doing:**  
We generate a rolling 12-month arrivals forecast for priority markets and simulate scenario impacts
(e.g., seats up, visa friction down, sentiment shift, event boost).

**What we’re solving:**  
Reactive planning driven by last year’s numbers rather than forward signals.

**Key takeaway:**  
**“We reduce uncertainty before major decisions are made.”**
"""
    )

    st.markdown("---")

    # Select markets to forecast
    default_markets = df_markets.sort_values("Arrivals_k", ascending=False)["Market"].head(4).tolist()
    selected_markets = st.multiselect(
        "Select markets for forecast",
        df_markets["Market"].tolist(),
        default=default_markets
    )
    if not selected_markets:
        st.warning("Select at least one market.")
        st.stop()

    # Scenario controls
    st.subheader("Scenario controls (POC)")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        delta_seats = st.slider("Seat capacity change (%)", -20, 30, 10)
    with s2:
        delta_sent = st.slider("Sentiment shift (points)", -10, 10, 2)
    with s3:
        delta_visa = st.slider("Visa friction improvement (steps)", 0, 2, 1)
    with s4:
        event_boost = st.slider("Event boost (arrival uplift %)", 0, 20, 5)

    # Apply scenario to a copy of market table (simple, explainable)
    df_scn = df_markets.copy()
    df_scn.loc[df_scn["Market"].isin(selected_markets), "Seat_Growth_%"] += delta_seats
    df_scn.loc[df_scn["Market"].isin(selected_markets), "Sentiment_Positive_Ratio"] += (delta_sent / 100)
    df_scn["Sentiment_Positive_Ratio"] = df_scn["Sentiment_Positive_Ratio"].clip(0.2, 0.9)

    # Visa friction improvement means reducing friction score
    df_scn.loc[df_scn["Market"].isin(selected_markets), "Visa_Friction_1to5"] = (
        df_scn.loc[df_scn["Market"].isin(selected_markets), "Visa_Friction_1to5"] - delta_visa
    ).clip(1, 5)

    forecast = rolling_arrival_forecast(df_scn, selected_markets, months=12)

    # Apply event boost uniformly as a demo (POC)
    forecast["Forecast_Arrivals"] = (forecast["Forecast_Arrivals"] * (1 + event_boost/100)).astype(int)

    st.subheader("12-month arrivals forecast (scenario-adjusted)")
    fig = px.line(forecast, x="Month", y="Forecast_Arrivals", color="Market", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast table")
    st.dataframe(forecast, use_container_width=True)

    st.info(
        "POC note: This forecast is intentionally explainable and uses a simple combination of seasonality + momentum signals.\n"
        "In the full program, accuracy is improved through calibration, back-testing, and richer inputs."
    )

# -----------------------------
# 6) Outputs
# -----------------------------
elif page == "6) Outputs (12-month deliverables)":
    st.title("6) Outputs: What STA gets in 12 months (Pilot)")
    st.markdown(
        """
This POC is organized to reflect **concrete pilot deliverables**, not buzzwords.

### Final outputs (committed scope)
- **Clear market prioritization logic (volume + revenue balanced)**  
  → Market Opportunity Index + Tier 1/2/3 + rationale

- **Measured CPIT across major campaigns**  
  → CPIT scorecards and comparison views by campaign/market/channel

- **Visitor yield segmentation**  
  → 3–4 segments, market mix, and segment playbook

- **Event volume & revenue impact framework**  
  → Event scorecards: incremental arrivals + revenue proxy + retention effect

- **12-month rolling arrival forecast**  
  → monthly forecast with scenario-adjusted views

- **Growth simulation engine**  
  → “what-if” scenarios for seats, visa friction, sentiment, event boost

### How to use this demo
Present it like a product walkthrough:
1) Set the mandate (Landing)
2) Show the market tiering engine
3) Show perception → narrative
4) Show CPIT discipline
5) Show event impact proof
6) End with forecast + scenarios (“wow” moment)
"""
    )

    st.success("If you want, add your logo and STA branding to make it feel like an internal tool demo.")


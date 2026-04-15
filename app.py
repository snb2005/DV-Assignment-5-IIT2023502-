import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REQUIRED_COLUMNS = [
    "Date",
    "Daily_Attacks",
    "Fatalities",
    "Inflation_Pressure_Idx",
    "Estimated_CO2_Tonnes",
    "Conflict_Intensity_Index",
    "Oil_Price",
    "Gold",
    "Stock_Index",
]

EVENTS = [
    {"date": "2026-03-06", "label": "Airstrikes", "color": "#ef4444"},
    {"date": "2026-03-18", "label": "Oil Facility Attacks", "color": "#f59e0b"},
    {"date": "2026-03-30", "label": "Strait Closure", "color": "#3b82f6"},
]


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv("final_master_dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def prep_series(series: pd.Series, rolling_window: int, normalize: bool) -> pd.Series:
    result = series.astype(float)
    if rolling_window > 1:
        result = result.rolling(rolling_window, min_periods=1).mean()
    if normalize:
        std = result.std(ddof=0)
        if std and std > 0:
            result = (result - result.mean()) / std
    return result


def lag_corr_frame(df: pd.DataFrame, source: str, target: str, max_lag: int) -> pd.DataFrame:
    rows = []
    for lag in range(1, max_lag + 1):
        corr = df[source].shift(lag).corr(df[target])
        rows.append({"Lag_Days": lag, "Correlation": corr})
    return pd.DataFrame(rows)


def event_impact_table(df: pd.DataFrame, events: list[dict], window_days: int) -> pd.DataFrame:
    rows = []
    for event in events:
        event_date = pd.Timestamp(event["date"])
        pre = df[(df["Date"] < event_date) & (df["Date"] >= event_date - pd.Timedelta(days=window_days))]
        post = df[(df["Date"] > event_date) & (df["Date"] <= event_date + pd.Timedelta(days=window_days))]
        if pre.empty or post.empty:
            continue
        oil_pct = ((post["Oil_Price"].mean() - pre["Oil_Price"].mean()) / pre["Oil_Price"].mean()) * 100
        stock_pct = ((post["Stock_Index"].mean() - pre["Stock_Index"].mean()) / pre["Stock_Index"].mean()) * 100
        infl_pct = (
            (post["Inflation_Pressure_Idx"].mean() - pre["Inflation_Pressure_Idx"].mean())
            / pre["Inflation_Pressure_Idx"].mean()
        ) * 100
        rows.append(
            {
                "Event": event["label"],
                "Date": event_date.date().isoformat(),
                "Oil Price Change (%)": round(oil_pct, 2),
                "Stock Index Change (%)": round(stock_pct, 2),
                "Inflation Change (%)": round(infl_pct, 2),
            }
        )
    return pd.DataFrame(rows)


def safe_corr(series_a: pd.Series, series_b: pd.Series) -> float:
    corr_val = series_a.corr(series_b)
    if pd.isna(corr_val):
        return 0.0
    return float(corr_val)


def pct_change(first_value: float, last_value: float) -> float:
    if pd.isna(first_value) or pd.isna(last_value) or first_value == 0:
        return 0.0
    return ((last_value - first_value) / first_value) * 100


def top_corr_pairs(df: pd.DataFrame, columns: list[str], top_n: int = 3) -> list[tuple[str, str, float]]:
    pairs: list[tuple[str, str, float]] = []
    for idx, col_left in enumerate(columns):
        for col_right in columns[idx + 1:]:
            corr_val = df[col_left].corr(df[col_right])
            if pd.notna(corr_val):
                pairs.append((col_left, col_right, float(corr_val)))
    pairs.sort(key=lambda row: abs(row[2]), reverse=True)
    return pairs[:top_n]


st.set_page_config(page_title="Iran Conflict Interactive Dashboard", layout="wide")
st.title("Iran Region Conflict Dashboard: Markets, Energy, and Environment")
st.caption("Data source: final_master_dataset.csv")
st.info("Run sequence: source venv/bin/activate  ->  streamlit run app.py")

df = load_data()
missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
if missing_cols:
    st.error(f"Missing required columns in final_master_dataset.csv: {', '.join(missing_cols)}")
    st.stop()

st.sidebar.header("Interactive Controls")
raw_range = st.sidebar.date_input(
    "Select timeline",
    value=(df["Date"].min().date(), df["Date"].max().date()),
    min_value=df["Date"].min().date(),
    max_value=df["Date"].max().date(),
)

if isinstance(raw_range, tuple) and len(raw_range) == 2:
    start_date, end_date = raw_range
else:
    start_date = raw_range
    end_date = raw_range

rolling_window = st.sidebar.slider("Smoothing window (days)", min_value=1, max_value=10, value=1)
normalize_series = st.sidebar.checkbox("Normalize time-series (z-score)", value=False)
event_window = st.sidebar.slider("Event impact window (+/- days)", min_value=2, max_value=10, value=3)

mask = (df["Date"] >= pd.Timestamp(start_date)) & (df["Date"] <= pd.Timestamp(end_date))
filtered_df = df.loc[mask].copy()

if filtered_df.empty:
    st.warning("No records found in the selected date range.")
    st.stop()

snapshot_options = filtered_df["Date"].dt.strftime("%Y-%m-%d").tolist()
snapshot_label = st.sidebar.selectbox("Map snapshot date", options=snapshot_options, index=len(snapshot_options) - 1)
snapshot_row = filtered_df.loc[filtered_df["Date"] == pd.Timestamp(snapshot_label)].iloc[0]

metric_cols = st.columns(5)
metric_cols[0].metric("Peak Oil Price", f"${filtered_df['Oil_Price'].max():.2f}")
metric_cols[1].metric("Stock Drawdown", f"{filtered_df['Stock_Index'].min() - filtered_df['Stock_Index'].max():.0f} pts")
metric_cols[2].metric("Peak Conflict Intensity", f"{filtered_df['Conflict_Intensity_Index'].max():.1f}")
metric_cols[3].metric("Total CO2", f"{int(filtered_df['Estimated_CO2_Tonnes'].sum()):,} t")
metric_cols[4].metric("Avg Inflation Index", f"{filtered_df['Inflation_Pressure_Idx'].mean():.1f}")

oil_stock_corr = safe_corr(filtered_df["Oil_Price"], filtered_df["Stock_Index"])
conflict_oil_corr = safe_corr(filtered_df["Conflict_Intensity_Index"], filtered_df["Oil_Price"])
attacks_co2_corr = safe_corr(filtered_df["Daily_Attacks"], filtered_df["Estimated_CO2_Tonnes"])
conflict_infl_corr = safe_corr(filtered_df["Conflict_Intensity_Index"], filtered_df["Inflation_Pressure_Idx"])
oil_infl_corr = safe_corr(filtered_df["Oil_Price"], filtered_df["Inflation_Pressure_Idx"])
gold_oil_corr = safe_corr(filtered_df["Gold"], filtered_df["Oil_Price"])

first_row = filtered_df.iloc[0]
last_row = filtered_df.iloc[-1]
oil_change_pct = pct_change(first_row["Oil_Price"], last_row["Oil_Price"])
stock_change_pct = pct_change(first_row["Stock_Index"], last_row["Stock_Index"])
inflation_change_pct = pct_change(first_row["Inflation_Pressure_Idx"], last_row["Inflation_Pressure_Idx"])
co2_change_pct = pct_change(first_row["Estimated_CO2_Tonnes"], last_row["Estimated_CO2_Tonnes"])

peak_co2_idx = filtered_df["Estimated_CO2_Tonnes"].idxmax()
peak_co2_date = filtered_df.loc[peak_co2_idx, "Date"].date().isoformat()
peak_co2_value = int(filtered_df.loc[peak_co2_idx, "Estimated_CO2_Tonnes"])

analytical_impact_df = event_impact_table(filtered_df, EVENTS, event_window)
top_pairs = top_corr_pairs(
    filtered_df,
    ["Oil_Price", "Stock_Index", "Gold", "Inflation_Pressure_Idx", "Estimated_CO2_Tonnes"],
    top_n=3,
)

dashboard_tab, report_tab = st.tabs(["1) Dashboard", "2) Analytical Report"])

with dashboard_tab:
    st.subheader("(A) Time-Series Correlation Analysis")
    tab_a1, tab_a2, tab_a3 = st.tabs([
        "Oil Prices vs Stock Market",
        "Conflict Intensity vs Oil Price",
        "CO2 Emissions vs War Timeline",
    ])

    with tab_a1:
        oil_ts = prep_series(filtered_df["Oil_Price"], rolling_window, normalize_series)
        stock_ts = prep_series(filtered_df["Stock_Index"], rolling_window, normalize_series)
        corr_oil_stock = oil_ts.corr(stock_ts)

        fig_a1 = make_subplots(specs=[[{"secondary_y": True}]])
        fig_a1.add_trace(go.Scatter(x=filtered_df["Date"], y=oil_ts, name="Oil Price", line=dict(color="#dc2626", width=2.5)), secondary_y=False)
        fig_a1.add_trace(go.Scatter(x=filtered_df["Date"], y=stock_ts, name="Stock Index", line=dict(color="#2563eb", width=2.5)), secondary_y=True)
        fig_a1.update_layout(title="Oil vs Stock Time-Series", hovermode="x unified")
        fig_a1.update_xaxes(title="Time")
        fig_a1.update_yaxes(title="Oil Price" + (" (normalized)" if normalize_series else " ($)"), secondary_y=False)
        fig_a1.update_yaxes(title="Stock Index" + (" (normalized)" if normalize_series else ""), secondary_y=True)
        st.plotly_chart(fig_a1, use_container_width=True)
        st.caption(f"Pearson correlation (Oil, Stock): {corr_oil_stock:.3f}")

    with tab_a2:
        conflict_ts = prep_series(filtered_df["Conflict_Intensity_Index"], rolling_window, normalize_series)
        oil_ts = prep_series(filtered_df["Oil_Price"], rolling_window, normalize_series)
        corr_conflict_oil = conflict_ts.corr(oil_ts)

        fig_a2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig_a2.add_trace(go.Scatter(x=filtered_df["Date"], y=conflict_ts, name="Conflict Intensity", line=dict(color="#7c3aed", width=2.5)), secondary_y=False)
        fig_a2.add_trace(go.Scatter(x=filtered_df["Date"], y=oil_ts, name="Oil Price", line=dict(color="#ea580c", width=2.5)), secondary_y=True)
        fig_a2.update_layout(title="Conflict Intensity vs Oil Price", hovermode="x unified")
        fig_a2.update_xaxes(title="Time")
        fig_a2.update_yaxes(title="Conflict Intensity" + (" (normalized)" if normalize_series else ""), secondary_y=False)
        fig_a2.update_yaxes(title="Oil Price" + (" (normalized)" if normalize_series else " ($)"), secondary_y=True)
        st.plotly_chart(fig_a2, use_container_width=True)
        st.caption(f"Pearson correlation (Conflict, Oil): {corr_conflict_oil:.3f}")

    with tab_a3:
        attacks_ts = prep_series(filtered_df["Daily_Attacks"], rolling_window, normalize_series)
        co2_ts = prep_series(filtered_df["Estimated_CO2_Tonnes"], rolling_window, normalize_series)
        corr_co2_war = attacks_ts.corr(co2_ts)

        fig_a3 = make_subplots(specs=[[{"secondary_y": True}]])
        fig_a3.add_trace(
            go.Bar(x=filtered_df["Date"], y=attacks_ts, name="War Timeline (Daily Attacks)", marker_color="#b91c1c", opacity=0.65),
            secondary_y=False,
        )
        fig_a3.add_trace(
            go.Scatter(x=filtered_df["Date"], y=co2_ts, name="CO2 Emissions", line=dict(color="#15803d", width=2.5)),
            secondary_y=True,
        )
        fig_a3.update_layout(title="CO2 Emissions vs War Timeline", hovermode="x unified")
        fig_a3.update_xaxes(title="Time")
        fig_a3.update_yaxes(title="Daily Attacks" + (" (normalized)" if normalize_series else ""), secondary_y=False)
        fig_a3.update_yaxes(title="Estimated CO2" + (" (normalized)" if normalize_series else " (tonnes)"), secondary_y=True)
        st.plotly_chart(fig_a3, use_container_width=True)
        st.caption(f"Pearson correlation (Daily Attacks, CO2): {corr_co2_war:.3f}")

    st.subheader("(B) Multi-Axis Visualization")
    oil_multi = prep_series(filtered_df["Oil_Price"], rolling_window, normalize_series)
    stock_multi = prep_series(filtered_df["Stock_Index"], rolling_window, normalize_series)
    co2_multi = prep_series(filtered_df["Estimated_CO2_Tonnes"], rolling_window, normalize_series)

    fig_b = go.Figure()
    fig_b.add_trace(go.Scatter(x=filtered_df["Date"], y=oil_multi, mode="lines", name="Oil Price", line=dict(color="#dc2626", width=2.5), yaxis="y"))
    fig_b.add_trace(go.Scatter(x=filtered_df["Date"], y=stock_multi, mode="lines", name="Stock Index", line=dict(color="#2563eb", width=2.5), yaxis="y2"))
    fig_b.add_trace(go.Scatter(x=filtered_df["Date"], y=co2_multi, mode="lines", name="CO2 Emissions", line=dict(color="#16a34a", width=2.5), yaxis="y3"))
    fig_b.update_layout(
        title="Time vs Oil, Stock, and CO2",
        xaxis=dict(title="Time"),
        yaxis=dict(
            title=dict(
                text="Oil Price" + (" (normalized)" if normalize_series else " ($)"),
                font=dict(color="#dc2626"),
            ),
            tickfont=dict(color="#dc2626"),
        ),
        yaxis2=dict(
            title=dict(
                text="Stock Index" + (" (normalized)" if normalize_series else ""),
                font=dict(color="#2563eb"),
            ),
            tickfont=dict(color="#2563eb"),
            overlaying="y",
            side="right",
        ),
        yaxis3=dict(
            title=dict(
                text="CO2" + (" (normalized)" if normalize_series else " (tonnes)"),
                font=dict(color="#16a34a"),
            ),
            tickfont=dict(color="#16a34a"),
            anchor="free",
            overlaying="y",
            side="right",
            position=0.93,
        ),
        hovermode="x unified",
    )
    st.plotly_chart(fig_b, use_container_width=True)

    st.subheader("(C) Geospatial Visualization")
    st.caption("Layers: conflict zones in Iran region, Strait of Hormuz oil route, and pollution hotspots")

    conflict_points = pd.DataFrame(
        {
            "Location": ["Tehran", "Isfahan", "Khuzestan", "Bushehr", "Bandar Abbas"],
            "Lat": [35.6892, 32.6546, 31.3183, 28.9234, 27.1832],
            "Lon": [51.3890, 51.6680, 48.6706, 50.8203, 56.2666],
            "Weight": [1.0, 0.75, 1.1, 0.85, 1.2],
        }
    )
    conflict_points["Intensity"] = snapshot_row["Conflict_Intensity_Index"] * conflict_points["Weight"]

    pollution_points = pd.DataFrame(
        {
            "Location": ["Asaluyeh", "Abadan", "Bandar Abbas", "Kharg Island"],
            "Lat": [27.4762, 30.3473, 27.1832, 29.2614],
            "Lon": [52.6096, 48.2934, 56.2666, 50.3300],
            "Weight": [1.2, 1.0, 1.1, 1.25],
        }
    )
    pollution_points["CO2_Proxy"] = snapshot_row["Estimated_CO2_Tonnes"] * pollution_points["Weight"]

    route_lons = [50.3300, 52.6096, 56.2666, 56.45, 58.30]
    route_lats = [29.2614, 27.4762, 27.1832, 26.40, 25.60]

    fig_c = go.Figure()
    fig_c.add_trace(
        go.Scattergeo(
            lon=route_lons,
            lat=route_lats,
            mode="lines+markers",
            line=dict(width=3, color="#f59e0b"),
            marker=dict(size=5, color="#f59e0b"),
            name="Oil Route (Strait of Hormuz)",
            hovertemplate="Oil route node<extra></extra>",
        )
    )
    fig_c.add_trace(
        go.Scattergeo(
            lon=conflict_points["Lon"],
            lat=conflict_points["Lat"],
            text=conflict_points["Location"],
            mode="markers",
            marker=dict(
                size=8 + (conflict_points["Intensity"] / max(conflict_points["Intensity"].max(), 1)) * 18,
                color="#dc2626",
                opacity=0.8,
                line=dict(color="#7f1d1d", width=1),
            ),
            name="Conflict Zones",
            hovertemplate="%{text}<br>Conflict Index Proxy: %{marker.size:.1f}<extra></extra>",
        )
    )
    fig_c.add_trace(
        go.Scattergeo(
            lon=pollution_points["Lon"],
            lat=pollution_points["Lat"],
            text=pollution_points["Location"],
            mode="markers",
            marker=dict(
                size=7 + (pollution_points["CO2_Proxy"] / max(pollution_points["CO2_Proxy"].max(), 1)) * 20,
                color="#16a34a",
                opacity=0.72,
                line=dict(color="#14532d", width=1),
            ),
            name="Pollution Hotspots",
            hovertemplate="%{text}<br>CO2 Proxy: %{marker.size:.1f}<extra></extra>",
        )
    )
    fig_c.update_layout(
        title=f"Iran Region Geospatial Risk Map ({snapshot_label})",
        geo=dict(
            scope="asia",
            projection_type="natural earth",
            showland=True,
            landcolor="rgb(242, 239, 233)",
            showcountries=True,
            countrycolor="rgb(120, 120, 120)",
            showocean=True,
            oceancolor="rgb(214, 234, 248)",
            lataxis=dict(range=[23, 38]),
            lonaxis=dict(range=[44, 62]),
        ),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(fig_c, use_container_width=True)

    st.subheader("(D) Heatmap Correlation Matrix")
    heatmap_vars = ["Oil_Price", "Stock_Index", "Gold", "Inflation_Pressure_Idx", "Estimated_CO2_Tonnes"]
    corr_matrix = filtered_df[heatmap_vars].corr().round(2)

    fig_d = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlation Matrix: Oil, Stock, Gold, Inflation, CO2",
    )
    st.plotly_chart(fig_d, use_container_width=True)

    st.subheader("(E) Event Impact Visualization")
    fig_e = make_subplots(specs=[[{"secondary_y": True}]])
    fig_e.add_trace(go.Scatter(x=filtered_df["Date"], y=filtered_df["Oil_Price"], mode="lines", name="Oil Price", line=dict(color="#dc2626", width=2.5)), secondary_y=False)
    fig_e.add_trace(go.Scatter(x=filtered_df["Date"], y=filtered_df["Stock_Index"], mode="lines", name="Stock Index", line=dict(color="#2563eb", width=2.5)), secondary_y=True)

    for event in EVENTS:
        event_date = pd.Timestamp(event["date"])
        if filtered_df["Date"].min() <= event_date <= filtered_df["Date"].max():
            fig_e.add_vline(x=event_date, line_dash="dash", line_width=2, line_color=event["color"])
            fig_e.add_annotation(
                x=event_date,
                y=1.08,
                yref="paper",
                text=event["label"],
                showarrow=False,
                bgcolor="white",
                bordercolor=event["color"],
            )

    fig_e.update_layout(title="Major Event Annotations: Airstrikes, Facility Attacks, Strait Closure", hovermode="x unified")
    fig_e.update_xaxes(title="Time")
    fig_e.update_yaxes(title_text="Oil Price ($)", secondary_y=False)
    fig_e.update_yaxes(title_text="Stock Index", secondary_y=True)
    st.plotly_chart(fig_e, use_container_width=True)

    if analytical_impact_df.empty:
        st.caption("Not enough pre/post data points in current date filter to estimate event impact.")
    else:
        st.dataframe(analytical_impact_df, use_container_width=True, hide_index=True)

    st.subheader("(F) Causal / Lag Analysis")
    lag_col1, lag_col2 = st.columns(2)
    oil_to_stock_lag = lag_col1.slider("Oil -> Stock lag (days)", min_value=3, max_value=5, value=4)
    conflict_to_infl_lag = lag_col2.slider("Conflict -> Inflation lag (days)", min_value=7, max_value=30, value=14)

    lag_os = filtered_df[["Date", "Oil_Price", "Stock_Index"]].copy()
    lag_os[f"Oil_Price_Lag_{oil_to_stock_lag}"] = lag_os["Oil_Price"].shift(oil_to_stock_lag)
    lag_os_plot = lag_os.dropna(subset=[f"Oil_Price_Lag_{oil_to_stock_lag}", "Stock_Index"])

    lag_ci = filtered_df[["Date", "Conflict_Intensity_Index", "Inflation_Pressure_Idx"]].copy()
    lag_ci[f"Conflict_Lag_{conflict_to_infl_lag}"] = lag_ci["Conflict_Intensity_Index"].shift(conflict_to_infl_lag)
    lag_ci_plot = lag_ci.dropna(subset=[f"Conflict_Lag_{conflict_to_infl_lag}", "Inflation_Pressure_Idx"])

    plot_col1, plot_col2 = st.columns(2)
    with plot_col1:
        fig_f1 = px.scatter(
            lag_os_plot,
            x=f"Oil_Price_Lag_{oil_to_stock_lag}",
            y="Stock_Index",
            title=f"Lag Plot: Oil (t-{oil_to_stock_lag}) -> Stock (t)",
            opacity=0.8,
        )
        fig_f1.update_traces(marker=dict(color="#7c3aed", size=10, line=dict(color="#4c1d95", width=1)))
        st.plotly_chart(fig_f1, use_container_width=True)

    with plot_col2:
        fig_f2 = px.scatter(
            lag_ci_plot,
            x=f"Conflict_Lag_{conflict_to_infl_lag}",
            y="Inflation_Pressure_Idx",
            title=f"Lag Plot: Conflict (t-{conflict_to_infl_lag}) -> Inflation (t)",
            opacity=0.8,
        )
        fig_f2.update_traces(marker=dict(color="#0891b2", size=10, line=dict(color="#155e75", width=1)))
        st.plotly_chart(fig_f2, use_container_width=True)

    corr_os = lag_corr_frame(filtered_df, "Oil_Price", "Stock_Index", 10)
    corr_ci = lag_corr_frame(filtered_df, "Conflict_Intensity_Index", "Inflation_Pressure_Idx", 30)

    fig_f3 = make_subplots(rows=1, cols=2, subplot_titles=["Oil -> Stock Lag Correlation", "Conflict -> Inflation Lag Correlation"])
    fig_f3.add_trace(
        go.Scatter(x=corr_os["Lag_Days"], y=corr_os["Correlation"], mode="lines+markers", line=dict(color="#7c3aed", width=2), name="Oil->Stock"),
        row=1,
        col=1,
    )
    fig_f3.add_trace(
        go.Scatter(x=corr_ci["Lag_Days"], y=corr_ci["Correlation"], mode="lines+markers", line=dict(color="#0891b2", width=2), name="Conflict->Inflation"),
        row=1,
        col=2,
    )
    fig_f3.add_vline(x=oil_to_stock_lag, line_dash="dash", line_color="#7c3aed", row=1, col=1)
    fig_f3.add_vline(x=conflict_to_infl_lag, line_dash="dash", line_color="#0891b2", row=1, col=2)
    fig_f3.update_xaxes(title="Lag (days)", row=1, col=1)
    fig_f3.update_xaxes(title="Lag (days)", row=1, col=2)
    fig_f3.update_yaxes(title="Correlation", row=1, col=1)
    fig_f3.update_yaxes(title="Correlation", row=1, col=2)
    fig_f3.update_layout(height=450, showlegend=False)
    st.plotly_chart(fig_f3, use_container_width=True)

    selected_oil_corr = filtered_df["Oil_Price"].shift(oil_to_stock_lag).corr(filtered_df["Stock_Index"])
    selected_conflict_corr = filtered_df["Conflict_Intensity_Index"].shift(conflict_to_infl_lag).corr(filtered_df["Inflation_Pressure_Idx"])
    st.caption(
        f"Selected lag correlations -> Oil(t-{oil_to_stock_lag}) to Stock(t): {selected_oil_corr:.3f} | "
        f"Conflict(t-{conflict_to_infl_lag}) to Inflation(t): {selected_conflict_corr:.3f}"
    )

with report_tab:
    st.subheader("2) Analytical Report")
    st.caption("Auto-generated from the selected timeline filter.")

    with st.expander("Key Patterns", expanded=True):
        if top_pairs:
            for col_left, col_right, corr_val in top_pairs:
                st.markdown(f"- Strong relationship: **{col_left} vs {col_right}** (r = {corr_val:.2f})")
        st.markdown(f"- Oil moved **{oil_change_pct:+.2f}%** across the selected timeline.")
        st.markdown(f"- Stock index moved **{stock_change_pct:+.2f}%** across the selected timeline.")
        st.markdown(f"- Conflict intensity vs oil correlation: **r = {conflict_oil_corr:.2f}**.")
        st.markdown(f"- Daily attacks vs CO2 correlation: **r = {attacks_co2_corr:.2f}**.")

    with st.expander("Economic Insights", expanded=True):
        stock_relation = "inverse" if oil_stock_corr < 0 else "positive"
        inflation_relation = "positive" if conflict_infl_corr > 0 else "inverse"
        st.markdown(
            f"- Oil and stock market show a **{stock_relation} linkage** (r = {oil_stock_corr:.2f}), indicating market stress during oil shocks."
        )
        st.markdown(
            f"- Conflict and inflation show a **{inflation_relation} linkage** (r = {conflict_infl_corr:.2f}), signaling pass-through from instability to prices."
        )
        st.markdown(f"- Oil vs inflation correlation is **r = {oil_infl_corr:.2f}**, with inflation shifting **{inflation_change_pct:+.2f}%** over the period.")
        st.markdown(f"- Gold vs oil correlation is **r = {gold_oil_corr:.2f}**, useful for tracking hedge behavior during conflict shocks.")
        if not analytical_impact_df.empty:
            avg_oil_impact = analytical_impact_df["Oil Price Change (%)"].mean()
            avg_stock_impact = analytical_impact_df["Stock Index Change (%)"].mean()
            st.markdown(
                f"- Event windows imply average post-event change of **{avg_oil_impact:+.2f}%** in oil and **{avg_stock_impact:+.2f}%** in stocks."
            )

    with st.expander("Environmental Insights", expanded=True):
        total_co2 = int(filtered_df["Estimated_CO2_Tonnes"].sum())
        co2_relation = "rises with" if attacks_co2_corr > 0 else "falls with"
        st.markdown(f"- Total estimated emissions in this window are **{total_co2:,} tonnes CO2**.")
        st.markdown(f"- Emissions generally **{co2_relation}** attack intensity (r = {attacks_co2_corr:.2f}).")
        st.markdown(f"- Peak estimated CO2 day is **{peak_co2_date}** at **{peak_co2_value:,} tonnes**.")
        st.markdown(f"- CO2 changed **{co2_change_pct:+.2f}%** between the start and end of the selected timeline.")

    report_lines = [
        "Analytical Report - Iran Conflict Dashboard",
        f"Date Window: {start_date} to {end_date}",
        "",
        "Key Patterns:",
        f"- Oil vs Stock correlation: {oil_stock_corr:.3f}",
        f"- Conflict vs Oil correlation: {conflict_oil_corr:.3f}",
        f"- Attacks vs CO2 correlation: {attacks_co2_corr:.3f}",
        f"- Oil timeline change: {oil_change_pct:+.2f}%",
        f"- Stock timeline change: {stock_change_pct:+.2f}%",
        "",
        "Economic Insights:",
        f"- Conflict vs Inflation correlation: {conflict_infl_corr:.3f}",
        f"- Oil vs Inflation correlation: {oil_infl_corr:.3f}",
        f"- Gold vs Oil correlation: {gold_oil_corr:.3f}",
        f"- Inflation timeline change: {inflation_change_pct:+.2f}%",
        "",
        "Environmental Insights:",
        f"- Total CO2 in window: {total_co2:,} tonnes",
        f"- CO2 timeline change: {co2_change_pct:+.2f}%",
        f"- Peak CO2 date: {peak_co2_date} ({peak_co2_value:,} tonnes)",
    ]

    st.download_button(
        "Download Analytical Report (.txt)",
        data="\n".join(report_lines),
        file_name=f"analytical_report_{start_date}_to_{end_date}.txt",
        mime="text/plain",
    )
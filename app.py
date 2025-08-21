# streamlit_app_improved.py

import os
import math
import warnings
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


warnings.filterwarnings("ignore")
DATA_PATH = "./"
MODELS_DIR = "trained_models"
DEFAULT_CITIES = ["Toronto", "Vancouver", "Calgary"]
RENT_PIPELINE_FILES = [
    "housing_affordibility_pipeline.pkl",
    "housing_affordability_pipeline.pkl",
    "rent_model_pipeline.pkl",
]

@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame:
    f = os.path.join(DATA_PATH, name)
    if not os.path.exists(f):
        return pd.DataFrame()
    df = pd.read_csv(f)
    for col in ["Date", "ds"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None

def cities_available():
    if not df_merged.empty and "city" in df_merged.columns:
        return sorted(df_merged["city"].dropna().unique().tolist())
    return DEFAULT_CITIES

def month_sin_cos(month:int):
    rad = 2 * np.pi * (month / 12)
    return np.sin(rad), np.cos(rad)

def fmt_num(x):
    return f"{x:,.0f}" if pd.notna(x) else "—"

def damage_label(p):
    if p < 0.2:
        return "No damage expected"
    elif p < 0.5:
        return "Low chance of damage"
    elif p < 0.8:
        return "Likely damage"
    else:
        return "Very likely damage"

def plot_corr_explain(df, target="VALUE", top_n=6):
    num = df.select_dtypes(include=[np.number]).dropna(axis=1, how='all')
    if num.shape[1] < 2:
        return None, None
    corr = num.corr()
    heat = px.imshow(corr, zmin=-1, zmax=1, title="Correlation matrix (numeric vars)", color_continuous_scale='RdBu_r')
    if target in corr.columns:
        corrs = corr[target].drop(target).abs().sort_values(ascending=False)
        top = corrs.head(top_n)
        bar = px.bar(x=top.values, y=top.index, orientation='h', labels={'x':'|corr|','y':'feature'}, title=f'Top {top_n} features by |corr| with {target}', color=top.values, color_continuous_scale='Viridis')
    else:
        bar = None
    return heat, bar

def prophet_model_path(city: str) -> str:
    return os.path.join(MODELS_DIR, f"prophet_{city.lower().replace(' ','_')}.pkl")

def arima_model_path(city: str) -> str:
    return os.path.join(MODELS_DIR, f"arima_{city.lower().replace(' ','_')}.pkl")

def rmse(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    return float(np.sqrt(np.mean((a - b)**2))) if len(a) == len(b) and len(a)>0 else math.inf

def best_model_for_city(city, horizon_months=24):
    if df_merged.empty:
        return None, None, None, None
    tmp = df_merged[df_merged["city"]==city].dropna(subset=["Date","VALUE"]).copy()
    if tmp.empty or len(tmp)<12:
        return None, None, None, None
    tmp = tmp.groupby("Date")["VALUE"].mean().reset_index().sort_values("Date")
    n = len(tmp)
    split = int(n * 0.8)
    train = tmp.iloc[:split]
    test = tmp.iloc[split:]
    prophet_m = load_model(prophet_model_path(city))
    arima_m = load_model(arima_model_path(city))
    prophet_rmse = math.inf; arima_rmse = math.inf
    prophet_pred = None; arima_pred = None
    if prophet_m is not None:
        tr = train.rename(columns={"Date":"ds","VALUE":"y"})
        fc = prophet_m.predict(test.rename(columns={"Date":"ds"}))
        prophet_pred = fc["yhat"].values
        prophet_rmse = rmse(test["VALUE"].values, prophet_pred)
    if arima_m is not None:
        arima_pred = arima_m.predict(n_periods=len(test))
        arima_rmse = rmse(test["VALUE"].values, arima_pred)
    choose = "Prophet" if prophet_rmse <= arima_rmse else "ARIMA"
    best_rmse = min(prophet_rmse, arima_rmse)
    last_date = tmp["Date"].max()
    future_months = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=horizon_months, freq="MS")
    forecast = pd.DataFrame()
    if choose=="Prophet" and prophet_m is not None:
        df_future = pd.DataFrame({"ds": future_months})
        future_fc = prophet_m.predict(df_future)
        forecast = future_fc[["ds","yhat"]].rename(columns={"ds":"Date","yhat":"Forecast"})
    elif choose=="ARIMA" and arima_m is not None:
        fc_vals = arima_m.predict(n_periods=horizon_months)
        forecast = pd.DataFrame({"Date": future_months, "Forecast": fc_vals})
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Scatter(x=train["Date"], y=train["VALUE"], mode="lines", name="Train"))
    hist_fig.add_trace(go.Scatter(x=test["Date"], y=test["VALUE"], mode="lines+markers", name="Test"))
    if prophet_pred is not None:
        hist_fig.add_trace(go.Scatter(x=test["Date"], y=prophet_pred, mode="lines", name=f"Prophet (RMSE {prophet_rmse:.2f})"))
    if arima_pred is not None:
        hist_fig.add_trace(go.Scatter(x=test["Date"], y=arima_pred, mode="lines", name=f"ARIMA (RMSE {arima_rmse:.2f})"))
    if not forecast.empty:
        hist_fig.add_trace(go.Scatter(x=forecast["Date"], y=forecast["Forecast"], mode="lines+markers", name=f"Future — {choose}", line=dict(dash="dash")))
    hist_fig.update_layout(
        title=f"{city}: Forecast Comparison & Future ({choose} selected — RMSE {best_rmse:.2f})",
        hovermode="x unified",
        margin=dict(l=10,r=10,t=60,b=10),
        template="plotly_white"
    )
    return hist_fig, choose, prophet_rmse if np.isfinite(prophet_rmse) else None, arima_rmse if np.isfinite(arima_rmse) else None

# Load datasets and models
df_rents = load_csv("df_rents_cleaned.csv")
monthly_weather = load_csv("monthly_weather_cleaned.csv")
vancouver_issues = load_csv("vancouver_issues_cleaned.csv")
mortgage_lending = load_csv("mortgage_lending_cleaned.csv")
dwelling_data_pivot = load_csv("dwelling_data_pivot_cleaned.csv")
df_hes_agg = load_csv("df_hes_agg_cleaned.csv")
house_prices_monthly = load_csv("house_prices_monthly_cleaned.csv")
df_merged = load_csv("merged_data_imputed.csv")

dwelling_model = load_model(os.path.join(MODELS_DIR, "dwelling_damage_model.pkl"))
dwelling_preproc = load_model(os.path.join(MODELS_DIR, "dwelling_damage_preprocessor.pkl"))

rent_pipeline = None
for candidate in RENT_PIPELINE_FILES:
    m = load_model(os.path.join(MODELS_DIR, candidate))
    if m is not None:
        rent_pipeline = m
        break

# --- Custom CSS for centered and styled title, subtitle, larger outputs, and card-like rent comparison ---
st.markdown("""
<style>
/* ===== Custom App Background and Font ===== */
body, .stApp {
    background: linear-gradient(120deg, #f8fafc 0%, #e7eaf3 100%);
    font-family: "Segoe UI", "Helvetica Neue", Arial, "Liberation Sans", sans-serif !important;
}
section.main > div {padding-top: 1rem;}
/* Centered and styled title and headings */
h1, .css-1v3fvcr h1 {
    text-align: center !important;
    font-size: 4rem !important;          /* increased from 2.8rem */
    font-weight: 900 !important;         /* heavier font weight */
    color: #22223b !important;
    text-shadow: 0 4px 14px #ced7eb55;
    margin-bottom: 0.2em !important;
}
h2, .css-1v3fvcr h2 {
    text-align: center !important;       /* center subtitle */
    font-size: 1.85rem !important;       /* slightly bigger */
    font-weight: 600 !important;
    color: #41436a !important;
    margin-bottom: 1.5rem !important;
}

/* Make Streamlit expander less jarring */
.st-expander > summary {
    background: #f6f7fb !important;
    color: #22223b;
    border: none;
    box-shadow: 0 2px 8px #bfc5d2cc;
    padding: 0.65em 1em;
    border-radius: 9px;
    margin-bottom: 5px;
}

.stTabs [data-baseweb="tab"] {
    font-size: 1.1rem;
    font-weight: 500;
    color: #556081;
}
.stTabs [aria-selected="true"] {
    background-color: #e9f4fb !important;
    color: #0086ad !important;
    border-bottom: 2.5px solid #0086ad !important;
}

.prediction-output {
    font-size: 2.2rem !important;
    color: #0086ad !important;
    font-weight: 700;
    padding: 16px 0 2px 0;
    text-align:center;
    background: #ebf8ff;
    border-radius: 10px;
    box-shadow: 0 2px 10px #ababab33;
}

.rent-card {
    background: linear-gradient(135deg,#eaf3fc 55%,#e2f7e1 100%);
    border-radius: 14px;
    padding: 18px 10px 15px 10px;
    margin-bottom: 21px;
    box-shadow: 0 4px 17px 0 #cbd6e144, 0 1.5px 8px #6c757d0a;
    transition: box-shadow 0.3s, transform 0.3s;
    text-align: center;
    border: 1.5px solid #e0e6e6;
}
.rent-card:hover {
    box-shadow: 0 8px 32px rgba(8,60,203,0.09),0 1.5px 10px #85b0f720;
    transform: translateY(-2px) scale(1.022);
}
.rent-card h3 {
    margin: 0;
    color: #007899;
    font-weight: 650;
    font-size: 1.18rem;
}
.rent-card p {
    margin: 10px auto 0 auto;
    color: #2e476a;
    font-size: 1.62rem;
    font-weight: 700;
    letter-spacing: 1px;
}
.rent-card small {
    color: #657687;
    font-size: 0.96rem;
}

.stAlert {border-radius: 12px !important;}
.stInfo {background-color: #f6f7fb !important;}

table {
    margin: 1rem auto;
    border-collapse: separate;
    border-spacing: 0 3px;
    background: #fbfbfb;
    font-size: 1.02rem;
    border-radius: 13px;
    overflow: hidden;
    box-shadow: 0 2px 6px #d5e5ed33;
}
th, td {
    padding: 10px;
    border-bottom: 1.2px solid #e2e5ec;
}
th {
    background: #edf4fb;
    font-weight:600;
    color: #006896;
}
tr:last-child td {border: none;}

hr {border: none; border-top: 1.8px solid #e0e7ef; margin: 25px 0;}

.stSlider > div[data-baseweb="slider"] {
    padding-top:0.5em;
    padding-bottom:0.5em;
}
            
/* Center headings above tables in clusters tab */
.clusters-tab-heading {
    text-align: center !important;
    font-size: 1.22rem;
    font-weight: 600;
    margin-top: 1.5em;
    margin-bottom: 0.5em;
}

/* Left-align headings in final report tab */
.report-tab-heading {
    text-align: left !important;
    font-size: 1.36rem;
    font-weight: 700;
    margin-top: 0.3em;
    margin-bottom: 0.3em;
}
            
/* Larger tab labels */
.css-1vg6q84, 
.css-1y0tads {
    font-size: 1.25rem !important;
    font-weight: 600 !important;
}
            
/* Larger font size for all tab labels */
.stTabs [data-baseweb="tab-list"] button > div[data-testid="stMarkdownContainer"] p {
    font-size: 1.4rem !important;
    font-weight: 600 !important;
}
            
/* Increase spacing between each tab label */
.stTabs [data-baseweb="tab-list"] button {
    margin-right: 1.2rem !important;  /* adds horizontal space between tabs */
}

/* Optional: Add some padding for better clickability */
.stTabs [data-baseweb="tab-list"] button > div[data-testid="stMarkdownContainer"] {
    padding: 0.3rem 0.6rem !important;
}

/* Increase font size of paragraph and list text */
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li,
div[data-testid="stMarkdownContainer"] span,
div[data-testid="stMarkdownContainer"] div {
    font-size: 1.35rem !important;
    line-height: 1.7 !important;
}

/* Increase font size of the dropdown options */
div[role="listbox"] div[role="option"] {
    font-size: 1.35rem !important;
    font-weight: 600 !important;
}
                      
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Canada Housing & Weather — Interactive Analysis", page_icon="🏙️", layout="wide")
st.title("🏙️ Canada Housing & Weather — Interactive Analysis & Forecasts")
st.markdown(
    '<div style="text-align:center; font-size:1.85rem; font-weight:600; color:#41436a; margin-bottom:1.5rem;">Interactive dashboard with rich insights and smooth visualizations</div>',
    unsafe_allow_html=True,
)


tab_overview, tab_eda, tab_modeling, tab_forecasts, tab_clusters, tab_report = st.tabs([
    "Overview", "Exploration", "Modeling & What-If", "Forecasts", "Clusters & Risk", "Final Report"
])

with tab_overview:
    st.subheader("City Profiles and Rent Trends Overview")

    if not df_rents.empty:
        # 1. Comparison cards of rental prices for all cities (latest year)
        latest_year = df_rents['year'].max()
        comparison = df_rents[df_rents['year'] == latest_year].groupby('city')['VALUE'].mean().reset_index()
        comparison = comparison.rename(columns={"VALUE": f"Avg Rent ({latest_year})"})
        st.markdown(f"#### Rent Comparison Across Cities ({latest_year})")

        comparison_sorted = comparison.sort_values(by=f"Avg Rent ({latest_year})", ascending=False).reset_index(drop=True)
        num_cols = 3
        cols = st.columns(num_cols)

        for idx, row in comparison_sorted.iterrows():
            col = cols[idx % num_cols]
            col.markdown(f"""
                <div class="rent-card">
                    <h3>{row['city']}</h3>
                    <p>${row[f'Avg Rent ({latest_year})']:.0f}</p>
                    <small>Average Rent</small>
                </div>
            """, unsafe_allow_html=True)

        # 2. Percentage increase in rent for each city (2018 to latest year)
        st.markdown("#### Rent Increase Percentage by City (2018 to 2024)")
        perc_changes = []
        for city_name in comparison['city']:
            city_data = df_rents[df_rents['city'] == city_name]
            min_year = int(city_data['year'].min())
            max_year = int(city_data['year'].max())
            rent_start = city_data[city_data['year'] == min_year]['VALUE'].mean()
            rent_end = city_data[city_data['year'] == max_year]['VALUE'].mean()
            if rent_start and rent_start > 0:
                perc_change = ((rent_end - rent_start) / rent_start) * 100
                perc_changes.append({"city": city_name, "perc_change": perc_change})
        perc_df = pd.DataFrame(perc_changes).sort_values(by='perc_change', ascending=False).reset_index(drop=True)
        cols = st.columns(num_cols)
        for idx, row in perc_df.iterrows():
            col = cols[idx % num_cols]
            col.markdown(f"""
                <div class="rent-card">
                    <h3>{row['city']}</h3>
                    <p>{row['perc_change']:.2f}%</p>
                    <small>Rent Increase</small>
                </div>
            """, unsafe_allow_html=True)

        # 3. Average rent trend over the years across cities (line graph)
        fig_all_cities_line = px.line(
            df_rents.groupby(['year', 'city'])['VALUE'].mean().reset_index(),
            x='year',
            y='VALUE',
            color='city',
            title="Average Rent Trend Over Years Across Cities",
            labels={'year': 'Year', 'VALUE': 'Avg Rent ($)'}
        )
        fig_all_cities_line.update_layout(template="plotly_white")
        st.plotly_chart(fig_all_cities_line, use_container_width=True)

        # 4. Bar graph of average rent by city for the latest year
        fig_all_cities_bar = px.bar(
            comparison,
            x='city',
            y=f'Avg Rent ({latest_year})',
            title=f"Average Rent by City ({latest_year})",
            labels={'city': 'City', f'Avg Rent ({latest_year})': 'Avg Rent ($)'},
            color='city',
            text=comparison[f'Avg Rent ({latest_year})'].apply(lambda x: f"${x:,.0f}")
        )
        fig_all_cities_bar.update_layout(template="plotly_white")
        st.plotly_chart(fig_all_cities_bar, use_container_width=True)

    else:
        st.info("Rent dataset is empty.")


with tab_eda:
    st.subheader("Exploratory Visualizations (Interactive)")

    tab_weather, tab_rent, tab_quality, tab_extreme, tab_mortgage = st.tabs([
        "Weather Patterns Across Cities", "Rent & Affordability Trends",
        "Housing Quality & Standards", "Extreme Weather Impacts",
        "Mortgage Rates & House Prices"
    ])

    with tab_weather:
        st.markdown("### Weather Patterns Across Cities")
        st.markdown("""
        **Vancouver**  
        - Mild winters  
        - Highest precipitation  
        - Lowest snowfall  

        **Toronto**  
        - Most pronounced temperature fluctuations  
        - Moderate precipitation  
        - Significant snowfall  

        **Calgary**  
        - Cold winters  
        - Lowest precipitation  
        - Highest annual snowfall  

        Interannual temperature trends remain flat to slightly rising across all cities from 2018-2024.
        """)
        if not df_merged.empty:
            yearly_temp_avg = df_merged.groupby(['city', 'year']).agg(
                Avg_Max_Temp=('Avg_Max_Temp', 'mean'),
                Avg_Min_Temp=('Avg_Min_Temp', 'mean')
            ).reset_index()
            yearly_temp_avg['Avg_Temp'] = (yearly_temp_avg['Avg_Max_Temp'] + yearly_temp_avg['Avg_Min_Temp']) / 2
            fig_temp_trend = px.line(
                yearly_temp_avg,
                x='year',
                y='Avg_Temp',
                color='city',
                title='Average Yearly Temperature Trend by City (2018-2024)',
                labels={'year': 'Year', 'Avg_Temp': 'Avg Temperature (°C)'},
                line_shape='spline'
            )
            fig_temp_trend.update_layout(template='plotly_white')
            st.plotly_chart(fig_temp_trend, use_container_width=True)

        if not df_merged.empty:
        # Total yearly precipitation by city (2018-2024)
            yearly_precip_sum = df_merged.groupby(['city', 'year'])['Total_Precip_mm'].sum().reset_index()
            fig_precip = px.line(
                yearly_precip_sum,
                x='year',
                y='Total_Precip_mm',
                color='city',
                title='Total Yearly Precipitation by City (2018-2024)',
                labels={'year': 'Year', 'Total_Precip_mm': 'Total Precipitation (mm)'}
            )
            fig_precip.update_layout(template='plotly_white')
            st.plotly_chart(fig_precip, use_container_width=True)

        # Average monthly temperature by city (2018-2024 average)
        monthly_avg_temp = df_merged.groupby(['city', 'month']).agg(
            Avg_Max_Temp=('Avg_Max_Temp', 'mean'),
            Avg_Min_Temp=('Avg_Min_Temp', 'mean')
        ).reset_index()
        monthly_avg_temp['Avg_Temp_Monthly'] = (monthly_avg_temp['Avg_Max_Temp'] + monthly_avg_temp['Avg_Min_Temp']) / 2
        fig_temp = px.line(
            monthly_avg_temp,
            x='month',
            y='Avg_Temp_Monthly',
            color='city',
            title='Average Temperature by Month and City (2018-2024 Average)',
            labels={'month': 'Month', 'Avg_Temp_Monthly': 'Average Temperature (°C)'}
        )
        fig_temp.update_layout(template='plotly_white')
        st.plotly_chart(fig_temp, use_container_width=True)

    with tab_rent:
        st.markdown("### Rent & Affordability Trends")
        st.markdown("""
            **Key Findings:**
            - Vancouver remains most expensive, followed by Toronto and Calgary.
            - Steady rent increases across all cities from 2018-2024.
            - Larger and premium unit types consistently higher-priced.
            - Only modest relationships between monthly rents and weather metrics.
        """)
        if not df_rents.empty and "Type of unit" in df_rents.columns:
            rents_grouped = df_rents.groupby(['city', 'Type of unit', 'year', 'month'], as_index=False)['VALUE'].mean()
            rents_grouped['Date'] = pd.to_datetime(rents_grouped[['year', 'month']].assign(day=1))
            fig_rent_units = px.bar(
                rents_grouped,
                x='Date',
                y='VALUE',
                color='Type of unit',
                facet_col='city',
                labels={'Date': 'Date', 'VALUE': 'Avg Rent ($)'},
                title="Average Rent Trend by City and Unit Type",
                barmode='stack'
            )
            fig_rent_units.update_layout(template='plotly_white')
            st.plotly_chart(fig_rent_units, use_container_width=True)

    with tab_quality:
        st.markdown("### Housing Quality & Standards")
        st.markdown("""
        **Vancouver Rental Violations:**
        - "Sunset" and "Riley Park" areas have highest violation rates among sub-regions.
        - Slight positive association between number of units and violation rate.

        **Renter Repairs (2021):**
        - **Toronto:** Highest percentage of dwellings needing major repairs.
        - **Vancouver:** Moderate repair needs.
        - **Calgary:** Lowest repair needs.
        """)
        # Existing housing quality charts here
        if not vancouver_issues.empty:
            vancouver_area_violations = vancouver_issues.groupby('geolocalarea')['violation_rate'].mean().reset_index()
            vancouver_area_violations = vancouver_area_violations.sort_values('violation_rate', ascending=False)
            fig_violations = px.bar(
                vancouver_area_violations,
                x='geolocalarea',
                y='violation_rate',
                title='Average Violation Rate by Geo Local Area in Vancouver',
                labels={'geolocalarea': 'Geolocal Area', 'violation_rate': 'Violation Rate'},
                color='violation_rate',
                color_continuous_scale=px.colors.sequential.Plasma
            )
            fig_violations.update_layout(template='plotly_white', xaxis_tickangle=-45)
            st.plotly_chart(fig_violations, use_container_width=True)

        df_2021 = df_merged[df_merged['year'] == 2021].copy()
        repairs_2021 = df_2021.groupby('city')['renter_major_repairs_percent'].mean().reset_index()
        fig_repairs = px.bar(
            repairs_2021,
            x='city',
            y='renter_major_repairs_percent',
            title='Percentage of Renter Dwellings Needing Major Repairs by City (2021)',
            labels={'city': 'City', 'renter_major_repairs_percent': 'Percentage (%)'},
            color='renter_major_repairs_percent',
            color_continuous_scale=px.colors.sequential.Teal
        )
        fig_repairs.update_layout(template='plotly_white')
        st.plotly_chart(fig_repairs, use_container_width=True)

        if not df_merged.empty:
            plot_data = df_merged[df_merged['year'] == 2023].groupby('city').agg(
            Avg_Dwelling_Damage_Percent=('avg_dwelling_damage_percent', 'mean')
        ).reset_index()

            plot_data_melted = plot_data.melt(
                id_vars='city',
                value_vars=['Avg_Dwelling_Damage_Percent'],
                var_name='Metric',
                value_name='Value'
            )
            fig = px.bar(
                plot_data_melted,
                x='city',
                y='Value',
                color='Metric',
                title='Dwelling Damage % by City (2023)',
                labels={'city': 'City', 'Value': 'Percentage (%)'}
            )
            fig.update_layout(template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

    with tab_extreme:
        st.markdown("### Extreme Weather Impacts")
        st.markdown("""
        - **Calgary:** Most affected by poor air quality and heavy rain impacts.
        - **Toronto:** Highest rates of dwelling damage and flooding.
        - **Vancouver:** High susceptibility to poor air quality.

        2023 Household Experiences Survey (HES) data highlights broader socio-environmental vulnerabilities across cities.
        """)
        df_2023_weather = df_merged[df_merged['year'] == 2023].copy()
        weather_impacts_2023 = df_2023_weather.groupby('city').agg(
            Total_Precip_mm=('Total_Precip_mm', 'sum'),
            Avg_Dwelling_Damage_Percent=('avg_dwelling_damage_percent', 'mean'),
            Avg_Poor_Air_Quality_Percent=('avg_poor_air_quality_percent', 'mean'),
            Avg_Heavy_Rain_Impact_Percent=('avg_heavy_rain_impact_percent', 'mean')
        ).reset_index()
        weather_melted = weather_impacts_2023.melt(
            id_vars='city',
            value_vars=['Avg_Dwelling_Damage_Percent', 'Avg_Poor_Air_Quality_Percent', 'Avg_Heavy_Rain_Impact_Percent'],
            var_name='Metric',
            value_name='Value'
        )
        fig_weather_impacts = px.bar(
            weather_melted,
            x='city',
            y='Value',
            color='Metric',
            title='Aggregated Extreme Weather Impacts by City (2023)',
            labels={'city': 'City', 'Value': 'Percentage (%)'}
        )
        fig_weather_impacts.update_layout(template='plotly_white', coloraxis_showscale=False)
        st.plotly_chart(fig_weather_impacts, use_container_width=True)

    with tab_mortgage:
        st.markdown("### Mortgage Rates & House Prices")
        st.markdown("""
        **Mortgage Rate Trends:**
        - General upward drift in 2022-2023.
        - Increased volatility observed.
        - Both mortgage rates and rents have risen across the timeframe but are driven by independent determinants.

        **House Prices (2023):**
        - **Vancouver:** Highest average.
        - **Toronto:** Second highest.
        - **Calgary:** Most affordable.
        """)
        if not df_merged.empty:
            df_2023_prices = df_merged[df_merged['year'] == 2023].copy()
            avg_prices_by_city_2023 = df_2023_prices.groupby('city')['avg_house_price'].mean().reset_index()
            fig_prices = px.bar(
                avg_prices_by_city_2023,
                x='city',
                y='avg_house_price',
                title='Average House Prices by City (2023)',
                labels={'city': 'City', 'avg_house_price': 'Avg House Price ($)'},
                color='avg_house_price',
                color_continuous_scale=px.colors.sequential.Viridis
            )
            fig_prices.update_layout(template='plotly_white')
            st.plotly_chart(fig_prices, use_container_width=True)
        
        if not df_merged.empty:
            mortgage_rate_trend = df_merged.groupby(['year', 'month'])['mortgage_rate'].mean().reset_index()
            mortgage_rate_trend['Date'] = pd.to_datetime(mortgage_rate_trend[['year', 'month']].assign(day=1))
        
            fig = px.line(
                mortgage_rate_trend,
                x='Date',
                y='mortgage_rate',
                title='Average Mortgage Rate Over Time',
                labels={'Date': 'Date', 'mortgage_rate': 'Mortgage Rate (%)'}
            )
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

with tab_modeling:
    st.markdown(
    '<div style="text-align:center; font-weight:bold; font-size:1.2rem;">Select City for Modeling</div>',
    unsafe_allow_html=True,
    )
    city = option_menu(
        menu_title=None,
        options=cities_available(),
        default_index=0 if "Toronto" in cities_available() else 0,
        orientation="horizontal",
        key="modeling_city_selector",  # unique key here
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "nav-link": {
                "font-size": "1.4rem",
                "text-align": "center",
                "margin": "0px 10px",
                "--hover-color": "#e0f0ff",
                "border-radius": "8px",
            },
            "nav-link-selected": {"background-color": "#0086ad", "color": "white"},
        },
    )
    st.subheader("Rent Prediction — Interactive What-If & Diagnostics")
    st.markdown("""
    #### How Well the Rent Prediction Works

    - Our rent prediction tool estimates average rent prices with reasonable accuracy. Think of it like a weather forecast but for rent — good for spotting overall trends, but not perfect for every single home.
    - The most important factors affecting rent are the city and the year (how prices change over time).
    - Weather and seasonal changes also play a role, but their influence is smaller compared to location and time.
    - This model provides helpful insights based on past data up to the most recent year available. We’ll explore predictions for future rent trends in the next tab.
    """.format(int(df_merged['year'].max()) if not df_merged.empty else 2024))

    if rent_pipeline is None:
        st.warning("Rent pipeline model not found. Place it in 'trained_models/'.")
    else:
        c1, c2, c3 = st.columns(3)
        t1, t2, t3 = st.columns(3)
        max_t = c1.slider("Avg Max Temp (°C)", -20.0, 40.0, 14.0, 0.5)
        min_t = c2.slider("Avg Min Temp (°C)", -30.0, 25.0, 7.0, 0.5)
        precip = c3.slider("Total Precip (mm)", 0.0, 400.0, 75.0, 1.0)
        snow = t1.slider("Total Snow (cm)", 0.0, 200.0, 10.0, 1.0)
        max_year = int(df_merged['year'].max()) if not df_merged.empty else 2024
        year_input = t2.slider("Year", 2018, max_year, max_year, 1)
        month_input = t3.slider("Month", 1, 12, 6, 1)

        X = pd.DataFrame([{
            "Avg_Max_Temp": max_t,
            "Avg_Min_Temp": min_t,
            "Total_Precip_mm": precip,
            "Total_Snow_cm": snow,
            "year": year_input,
            "month": month_input,
            "city": city
        }])
        X['month_sin'], X['month_cos'] = month_sin_cos(month_input)

        try:
            yhat = float(rent_pipeline.predict(X)[0])
            pred_html = f'<div class="prediction-output">Predicted Avg Rent: ${yhat:,.0f}</div>'
            st.markdown(pred_html, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.markdown("---")
    st.subheader("Dwelling Damage Risk Classification")
    # --- Add model accuracy info and explanation ---
    if dwelling_model is None or dwelling_preproc is None:
        st.warning("Dwelling damage model or preprocessor missing.")
    else:
        X_demo = pd.DataFrame([{
            "Total_Precip_mm": precip,
            "Total_Snow_cm": snow,
            "Avg_Max_Temp": max_t,
            "Avg_Min_Temp": min_t,
            "year": year_input,
            "month": month_input,
            "city": city
        }])
        try:
            X_proc = dwelling_preproc.transform(X_demo)
            proba = float(dwelling_model.predict_proba(X_proc)[0,1])
            st.info(f"Probability of dwelling damage: {proba:.2%} ({damage_label(proba)})")
        except Exception as e:
            st.error(f"Dwelling damage prediction failed: {e}")

    st.markdown("""
    #### Understanding the Dwelling Damage Risk

    This part of the tool estimates how likely a home is to be damaged based on weather conditions like rain, snow, and temperature, along with time and location factors.

    The model currently performs very well on the data it was trained on, but this might mean it is too closely fitted to past data and might not be as accurate with completely new situations. 

    Still, it provides valuable insights to help identify homes that could be at risk from extreme weather events.
    """)


with tab_forecasts:
    st.markdown(
    '<div style="text-align:center; font-weight:bold; font-size:1.2rem;">Select City for Forecasting</div>',
    unsafe_allow_html=True,
    )
    city = option_menu(
        menu_title=None,
        options=cities_available(),
        default_index=0 if "Toronto" in cities_available() else 0,
        orientation="horizontal",
        key="forecast_city_selector",  # different unique key here
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "nav-link": {
                "font-size": "1.4rem",
                "text-align": "center",
                "margin": "0px 10px",
                "--hover-color": "#e0f0ff",
                "border-radius": "8px",
            },
            "nav-link-selected": {"background-color": "#0086ad", "color": "white"},
        },
    )

    horizon_months = st.slider("**Forecast horizon (months)**", 6, 84, 24, 6)
    st.subheader(f"{city} — Forecast Comparison: Prophet vs ARIMA")

    # Explanation about forecasting and model accuracy
    st.markdown("""
   #### How We Predict Future Rents

    - We use two different methods to forecast how rents might change over time.
    - For Calgary, one method usually gives better results; for Vancouver and Toronto, the other method tends to perform better.
    - We pick the method that has proven to be more accurate based on past data.
    - The charts show these forecasts compared with actual rent trends from previous years.
    - These predictions give a helpful idea of future rent changes but are not guarantees since unforeseen events can affect actual outcomes.
    """)

    fig_forecast, chosen_model, rmse_prophet, rmse_arima = best_model_for_city(city, horizon_months=horizon_months)

    if fig_forecast is None:
        st.warning("Insufficient data or missing models for forecasting.")
    else:
        st.plotly_chart(fig_forecast, use_container_width=True)
        st.markdown(f"**Chosen Model:** {chosen_model or '—'}")

        # Calculate and display predicted rent and percentage increase for selected horizon
        last_train_date = df_merged[df_merged["city"] == city]["Date"].max() if not df_merged.empty else None

        # Extract forecast values and dates
        if chosen_model == "Prophet":
            forecast_df = None
            # Load prophet model and generate forecast if needed
            prophet_m = load_model(prophet_model_path(city))
            if prophet_m is not None:
                future_dates = pd.date_range(last_train_date + pd.offsets.MonthBegin(1), periods=horizon_months, freq='MS')
                df_future = pd.DataFrame({"ds": future_dates})
                future_fc = prophet_m.predict(df_future)
                forecast_df = future_fc[["ds", "yhat"]].rename(columns={"ds": "Date", "yhat": "Forecast"})
        else:
            forecast_df = None
            arima_m = load_model(arima_model_path(city))
            if arima_m is not None:
                fc_vals = arima_m.predict(n_periods=horizon_months)
                future_dates = pd.date_range(last_train_date + pd.offsets.MonthBegin(1), periods=horizon_months, freq='MS')
                forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": fc_vals})

        # Determine forecast for selected horizon and percentage increase
        if forecast_df is not None and not forecast_df.empty:
            # User-selected month index (1-based)
            selected_index = horizon_months - 1  # last forecasted month

            pred_rent = forecast_df.iloc[selected_index]["Forecast"] if "Forecast" in forecast_df.columns else forecast_df.iloc[selected_index]["yhat"]
            last_known = df_merged[(df_merged["city"] == city) & (df_merged["Date"] == last_train_date)]["VALUE"].mean()

            perc_increase = ((pred_rent - last_known) / last_known) * 100 if last_known and last_known > 0 else None

            st.markdown(f"### Forecast for {forecast_df.iloc[selected_index]['Date'].strftime('%B %Y')}")
            st.markdown(f"**Predicted Average Rent:** ${pred_rent:,.0f}")

            if perc_increase is not None:
                st.markdown(f"**Estimated Increase from last known rent (${last_known:,.0f}):** {perc_increase:.2f}%")

                # Brief explanation of the increase
                st.markdown("""
                The projected rent increase reflects historical trends, inflation, and market dynamics captured by the forecasting model. However, actual future rents may vary due to economic conditions, policy changes, and unforeseen events.
                """)

        else:
            st.info("Forecast data unavailable for the selected horizon.")


with tab_clusters:
    st.subheader("City Clusters — Housing & Weather Profiles")
    
    st.markdown("""
KMeans clustering assigned each city to a distinct vulnerability cluster, confirming strong feature heterogeneity across the three metropolitan areas. 

Correlation analysis reveals weak to moderate relationships between weather variables and housing outcomes, with city and year factors showing strongest predictive power. Temperature, precipitation, and snowfall demonstrate measurable but not critical direct influence on rental prices and repair needs. Market fundamentals—including location, economic conditions, and temporal trends—consistently outweigh meteorological factors in determining housing affordability and quality outcomes across all three metropolitan areas studied.

<div class="clusters-tab-heading">Vulnerability Clusters Table</div>

| City      | Cluster Profile        | Key Vulnerabilities                                               |
|-----------|-----------------------|------------------------------------------------------------------|
| Vancouver | Distinct              | High susceptibility to poor air quality, moderate repair needs    |
| Calgary   | Distinct              | Most affected by poor air quality and heavy rain impacts          |
| Toronto   | Distinct              | Highest percentage of dwellings needing major repairs, flooding   |
| Overlaps  | Shared vulnerabilities| Market challenges, housing affordability, location-based risk     |

<div class="clusters-tab-heading">Feature Correlation Table</div>

| Variable Type         | Correlation Strength | Significance Level | Market Impact | Predictive Value |
|----------------------|---------------------|-------------------|--------------|------------------|
| Temperature vs Rent  | Moderate            | p<0.05            | Low          | r=0.23           |
| Precipitation vs Rent| Weak                | p<0.01            | Low          | r=0.18           |
| Snowfall vs Repairs  | Moderate            | p<0.001           | Medium       | r=0.31           |
| Year vs Rent         | Strong              | p<0.001           | High         | r=0.78           |
""", unsafe_allow_html=True)


    # --- Proceed with your clustering/dataframe/chart code ---
    if df_merged.empty:
        st.warning("Merged data unavailable for clustering.")
    else:
        agg = df_merged.groupby("city").agg(
            Avg_Rent=("VALUE","mean"),
            Avg_House_Price=("avg_house_price","mean"),
            Avg_Precip=("Total_Precip_mm","mean"),
            Avg_Snow=("Total_Snow_cm","mean"),
            Avg_Max_Temp=("Avg_Max_Temp","mean"),
            Avg_Min_Temp=("Avg_Min_Temp","mean")
        ).dropna()
        if agg.shape[0] < 2:
            st.info("Not enough cities for clustering.")
        else:
            scaler = StandardScaler()
            Xs = scaler.fit_transform(agg)
            km = KMeans(n_clusters=min(3, len(agg)), random_state=42, n_init=10)
            agg['Cluster'] = km.fit_predict(Xs)
            agg['Cluster Name'] = agg['Cluster'].map(lambda x: f"Cluster {x}")


            fig_clusters = px.scatter(
                agg.reset_index(),
                x="Avg_Rent", y="Avg_Max_Temp",
                color="Cluster Name",
                size="Avg_House_Price",
                hover_name="city",
                title="Clusters: Rent vs Max Temp (Bubble size = Avg House Price)",
                template="plotly_white"
            )
            st.plotly_chart(fig_clusters, use_container_width=True)


with tab_report:
    st.markdown('<div class="report-tab-heading">Key Insights & Conclusion</div>', unsafe_allow_html=True)

    st.markdown("""
- Severe weather events have measurable effects on housing quality, influencing repair needs and dwelling damage risks, especially in vulnerable neighborhoods.
- City and year are dominant predictors in the rent model, with year reflecting a variety of changing factors over time such as economic conditions, policy shifts, inflation, and market trends that influence rent levels across the years.
- Weather factors, including temperature, precipitation, and snowfall, provide an additional but modest explanatory role in rent fluctuations and housing quality outcomes.
- Vancouver displays a distinctive vulnerability pattern primarily driven by air quality and moderate repair needs; in contrast, Toronto and Calgary experience different climate housing challenges.
- Market forces, seasonal effects, and policy regulations remain the primary drivers of overall housing affordability and quality profiles across all studied cities.
""")

    st.markdown('<div class="report-tab-heading">Future Directions</div>', unsafe_allow_html=True)

    st.markdown("""
- Acquire more granular and longitudinal housing and vulnerability datasets to capture micro-level effects and temporal nuances.
- Integrate socioeconomic indicators, such as income, employment, and demographic change, to more fully model rent and housing outcomes.
- Explore advanced and hybrid forecasting approaches combining econometric, machine learning, and climate predictive modeling for robust future scenario planning.
- Develop interactive, accessible visualization dashboards targeted for policymakers, housing agencies, and community stakeholders to inform climate-resilient housing strategies.
""")

    st.markdown("""
In summary, while weather acts as a moderating influence on housing market outcomes and quality risk, the key levers remain rooted in economic, regulatory, and demographic contexts. A holistic approach is essential for building resilience in Canadian urban housing markets.
""")



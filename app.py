import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

conn = sqlite3.connect('ipl_data.db')
cursor = conn.cursor()

st.title("🏏 IPL Player Matchup Engine")

# 1. Fetch unique player lists from database for dropdowns
def get_player_lists():
    df_batter = pd.read_sql_query("SELECT batter FROM ipldata WHERE (valid_ball = 1) GROUP BY batter HAVING SUM(runs_total) > 3000 ORDER BY SUM(runs_total) DESC;", conn)
    batters = df_batter['batter'].tolist()
    df_bowler = pd.read_sql_query("SELECT bowler FROM ipldata WHERE (valid_ball = 1) GROUP BY bowler HAVING SUM(valid_ball) > 3000 ORDER BY SUM(valid_ball) DESC;", conn)
    bowlers = df_bowler['bowler'].tolist()
    return batters, bowlers

batters_list, bowlers_list = get_player_lists()

# 2. Input Widgets (Dropdowns side by side)
col_input1, col_input2 = st.columns(2)
with col_input1:
    selected_batter = st.selectbox("Select Batsman", batters_list, index=batters_list.index("V Kohli") if "V Kohli" in batters_list else 0)
with col_input2:
    selected_bowler = st.selectbox("Select Bowler", bowlers_list, index=bowlers_list.index("R Ashwin") if "R Ashwin" in bowlers_list else 0)

# 3. Query Matchup Data directly from SQLite
def get_matchup_stats(batter, bowler):
    # We aggregate the stats directly in SQL to save memory and increase speed
    query = """
    SELECT 
        SUM(runs_total) AS runs,
        SUM(valid_ball) AS balls,
        SUM(bowler_wicket) AS dismissals
    FROM ipldata
    WHERE batter = ? AND bowler = ? AND valid_ball = 1
    """
    
    # Execute the query, safely passing the selected dropdown values
    df_h2h = pd.read_sql_query(query, conn, params=(batter, bowler))
    return df_h2h

# Fetch the aggregated single-row dataframe
df_h2h = get_matchup_stats(selected_batter, selected_bowler)

# 4. Extract Data and Store in Python Variables
# We use .fillna(0) and .iloc[0] to safely extract the numbers, 
# ensuring they default to 0 if the two players have never faced each other.
runs = int(df_h2h['runs'].fillna(0).iloc[0])
balls = int(df_h2h['balls'].fillna(0).iloc[0])
dismissals = int(df_h2h['dismissals'].fillna(0).iloc[0])

# Calculate strike rate safely to avoid a DivisionByZero error
strike_rate = round((runs / balls) * 100, 1) if balls > 0 else 0.0

# 5. Display KPI Cards using Streamlit Columns
st.subheader(f"Head-to-Head: {selected_batter} vs {selected_bowler}")

metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    st.metric(label="Runs Scored", value=runs)
    
with metric_col2:
    st.metric(label="Strike Rate", value=f"{strike_rate}%", delta=f"{balls} balls faced" if balls > 0 else "No history")
    
with metric_col3:
    st.metric(label="Times Out", value=dismissals)

# --- New Section: Yearly Runs Bar Chart ---
st.markdown("---") # Adds a horizontal line to separate the app sections
st.subheader("📊 Yearly Runs Progression")

# 1. Create the new dropdown (reusing the batters_list we already generated)
selected_solo_batter = st.selectbox("Select Batter for Season Stats", batters_list,index=batters_list.index("V Kohli") if "V Kohli" in batters_list else 0)

def get_runs_by_year(batter):
    query = """
    SELECT year, SUM(runs_total) AS total_runs
    FROM ipldata
    WHERE batter = ? AND valid_ball = 1
    GROUP BY year
    ORDER BY year
    """
    df_yearly = pd.read_sql_query(query, conn, params=(batter,))
    return df_yearly

df_yearly_runs = get_runs_by_year(selected_solo_batter)
st.subheader(f"Yearly Runs for {selected_solo_batter}")

st.bar_chart(df_yearly_runs.set_index('year')['total_runs'])

# --- New Section: Yearly Wickets Bar Chart ---
st.markdown("---") # Adds a horizontal line to separate the app sections
st.subheader("📊 Yearly Wickets Progression")

selected_solo_bowler = st.selectbox("Select Bowler for Season Stats", bowlers_list, index=bowlers_list.index("R Ashwin") if "R Ashwin" in bowlers_list else 0)

def get_wickets_by_year(bowler):
    query = """
    SELECT year, SUM(bowler_wicket) AS total_wickets
    FROM ipldata
    WHERE bowler = ? AND valid_ball = 1
    GROUP BY year
    ORDER BY year
    """
    df_yearly_wickets = pd.read_sql_query(query, conn, params=(bowler,))
    return df_yearly_wickets

df_yearly_wickets = get_wickets_by_year(selected_solo_bowler)
st.subheader(f"Yearly Wickets for {selected_solo_bowler}")      
st.bar_chart(df_yearly_wickets.set_index('year')['total_wickets'])

# --- New Section: Yearly Runs Bar Chart ---
st.markdown("---") # Adds a horizontal line to separate the app sections
st.subheader("📊 Most Destructive in the Powerplay")

query = """
SELECT batter, year, SUM(runs_total) AS total_runs, SUM(valid_ball) AS total_balls, ROUND(CAST(SUM(runs_total) AS FLOAT) / SUM(valid_ball) * 100, 2) AS strike_rate
FROM ipldata
WHERE valid_ball = 1 AND ball_no < 6
GROUP BY batter, year
HAVING SUM(runs_total) >= 200;
"""
df_powerplay_runs = pd.read_sql_query(query, conn)

def plot_pp_stats_by_year(year):


    # 2. Render the interactive bubble chart
    if not df_powerplay_runs.empty:
        fig = px.scatter(
            df_powerplay_runs[df_powerplay_runs['year'] == year],  # Filter data for the selected year
            x="total_runs",            # X-axis
            y="strike_rate",             # Y-axis
            size="strike_rate",         # Bubble area represents Strike Rate
            # hover_name="batter",        # Shows player name on mouse hover
            text="batter",              # Display player name as text on the bubble
            size_max=10,                # Scales the bubbles so differences are obvious
            labels={
                "total_balls": "Total Balls Faced",
                "total_runs": "Total Runs Scored",
                "strike_rate": "Strike Rate"
            }
        )
        fig.update_traces(
        mode='markers+text',
        textposition='top center',
        hoverinfo='skip' # Disables the hover tooltip completely
)
        
        # Plotly charts in Streamlit are fully interactive (zoom, pan, hover)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No players found with 200+ runs in this dataset.")

# 1. Year selection dropdown

years = df_powerplay_runs['year'].unique()
selected_year = st.selectbox("Select Year for Powerplay Analysis", years)   
plot_pp_stats_by_year(selected_year)


# --- New Section: Runs in the 1st innings by team and venue ---
st.markdown("---") # Adds a horizontal line to separate the app sections
st.subheader("📊 Runs in the 1st innings by team and venue")

query = """
SELECT venue, batting_team, match_won_by, match_id, year, SUM(runs_total) AS total_runs
FROM ipldata
WHERE innings = 1
GROUP BY venue, batting_team, innings, match_won_by, match_id, year
ORDER BY total_runs DESC;
"""

df_first_innings = pd.read_sql_query(query, conn)

def plot_first_innings_runs(venue, year):
    df_first_innings_filtered = df_first_innings[(df_first_innings['venue'] == venue) & (df_first_innings['year'] == year)]
    df_first_innings_filtered['result']  = np.where(df_first_innings_filtered['match_won_by'] == df_first_innings_filtered['batting_team'], 'won', 'lost')
    if not df_first_innings_filtered.empty:
        fig = px.scatter(
            df_first_innings_filtered,
            x="batting_team",
            y="total_runs",
            color="result",
            color_discrete_map={
            "lost": "red",   # Keys must be strings matching the converted data
            "won": "blue"
            },
            hover_data=["match_id"],
            labels={
                "batting_team": "Batting Team",
                "total_runs": "Total Runs in 1st Innings",
                "year": "Year"
            }
        )
        # Clean up the layout
        fig.update_layout(
        xaxis_title="", 
        showlegend=True,       # Hides the legend since the x-axis already lists the teams
        margin=dict(t=20, b=0)
    )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data found for this venue.")


selected_year, selected_venue = st.columns(2)

selected_year = st.selectbox("Select Year for 1st Innings Analysis", df_first_innings['year'].unique())

venues = df_first_innings[df_first_innings['year'] == selected_year]['venue'].unique()

selected_venue = st.selectbox("Select Venue for 1st Innings Analysis", venues)
plot_first_innings_runs(selected_venue, selected_year)

st.markdown("---")
st.subheader("🔮 Live 1st Innings Projected Score Calculator")

# Load model and distinct categorization filters safely
@st.cache_resource
def load_prediction_assets():
    model = joblib.load('score_predictor_model.pkl')
    return model

model = load_prediction_assets()

# Fetch categorical option lists directly from your existing cache functions
# Assuming batters_list, bowlers_list, years_list, and venues_list are available globally
# We can pull teams lists from your database

teams = pd.read_sql_query("SELECT DISTINCT batting_team FROM ipldata ORDER BY batting_team;", conn)['batting_team'].tolist()
venues = pd.read_sql_query("SELECT DISTINCT venue FROM ipldata ORDER BY venue;", conn)['venue'].tolist()

# --- UI Input Layout ---
ui_col1, ui_col2, ui_col3 = st.columns(3)

with ui_col1:
    batting_team = st.selectbox("Batting Team", teams, key="pred_bat")
    current_runs = st.number_input("Current Runs Scored", min_value=0, max_value=300, value=50)

with ui_col2:
    bowling_team = st.selectbox("Bowling Team", teams, key="pred_bowl")
    wickets_down = st.slider("Wickets Down", min_value=0, max_value=9, value=2)

with ui_col3:
    selected_venue = st.selectbox("Match Venue", venues, key="pred_ven")
    runs_last_3 = st.number_input("Runs in Last 3 Overs (Momentum)", min_value=0, max_value=100, value=24)

# Over and Ball Slider configs
slider_col1, slider_col2 = st.columns(2)
with slider_col1:
    current_over = st.slider("Current Over Number", min_value=2, max_value=19, value=6)
with slider_col2:
    current_ball = st.slider("Ball of the Over", min_value=1, max_value=6, value=3)


# --- Live Inference Execution ---
if st.button("Generate Live Projection"):
    if batting_team == bowling_team:
        st.error("Error: Batting team and Bowling team cannot be identical.")
    else:
        # Construct exact same structural data layout schema expected by the Pipeline steps
        input_data = pd.DataFrame([{
            'venue': selected_venue,
            'batting_team': batting_team,
            'bowling_team': bowling_team,
            'current_over': current_over,
            'current_ball': current_ball,
            'current_runs': current_runs,
            'wickets_down': wickets_down,
            'runs_last_3_overs': runs_last_3
        }])
        
        # Calculate inference
        predicted_score = model.predict(input_data)[0]
        lower_bound = int(predicted_score - 12)
        upper_bound = int(predicted_score + 12)
        
        # Visual presentation of results
        st.markdown("### **Prediction Results**")
        p_col1, p_col2 = st.columns(2)
        
        with p_col1:
            st.metric(
                label="Predicted Final Score", 
                value=f"{int(predicted_score)} Runs",
                delta=f"{int(predicted_score - current_runs)} additional runs expected"
            )
        with p_col2:
            st.metric(
                label="Estimated Safe Target Window (±12 Runs)", 
                value=f"{lower_bound} — {upper_bound}"
            )

st.markdown("---")
st.subheader("🎲 Stochastic Monte Carlo Innings Simulator")
st.markdown("Simulate the remainder of an innings 10,000 times to see the full distribution of possible final scores.")

#--- Step 1: Fetch Empirical Probabilities from Database ---
@st.cache_data
def get_phase_probabilities():
    # We break cricket into 3 distinct phases because scoring rates and wicket risks change drastically
    query = """
    SELECT 
        CASE 
            WHEN ball_no < 6 THEN 'Powerplay'
            WHEN ball_no BETWEEN 6 AND 14 THEN 'Middle'
            ELSE 'Death'
        END AS phase,
        runs_total,
        -- Check your database schema to confirm if column is named 'is_wicket' or 'bowler_wicket'
        CASE WHEN bowler_wicket = 1 THEN 1 ELSE 0 END as wkt,
        COUNT(*) as ball_count
    FROM ipldata
    WHERE innings IN (1, 2) AND valid_ball = 1
    GROUP BY phase, runs_total, wkt;
    """
    df_events = pd.read_sql_query(query, conn)
    df_events = df_events[(df_events['runs_total'] > 0) & (df_events['wkt'] != 1)] # Filter out extras and non-ball events
    
    # Process the database records into unique outcomes and mapping weights
    phases = ['Powerplay', 'Middle', 'Death']
    probabilities = {}
    
    for phase in phases:
        df_phase = df_events[df_events['phase'] == phase]
        total_balls = df_phase['ball_count'].sum()
        
        # Create unique state tags (e.g., "4_runs", "0_runs", "wicket")
        outcomes = []
        weights = []
        
        for _, row in df_phase.iterrows():
            if row['wkt'] == 1:
                outcomes.append('WICKET')
            else:
                outcomes.append(int(row['runs_total']))
            weights.append(row['ball_count'] / total_balls)
            
        probabilities[phase] = {"outcomes": outcomes, "weights": weights}

    return probabilities

# Load the historical probability matrices
prob_matrix = get_phase_probabilities()

# --- Step 2: Streamlit User Interface Setup ---
col_mc1, col_mc2, col_mc3 = st.columns(3)

with col_mc1:
    mc_runs = st.number_input("Current Runs", min_value=0, max_value=250, value=75, key="mc_r")
    mc_overs = st.slider("Overs Completed", min_value=0, max_value=19, value=10, key="mc_o")

with col_mc2:
    mc_wickets = st.slider("Wickets Lost", min_value=0, max_value=9, value=3, key="mc_w")
    simulations = st.select_slider("Number of Simulations", options=[1000, 5000, 10000], value=10000)

with col_mc3:
    st.markdown("<br>", unsafe_allow_html=True) # Spacer
    run_sim = st.button("🚀 Run Monte Carlo Simulation", use_container_width=True)

# --- Step 3: Vectorized Simulation Engine Loop ---
if run_sim:
    final_scores = []
    balls_remaining = 120 - (mc_overs * 6)
    
    # Run the stochastic process loop
    for _ in range(simulations):
        sim_runs = mc_runs
        sim_wkts = mc_wickets
        
        for ball in range(balls_remaining):
            # Dynamic game phase tracking based on active ball count
            current_ball_idx = 120 - balls_remaining + ball
            if current_ball_idx < 36:
                phase = 'Powerplay'
            elif current_ball_idx < 90:
                phase = 'Middle'
            else:
                phase = 'Death'
            
            # Randomly sample an event based on historical weights
            event = np.random.choice(
                prob_matrix[phase]["outcomes"], 
                p=prob_matrix[phase]["weights"]
            )
            
            if event == 'WICKET':
                sim_wkts += 1
                if sim_wkts == 10:
                    break # Innings terminated: all out
            else:
                sim_runs += event
                
        final_scores.append(sim_runs)
        
    # --- Step 4: Statistical Processing & Visualizations ---
    df_sim = pd.DataFrame({'Final_Score': final_scores})
    
    # Calculate statistical percentiles
    percentile_5 = int(np.percentile(final_scores, 5))
    median_score = int(np.percentile(final_scores, 50))
    percentile_95 = int(np.percentile(final_scores, 95))
    
    # Display Summary KPI Blocks
    m1, m2, m3 = st.columns(3)
    m1.metric("Floor Score (5% Chance Below)", percentile_5)
    m2.metric("Median Projected Score", median_score)
    m3.metric("Ceiling Score (5% Chance Above)", percentile_95)
    
    # Generate Interactive Plotly Histogram
    fig_hist = px.histogram(
        df_sim, 
        x="Final_Score",
        nbins=40,
        title="Distribution of Projected Final Scores across 10,000 Parallel Universes",
        labels={"Final_Score": "Final Innings Score"},
        color_discrete_sequence=['#4A90E2']
    )
    
    # Add vertical indicators for the percentiles
    fig_hist.add_vline(x=median_score, line_dash="dash", line_color="white", annotation_text=f"Median: {median_score}")
    fig_hist.add_vline(x=percentile_5, line_dash="dot", line_color="red", annotation_text="5th Percentile")
    fig_hist.add_vline(x=percentile_95, line_dash="dot", line_color="green", annotation_text="95th Percentile")
    
    st.plotly_chart(fig_hist, use_container_width=True)
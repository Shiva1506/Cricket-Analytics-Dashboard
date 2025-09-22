import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import joblib
from sklearn.preprocessing import LabelEncoder

# Set page configuration
st.set_page_config(
    page_title="Advanced Cricket Statistics Dashboard",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .bowling-metric {
        border-left: 4px solid #ff6b6b;
    }
    .section-header {
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    .bowling-header {
        color: #ff6b6b;
        border-bottom: 2px solid #ff6b6b;
    }
    .format-badge {
        background-color: #1f77b4;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        margin-left: 1rem;
    }
    .stats-type-badge {
        background-color: #ff6b6b;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
    }
    .player-vs-player {
        border-left: 4px solid #2ecc71;
    }
    .team-analysis {
        border-left: 4px solid #9b59b6;
    }
    .prediction-card {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2ecc71;
        margin-bottom: 1rem;
    }
    .trend-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper function for safe max calculation
def safe_max_calculation(df, metrics):
    """Safely calculate max values for metrics, handling missing columns"""
    max_vals = {}
    for metric in metrics:
        if metric in df.columns:
            max_val = df[metric].max()
            # Handle cases where max might be 0 or NaN
            if pd.isna(max_val) or max_val == 0:
                max_vals[metric] = 1
            else:
                max_vals[metric] = max_val
        else:
            max_vals[metric] = 1  # Default value
    return max_vals

# ML Model Integration Classes
class ModelPredictor:
    def __init__(self):
        self.batting_models = {}
        self.bowling_models = {}
        self.xi_models = {}
        self.label_encoders = {}
        
    def load_models(self, format_name):
        """Load models for a specific format"""
        try:
            # Load performance models
            perf_models = joblib.load(f'player_performance_models_{format_name.lower()}.pkl')
            self.batting_models[format_name] = perf_models['batting_model']
            self.bowling_models[format_name] = perf_models['bowling_model']
            self.label_encoders[format_name] = perf_models['label_encoders']
            
            # Load playing XI model
            xi_model_data = joblib.load(f'playing_xi_model_{format_name.lower()}.pkl')
            self.xi_models[format_name] = xi_model_data['selector_model']
            
            return True
        except Exception as e:
            st.sidebar.error(f"Error loading models for {format_name}: {str(e)}")
            return False
    
    def predict_player_performance(self, player_name, role, opponent, venue, is_home, format_name):
        """Predict player performance using trained model"""
        if format_name not in self.batting_models or format_name not in self.bowling_models:
            return None
            
        try:
            # Get player's recent performance data
            player_data = ball_data[format_name][
                ((ball_data[format_name]['batsman'] == player_name) | 
                 (ball_data[format_name]['bowler'] == player_name))
            ]
            
            if player_data.empty:
                return None
                
            # Calculate recent form metrics
            if role == 'batsman':
                batting_stats = player_data[player_data['batsman'] == player_name].groupby('match_id').agg({
                    'runs_batsman': 'sum',
                    'ball_in_over': 'count'
                }).tail(10)
                
                if batting_stats.empty:
                    return None
                    
                prev_avg_runs = batting_stats['runs_batsman'].mean()
                prev_avg_sr = (batting_stats['runs_batsman'].sum() / batting_stats['ball_in_over'].sum() * 100) if batting_stats['ball_in_over'].sum() > 0 else 0
                form = prev_avg_runs * 0.7 + prev_avg_sr * 0.3
                
                # Prepare features
                le_dict = self.label_encoders[format_name]
                opponent_encoded = le_dict[f'batting_opponent'].transform([opponent])[0] if opponent in le_dict[f'batting_opponent'].classes_ else 0
                venue_encoded = le_dict[f'batting_venue'].transform([venue])[0] if venue in le_dict[f'batting_venue'].classes_ else 0
                
                features = [[opponent_encoded, venue_encoded, is_home, prev_avg_runs, prev_avg_sr, form]]
                
                # Scale features and predict
                scaled_features = StandardScaler().fit_transform(features)
                predicted_runs = self.batting_models[format_name].predict(scaled_features)[0]
                
                # For strike rate, we can use a simpler approach or train a separate model
                predicted_sr = max(50, min(200, prev_avg_sr * 0.7 + 100 * 0.3))  # Weighted average
                
                return {
                    'predicted_runs': max(0, round(predicted_runs)),
                    'predicted_strike_rate': round(predicted_sr, 1),
                    'confidence': 0.7  # Could be based on model confidence
                }
                
            else:  # bowler
                bowling_stats = player_data[player_data['bowler'] == player_name].groupby('match_id').agg({
                    'wicket_kind': lambda x: x.notna().sum(),
                    'runs_total': 'sum',
                    'ball_in_over': 'count'
                }).tail(10)
                
                if bowling_stats.empty:
                    return None
                    
                prev_avg_wickets = bowling_stats['wicket_kind'].mean()
                prev_avg_economy = (bowling_stats['runs_total'].sum() / bowling_stats['ball_in_over'].sum() * 6) if bowling_stats['ball_in_over'].sum() > 0 else 0
                form = prev_avg_wickets * 0.7 - prev_avg_economy * 0.3
                
                # Prepare features
                le_dict = self.label_encoders[format_name]
                opponent_encoded = le_dict[f'bowling_opponent'].transform([opponent])[0] if opponent in le_dict[f'bowling_opponent'].classes_ else 0
                venue_encoded = le_dict[f'bowling_venue'].transform([venue])[0] if venue in le_dict[f'bowling_venue'].classes_ else 0
                
                features = [[opponent_encoded, venue_encoded, is_home, prev_avg_wickets, prev_avg_economy, form]]
                
                # Scale features and predict
                scaled_features = StandardScaler().fit_transform(features)
                predicted_wickets = self.bowling_models[format_name].predict(scaled_features)[0]
                
                # For economy, use weighted average
                predicted_economy = max(3, min(12, prev_avg_economy * 0.7 + 7 * 0.3))
                
                return {
                    'predicted_wickets': max(0, round(predicted_wickets, 1)),
                    'predicted_economy': round(predicted_economy, 1),
                    'confidence': 0.7
                }
                
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            return None
    
    def predict_playing_xi(self, team_name, format_name):
        """Predict optimal playing XI using trained model"""
        if format_name not in self.xi_models:
            return None
            
        try:
            # Get team players from ball-by-ball data
            team_players = []
            team_matches = ball_data[format_name][
                (ball_data[format_name]['team1'] == team_name) | 
                (ball_data[format_name]['team2'] == team_name)
            ]
            
            for match_id in team_matches['match_id'].unique():
                match_data = team_matches[team_matches['match_id'] == match_id]
                
                if match_data['team1'].iloc[0] == team_name:
                    team_batsmen = match_data[match_data['batting_team'] == team_name]['batsman'].unique()
                    team_bowlers = match_data[match_data['team2'] == team_name]['bowler'].unique()
                else:
                    team_batsmen = match_data[match_data['batting_team'] == team_name]['batsman'].unique()
                    team_bowlers = match_data[match_data['team1'] == team_name]['bowler'].unique()
                
                team_players.extend(team_batsmen)
                team_players.extend(team_bowlers)
            
            team_players = list(set([p for p in team_players if pd.notna(p)]))
            
            if not team_players:
                return None
                
            # Get player stats and predict selection probability
            player_predictions = []
            
            for player in team_players:
                batting_data = ball_data[format_name][ball_data[format_name]['batsman'] == player]
                bowling_data = ball_data[format_name][ball_data[format_name]['bowler'] == player]
                
                if batting_data.empty and bowling_data.empty:
                    continue
                
                # Determine role
                is_batsman = not batting_data.empty
                is_bowler = not bowling_data.empty
                
                if is_batsman and is_bowler:
                    role = 'all_rounder'
                elif is_batsman:
                    role = 'batsman'
                else:
                    role = 'bowler'
                
                # Calculate stats
                if is_batsman:
                    batting_stats = batting_data.groupby('match_id').agg({
                        'runs_batsman': 'sum',
                        'ball_in_over': 'count'
                    })
                    avg_runs = batting_stats['runs_batsman'].mean() if not batting_stats.empty else 0
                    avg_sr = (batting_stats['runs_batsman'].sum() / batting_stats['ball_in_over'].sum() * 100) if batting_stats['ball_in_over'].sum() > 0 else 0
                    total_runs = batting_stats['runs_batsman'].sum()
                    matches_batted = len(batting_stats)
                else:
                    avg_runs = 0
                    avg_sr = 0
                    total_runs = 0
                    matches_batted = 0
                
                if is_bowler:
                    bowling_stats = bowling_data.groupby('match_id').agg({
                        'wicket_kind': lambda x: x.notna().sum(),
                        'runs_total': 'sum',
                        'ball_in_over': 'count'
                    })
                    avg_wickets = bowling_stats['wicket_kind'].mean() if not bowling_stats.empty else 0
                    avg_economy = (bowling_stats['runs_total'].sum() / bowling_stats['ball_in_over'].sum() * 6) if bowling_stats['ball_in_over'].sum() > 0 else 0
                    total_wickets = bowling_stats['wicket_kind'].sum()
                    matches_bowled = len(bowling_stats)
                else:
                    avg_wickets = 0
                    avg_economy = 0
                    total_wickets = 0
                    matches_bowled = 0
                
                # Encode role
                role_encoded = 0  # Default
                if 'role' in self.label_encoders:
                    try:
                        role_encoded = self.label_encoders['role'].transform([role])[0]
                    except:
                        # Fallback encoding
                        role_encoding = {'batsman': 0, 'bowler': 1, 'all_rounder': 2}
                        role_encoded = role_encoding.get(role, 0)
                
                # Prepare features
                features = [[role_encoded, avg_runs, avg_sr, avg_wickets, avg_economy, 
                           total_runs, total_wickets, matches_batted, matches_bowled]]
                
                # Predict selection probability
                probability = self.xi_models[format_name].predict_proba(features)[0][1]
                
                player_predictions.append({
                    'player': player,
                    'role': role,
                    'selection_probability': probability,
                    'avg_runs': avg_runs,
                    'avg_sr': avg_sr,
                    'avg_wickets': avg_wickets,
                    'avg_economy': avg_economy
                })
            
            # Select top 11 players
            predictions_df = pd.DataFrame(player_predictions)
            if len(predictions_df) >= 11:
                optimal_xi = predictions_df.nlargest(11, 'selection_probability')
            else:
                optimal_xi = predictions_df
            
            return optimal_xi.sort_values('selection_probability', ascending=False)
            
        except Exception as e:
            st.error(f"Error in playing XI prediction: {str(e)}")
            return None

# Initialize the model predictor
model_predictor = ModelPredictor()

# Load data for all formats
@st.cache_data
def load_all_data():
    batting_files = {
        'ODI': 'ODI.csv',
        'T20': 't20.csv',
        'Test': 'test.csv'
    }
    
    bowling_files = {
        'ODI': 'Bowling_ODI.csv',
        'T20': 'Bowling_t20.csv',
        'Test': 'Bowling_test.csv'
    }
    
    fielding_files = {
        'ODI': 'Fielding_ODI.csv',
        'T20': 'Fielding_t20.csv',
        'Test': 'Fielding_test.csv'
    }
    
    # New data files for additional features
    batsman_vs_bowler_files = {
        'ODI': 'batsman_vs_bowler_odi.csv',
        'T20': 'batsman_vs_bowler_t20.csv',
        'Test': 'batsman_vs_bowler_test.csv'
    }
    
    playing_xi_files = {
        'ODI': 'playing_xi_selection_odi.csv',
        'T20': 'playing_xi_selection_t20.csv',
        'Test': 'playing_xi_selection_test.csv'
    }
    
    match_files = {
        'Test': 'testcricsheet_match_summary.csv',
        'ODI': 'odicricsheet_match_summary.csv',
        'T20': 't20cricsheet_match_summary.csv'
    }
    
    batting_data = {}
    bowling_data = {}
    fielding_data = {}
    batsman_vs_bowler_data = {}
    playing_xi_data = {}
    match_data = {}
    
    # Load ball-by-ball data
    ball_by_ball_files = {
        'T20': 't20cricsheet_ball_by_ball.csv',
        'ODI': 'odicricsheet_ball_by_ball.csv',
        'Test': 'testcricsheet_ball_by_ball.csv'
    }
    
    ball_data = {}
    
    # Load batting data
    for format_name, filename in batting_files.items():
        try:
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                df['format'] = format_name
                
                # Rename columns to standardize across formats
                column_mapping = {
                    'Player': 'batsman',
                    'Mat': 'matches',
                    'Inns': 'innings',
                    'NO': 'not_outs',
                    'Runs': 'runs',
                    'HS': 'highest_score',
                    'Ave': 'batting_average',
                    'BF': 'balls_faced',
                    'SR': 'strike_rate',
                    '100': 'centuries',
                    '50': 'fifties',
                    '0': 'ducks'
                }
                
                df.rename(columns=column_mapping, inplace=True)
                
                # Convert numeric columns to appropriate data types
                numeric_columns = ['matches', 'innings', 'not_outs', 'runs', 'balls_faced', 
                                'strike_rate', 'centuries', 'fifties', 'ducks']
                
                for col in numeric_columns:
                    if col in df.columns:
                        # Remove any non-numeric characters and convert to float
                        df[col] = pd.to_numeric(df[col].astype(str).str.replace('*', '', regex=False), errors='coerce')
                
                # Handle batting average separately as it might have special characters
                if 'batting_average' in df.columns:
                    df['batting_average'] = pd.to_numeric(df['batting_average'].astype(str).str.replace('*', '', regex=False), errors='coerce')
                
                batting_data[format_name] = df
                st.sidebar.success(f"✅ Loaded {format_name} batting data: {len(df)} players")
            else:
                st.sidebar.warning(f"⚠️ {filename} not found")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading {format_name} batting: {str(e)}")

    # Load bowling data
    for format_name, filename in bowling_files.items():
        try:
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                df['format'] = format_name
                
                # Rename columns to standardize across formats
                column_mapping = {
                    'Player': 'bowler',
                    'Mat': 'matches',
                    'Inns': 'innings',
                    'Balls': 'balls_bowled',
                    'Runs': 'runs_conceded',
                    'Wkts': 'wickets',
                    'BBI': 'best_bowling',
                    'Ave': 'bowling_average',
                    'Econ': 'economy',
                    'SR': 'bowling_strike_rate',
                    '4': 'four_wickets',
                    '5': 'five_wickets'
                }
                
                df.rename(columns=column_mapping, inplace=True)
                
                # Convert numeric columns to appropriate data types
                numeric_columns = ['matches', 'innings', 'balls_bowled', 'runs_conceded', 'wickets',
                                'economy', 'bowling_strike_rate', 'four_wickets', 'five_wickets']
                
                for col in numeric_columns:
                    if col in df.columns:
                        # Remove any non-numeric characters and convert to float
                        df[col] = pd.to_numeric(df[col].astype(str).str.replace('*', '', regex=False), errors='coerce')
                
                # Handle bowling average separately as it might have special characters
                if 'bowling_average' in df.columns:
                    df['bowling_average'] = pd.to_numeric(df['bowling_average'].astype(str).str.replace('*', '', regex=False), errors='coerce')
                
                bowling_data[format_name] = df
                st.sidebar.success(f"✅ Loaded {format_name} bowling data: {len(df)} players")
            else:
                st.sidebar.warning(f"⚠️ {filename} not found")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading {format_name} bowling: {str(e)}")

    # Load fielding data
    for format_name, filename in fielding_files.items():
        try:
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                df['format'] = format_name
                
                # Rename columns to standardize across formats
                column_mapping = {
                    'Player': 'player',
                    'Mat': 'matches',
                    'Inns': 'innings',
                    'Dis': 'dismissals',
                    'Ct': 'catches',
                    'St': 'stumpings',
                    'Ct Wk': 'catches_as_wk',
                    'Ct Fi': 'catches_as_fielder',
                    'MD': 'most_dismissals_inning',
                    'D/I': 'dismissals_per_inning'
                }
                
                df.rename(columns=column_mapping, inplace=True)
                
                # Convert numeric columns to appropriate data types
                numeric_columns = ['matches', 'innings', 'dismissals', 'catches', 'stumpings',
                                'catches_as_wk', 'catches_as_fielder', 'most_dismissals_inning', 'dismissals_per_inning']
                
                for col in numeric_columns:
                    if col in df.columns:
                        # Remove any non-numeric characters and convert to float
                        df[col] = pd.to_numeric(df[col].astype(str).str.replace('*', '', regex=False), errors='coerce')
                
                fielding_data[format_name] = df
                st.sidebar.success(f"✅ Loaded {format_name} fielding data: {len(df)} players")
            else:
                st.sidebar.warning(f"⚠️ {filename} not found")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading {format_name} fielding: {str(e)}")
    
    # Load batsman vs bowler data
    for format_name, filename in batsman_vs_bowler_files.items():
        try:
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                df['format'] = format_name
                batsman_vs_bowler_data[format_name] = df
                st.sidebar.success(f"✅ Loaded {format_name} batsman vs bowler data: {len(df)} matchups")
            else:
                st.sidebar.warning(f"⚠️ {filename} not found")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading {format_name} batsman vs bowler: {str(e)}")
    
    # Load playing XI selection data
    for format_name, filename in playing_xi_files.items():
        try:
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                df['format'] = format_name
                # Convert date column if exists
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                playing_xi_data[format_name] = df
                st.sidebar.success(f"✅ Loaded {format_name} playing XI data: {len(df)} records")
            else:
                st.sidebar.warning(f"⚠️ {filename} not found")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading {format_name} playing XI data: {str(e)}")
    
    # Load match data
    for format_name, filename in match_files.items():
        try:
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                df['format'] = format_name
                # Convert date column to datetime
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                match_data[format_name] = df
                st.sidebar.success(f"✅ Loaded {format_name} match data: {len(df)} matches")
            else:
                st.sidebar.warning(f"⚠️ {filename} not found")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading {format_name} match data: {str(e)}")
    
    # Load ball-by-ball data
    for format_name, filename in ball_by_ball_files.items():
        try:
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                # Convert date column if exists
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                ball_data[format_name] = df
                st.sidebar.success(f"✅ Loaded {format_name} ball-by-ball data: {len(df)} deliveries")
            else:
                st.sidebar.warning(f"⚠️ {filename} not found")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading {format_name} ball-by-ball data: {str(e)}")
    
    return batting_data, bowling_data, fielding_data, batsman_vs_bowler_data, playing_xi_data, match_data, ball_data

# Load all data
batting_data, bowling_data, fielding_data, batsman_vs_bowler_data, playing_xi_data, match_data, ball_data = load_all_data()

if not batting_data and not bowling_data and not match_data:
    st.error("No data files found. Please make sure at least one CSV file exists.")
    st.stop()

# Load ML models for each format
available_formats = list(batting_data.keys()) if batting_data else list(bowling_data.keys()) if bowling_data else []
for format_name in available_formats:
    try:
        if model_predictor.load_models(format_name):
            st.sidebar.success(f"✅ Loaded ML models for {format_name}")
        else:
            st.sidebar.warning(f"⚠️ Could not load ML models for {format_name}")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading ML models for {format_name}: {str(e)}")

# Ball-by-ball analysis functions
def over_by_over_analysis(match_id, format_name):
    """Analyze runs scored per over in a match"""
    if format_name not in ball_data:
        return None
        
    match_data = ball_data[format_name][ball_data[format_name]['match_id'] == match_id]
    
    if match_data.empty:
        return None
        
    # Calculate runs per over
    over_stats = match_data.groupby(['inning', 'over']).agg({
        'runs_total': 'sum',
        'wicket_kind': 'count'
    }).reset_index()
    
    return over_stats

def partnership_analysis(match_id, inning, format_name=None):
    """Analyze partnerships in a match"""
    if format_name is None:
        format_name = selected_format
    if format_name not in ball_data:
        return pd.DataFrame()
        
    match_data = ball_data[format_name][
        (ball_data[format_name]['match_id'] == match_id) & 
        (ball_data[format_name]['inning'] == inning)
    ]
    
    if match_data.empty:
        return pd.DataFrame()
    
    # Identify partnerships
    partnerships = []
    current_partners = set()
    partnership_runs = 0
    partnership_balls = 0
    start_over = 0
    
    for _, ball in match_data.iterrows():
        if pd.notna(ball['wicket_kind']):
            # Wicket fell, record partnership if it had 2 batsmen
            if len(current_partners) == 2:
                partnerships.append({
                    'batsman1': list(current_partners)[0],
                    'batsman2': list(current_partners)[1],
                    'runs': partnership_runs,
                    'balls': partnership_balls,
                    'start_over': start_over,
                    'end_over': ball['over']
                })
            partnership_runs = 0
            partnership_balls = 0
            current_partners = set()
            start_over = ball['over'] + 0.1 if ball['over'] < 20 else 20
        else:
            # Add batsmen to partnership
            current_partners.add(ball['batsman'])
            if ball['non_striker'] and pd.notna(ball['non_striker']):
                current_partners.add(ball['non_striker'])
            
            # If we have 2 batsmen, count the partnership
            if len(current_partners) == 2:
                partnership_runs += ball['runs_total']
                partnership_balls += 1
            elif partnership_balls == 0:
                start_over = ball['over']
    
    return pd.DataFrame(partnerships)

def bowler_spell_analysis(match_id, bowler_name, format_name='T20'):
    """Analyze a bowler's spell in a match"""
    if format_name not in ball_data:
        return pd.DataFrame()
        
    bowler_data = ball_data[format_name][
        (ball_data[format_name]['match_id'] == match_id) & 
        (ball_data[format_name]['bowler'] == bowler_name)
    ]
    
    if bowler_data.empty:
        return pd.DataFrame()
    
    # Calculate performance by over
    spell_stats = bowler_data.groupby('over').agg({
        'runs_total': 'sum',
        'wicket_kind': lambda x: x.notna().sum(),
        'ball': 'count'
    }).reset_index()
    
    spell_stats.rename(columns={'ball': 'balls_bowled'}, inplace=True)
    return spell_stats

def phase_analysis(match_id, team, format_name='T20'):
    """Analyze performance by phase of play"""
    if format_name not in ball_data:
        return pd.DataFrame()
        
    match_data = ball_data[format_name][
        (ball_data[format_name]['match_id'] == match_id) & 
        (ball_data[format_name]['batting_team'] == team)
    ]
    
    if match_data.empty:
        return pd.DataFrame()
    
    # Define phases (Powerplay, Middle, Death)
    match_data['phase'] = pd.cut(match_data['over'], 
                                bins=[0, 6, 15, 20],
                                labels=['Powerplay', 'Middle', 'Death'])
    
    phase_stats = match_data.groupby('phase').agg({
        'runs_total': ['sum', 'mean'],
        'wicket_kind': lambda x: x.notna().sum(),
        'ball': 'count'
    }).reset_index()
    
    phase_stats.columns = ['phase', 'total_runs', 'avg_runs_per_over', 'wickets', 'balls']
    return phase_stats

# Player form and prediction functions
def calculate_player_form(player_name, role, format_name, days_back=365, min_matches=3):
    """Calculate player form based on recent performances"""
    if format_name not in ball_data:
        return None
        
    current_date = pd.to_datetime('today')
    
    if role == 'batsman':
        player_matches = ball_data[format_name][ball_data[format_name]['batsman'] == player_name]
    else:
        player_matches = ball_data[format_name][ball_data[format_name]['bowler'] == player_name]
    
    if player_matches.empty:
        return None
        
    # Filter by date
    player_matches['date'] = pd.to_datetime(player_matches['date'])
    recent_matches = player_matches[player_matches['date'] > (current_date - timedelta(days=days_back))]
    
    if len(recent_matches['match_id'].unique()) < min_matches:
        return None  # Not enough recent data
    
    # Calculate form metrics
    if role == 'batsman':
        # For batsmen, we can count the number of rows as balls faced
        batting_stats = recent_matches.groupby('match_id').agg({
            'runs_batsman': 'sum',
        }).reset_index()
        
        # Count balls by counting the number of deliveries per match
        ball_counts = recent_matches.groupby('match_id').size().reset_index(name='balls')
        batting_stats = batting_stats.merge(ball_counts, on='match_id')
        
        if batting_stats.empty:
            return None
            
        total_runs = batting_stats['runs_batsman'].sum()
        total_balls = batting_stats['balls'].sum()
        
        if total_balls == 0:
            return 0
            
        strike_rate = (total_runs / total_balls) * 100
        average_runs = total_runs / len(batting_stats)
        
        # Simple form score (0-100)
        form_score = min(100, (average_runs / 50 * 40) + (strike_rate / 200 * 60))
        return form_score
    else:
        # For bowlers, we can count the number of rows as balls bowled
        bowling_stats = recent_matches.groupby('match_id').agg({
            'runs_total': 'sum',
            'wicket_kind': 'sum'  # Sum of wickets (already counted as 1s and 0s)
        }).reset_index()
        
        # Count balls by counting the number of deliveries per match
        ball_counts = recent_matches.groupby('match_id').size().reset_index(name='balls')
        bowling_stats = bowling_stats.merge(ball_counts, on='match_id')
        
        if bowling_stats.empty:
            return None
            
        total_wickets = bowling_stats['wicket_kind'].sum()
        total_runs = bowling_stats['runs_total'].sum()
        total_balls = bowling_stats['balls'].sum()
        
        if total_balls == 0:
            return 0
            
        economy = (total_runs / total_balls) * 6
        average = total_runs / total_wickets if total_wickets > 0 else total_runs
        
        # Simple form score (0-100), lower economy and average are better
        form_score = min(100, max(0, 100 - (economy * 5) - (average / 2)))
        return form_score

def predict_next_match_performance(player_name, role, opponent, venue, pitch_conditions, format_name=None):
    """Predict player performance for next match using ML model"""
    if format_name is None:
        format_name = selected_format
    
    # Try to use the ML model first
    try:
        # Determine if home game (simplified logic)
        is_home = 1  # Default to home
        
        # Load models if not already loaded
        if format_name not in model_predictor.batting_models:
            model_predictor.load_models(format_name)
        
        # Use ML model for prediction
        prediction = model_predictor.predict_player_performance(
            player_name, role, opponent, venue, is_home, format_name
        )
        
        if prediction:
            return prediction
    except Exception as e:
        st.warning(f"ML model prediction failed, using fallback: {str(e)}")
    
    # Fallback to the original simple logic if ML model fails
    base_performance = {
        'batsman': {'runs': 25, 'strike_rate': 120},
        'bowler': {'wickets': 1.5, 'economy': 7.5}
    }
    
    # Get historical data
    if format_name not in ball_data:
        return base_performance[role]
        
    player_data = ball_data[format_name][
        (ball_data[format_name]['batsman'] == player_name) | 
        (ball_data[format_name]['bowler'] == player_name)
    ]
    
    if player_data.empty:
        return base_performance[role]
    
    # Calculate opponent factor
    opponent_factor = 1.0
    
    if role == 'batsman':
        # For batsmen: find matches where they batted against this opponent (team2 == opponent)
        opponent_matches = player_data[
            (player_data['batsman'] == player_name) &
            (player_data['team2'] == opponent)
        ]
        
        if not opponent_matches.empty:
            opponent_runs = opponent_matches['runs_batsman'].sum() / len(opponent_matches['match_id'].unique())
            opponent_factor = opponent_runs / base_performance['batsman']['runs']
    else:
        # For bowlers: find matches where they bowled against this opponent (batting_team == opponent)
        opponent_matches = player_data[
            (player_data['bowler'] == player_name) &
            (player_data['batting_team'] == opponent)
        ]
        
        if not opponent_matches.empty:
            opponent_wickets = opponent_matches['wicket_kind'].notna().sum() / len(opponent_matches['match_id'].unique())
            opponent_factor = opponent_wickets / base_performance['bowler']['wickets']
    
    # Calculate venue factor
    venue_factor = 1.0
    
    venue_matches = player_data[player_data['venue'] == venue]
    
    if not venue_matches.empty:
        if role == 'batsman':
            venue_batting = venue_matches[venue_matches['batsman'] == player_name]
            if not venue_batting.empty:
                venue_runs = venue_batting['runs_batsman'].sum() / len(venue_batting['match_id'].unique())
                venue_factor = venue_runs / base_performance['batsman']['runs']
        else:
            venue_bowling = venue_matches[venue_matches['bowler'] == player_name]
            if not venue_bowling.empty:
                venue_wickets = venue_bowling['wicket_kind'].notna().sum() / len(venue_bowling['match_id'].unique())
                venue_factor = venue_wickets / base_performance['bowler']['wickets']
    
    # Calculate form
    form = calculate_player_form(player_name, role, format_name=format_name) or 50
    form_factor = form / 50  # Convert 0-100 scale to factor
    
    # Apply factors to base performance
    if role == 'batsman':
        predicted_runs = base_performance['batsman']['runs'] * opponent_factor * venue_factor * form_factor
        predicted_sr = base_performance['batsman']['strike_rate'] * form_factor
        
        return {
            'predicted_runs': round(predicted_runs, 1),
            'predicted_strike_rate': round(predicted_sr, 1),
            'confidence': min(0.9, (len(player_data) / 50) * 0.9)  # Simple confidence metric
        }
    else:
        predicted_wickets = base_performance['bowler']['wickets'] * opponent_factor * venue_factor * form_factor
        predicted_economy = base_performance['bowler']['economy'] / form_factor  # Lower economy is better
        
        return {
            'predicted_wickets': round(predicted_wickets, 1),
            'predicted_economy': round(predicted_economy, 1),
            'confidence': min(0.9, (len(player_data) / 50) * 0.9)  # Simple confidence metric
        }

# Team selection functions
def recommend_playing_xi(team_name, opponent, venue, pitch_conditions='normal', format_name=None):
    """Recommend optimal Playing XI based on ML model"""
    if format_name is None:
        format_name = selected_format
    
    # Try to use the ML model first
    try:
        # Load models if not already loaded
        if format_name not in model_predictor.xi_models:
            model_predictor.load_models(format_name)
        
        # Use ML model for prediction
        optimal_xi = model_predictor.predict_playing_xi(team_name, format_name)
        
        if optimal_xi is not None:
            # Add additional context factors (opponent, venue, pitch)
            for _, player in optimal_xi.iterrows():
                # Calculate opponent performance factor
                opponent_performance = 50  # Default
                
                if player['role'] in ['batsman', 'all_rounder']:
                    opponent_matches = ball_data[format_name][
                        (ball_data[format_name]['batsman'] == player['player']) &
                        (ball_data[format_name]['team2'] == opponent)
                    ]
                else:
                    opponent_matches = ball_data[format_name][
                        (ball_data[format_name]['bowler'] == player['player']) &
                        (ball_data[format_name]['batting_team'] == opponent)
                    ]
                
                if not opponent_matches.empty:
                    if player['role'] in ['batsman', 'all_rounder']:
                        runs = opponent_matches['runs_batsman'].sum()
                        innings = len(opponent_matches['match_id'].unique())
                        opponent_performance = min(100, (runs / innings) / 50 * 100) if innings > 0 else 50
                    else:
                        wickets = opponent_matches['wicket_kind'].notna().sum()
                        innings = len(opponent_matches['match_id'].unique())
                        opponent_performance = min(100, (wickets / innings) / 2 * 100) if innings > 0 else 50
                
                # Calculate venue performance factor
                venue_performance = 50  # Default
                venue_matches = ball_data[format_name][
                    ((ball_data[format_name]['batsman'] == player['player']) | 
                     (ball_data[format_name]['bowler'] == player['player'])) &
                    (ball_data[format_name]['venue'] == venue)
                ]
                
                if not venue_matches.empty:
                    if player['role'] in ['batsman', 'all_rounder']:
                        runs = venue_matches['runs_batsman'].sum()
                        innings = len(venue_matches['match_id'].unique())
                        venue_performance = min(100, (runs / innings) / 50 * 100) if innings > 0 else 50
                    else:
                        wickets = venue_matches['wicket_kind'].notna().sum()
                        innings = len(venue_matches['match_id'].unique())
                        venue_performance = min(100, (wickets / innings) / 2 * 100) if innings > 0 else 50
                
                # Adjust selection probability based on context
                context_factor = (opponent_performance * 0.3 + venue_performance * 0.3 + 40) / 100
                optimal_xi.at[_, 'composite_score'] = player['selection_probability'] * context_factor
                optimal_xi.at[_, 'opponent_performance'] = opponent_performance
                optimal_xi.at[_, 'venue_performance'] = venue_performance
                optimal_xi.at[_, 'form'] = player['selection_probability'] * 100  # Convert to 0-100 scale
            
            return optimal_xi.sort_values('composite_score', ascending=False)
    except Exception as e:
        st.warning(f"ML model prediction failed, using fallback: {str(e)}")
    
    # Fallback to the original logic if ML model fails
    # Get team players from ball-by-ball data
    if format_name not in ball_data:
        return pd.DataFrame()
    
    # Get team players from ball-by-ball data
    # Batsmen: when team is batting (batting_team == team_name)
    team_players = list(ball_data[format_name][ball_data[format_name]['batting_team'] == team_name]['batsman'].unique())
    
    # Bowlers: when team is bowling (team2 == team_name)
    team_players += list(ball_data[format_name][ball_data[format_name]['team2'] == team_name]['bowler'].unique())
    team_players = list(set([p for p in team_players if pd.notna(p)]))
    
    if not team_players:
        return pd.DataFrame()
    
    player_recommendations = []
    
    for player in team_players:
        # Determine player role (simplified)
        if player in ball_data[format_name]['batsman'].values:
            if player in ball_data[format_name]['bowler'].values:
                role = 'all_rounder'
            else:
                role = 'batsman'
        else:
            role = 'bowler'
        
        # Calculate form
        form = calculate_player_form(player, role, format_name=format_name) or 50
        
        # Calculate opponent performance
        opponent_performance = 50  # Default
        
        if role == 'batsman':
            # For batsmen: find matches where they batted against this opponent (team2 == opponent)
            opponent_matches = ball_data[format_name][
                (ball_data[format_name]['batsman'] == player) &
                (ball_data[format_name]['team2'] == opponent)
            ]
        else:
            # For bowlers: find matches where they bowled against this opponent (batting_team == opponent)
            opponent_matches = ball_data[format_name][
                (ball_data[format_name]['bowler'] == player) &
                (ball_data[format_name]['batting_team'] == opponent)
            ]
        
        if not opponent_matches.empty:
            if role == 'batsman':
                runs = opponent_matches['runs_batsman'].sum()
                innings = len(opponent_matches['match_id'].unique())
                opponent_performance = min(100, (runs / innings) / 50 * 100) if innings > 0 else 50
            else:
                wickets = opponent_matches['wicket_kind'].notna().sum()
                innings = len(opponent_matches['match_id'].unique())
                opponent_performance = min(100, (wickets / innings) / 2 * 100) if innings > 0 else 50
        
        # Calculate venue performance
        venue_performance = 50  # Default
        venue_matches = ball_data[format_name][
            ((ball_data[format_name]['batsman'] == player) | 
             (ball_data[format_name]['bowler'] == player)) &
            (ball_data[format_name]['venue'] == venue)
        ]
        
        if not venue_matches.empty:
            if role == 'batsman':
                runs = venue_matches['runs_batsman'].sum()
                innings = len(venue_matches['match_id'].unique())
                venue_performance = min(100, (runs / innings) / 50 * 100) if innings > 0 else 50
            else:
                wickets = venue_matches['wicket_kind'].notna().sum()
                innings = len(venue_matches['match_id'].unique())
                venue_performance = min(100, (wickets / innings) / 2 * 100) if innings > 0 else 50
        
        # Composite score
        composite_score = (form * 0.4) + (opponent_performance * 0.3) + (venue_performance * 0.3)
        
        player_recommendations.append({
            'player': player,
            'role': role,
            'composite_score': composite_score,
            'form': form,
            'opponent_performance': opponent_performance,
            'venue_performance': venue_performance
        })
    
    player_df = pd.DataFrame(player_recommendations)
    
    # Select optimal XI (simplified selection logic)
    optimal_xi = player_df.nlargest(11, 'composite_score')
    return optimal_xi.sort_values('composite_score', ascending=False)

# Trend analysis functions
def player_performance_trends(player_name, role, format_name, days_back=365):
    """Analyze player performance trends over time"""
    if format_name not in ball_data:
        return pd.DataFrame()
        
    if role == 'batsman':
        player_data = ball_data[format_name][ball_data[format_name]['batsman'] == player_name]
    else:
        player_data = ball_data[format_name][ball_data[format_name]['bowler'] == player_name]
    
    if player_data.empty:
        return pd.DataFrame()
    
    player_data['date'] = pd.to_datetime(player_data['date'])
    cutoff_date = pd.to_datetime('today') - timedelta(days=days_back)
    recent_data = player_data[player_data['date'] > cutoff_date]
    
    if recent_data.empty:
        return pd.DataFrame()
    
    # Group by match and calculate performance metrics
    if role == 'batsman':
        match_stats = recent_data.groupby(['match_id', 'date']).agg({
            'runs_batsman': 'sum',
            'wicket_kind': 'count'  # Count wickets (dismissals)
        }).reset_index()
        
        # Count balls by counting the number of deliveries per match
        ball_counts = recent_data.groupby(['match_id', 'date']).size().reset_index(name='balls')
        match_stats = match_stats.merge(ball_counts, on=['match_id', 'date'])
        
    else:
        match_stats = recent_data.groupby(['match_id', 'date']).agg({
            'runs_total': 'sum',
            'wicket_kind': 'sum'  # Sum of wickets (already counted as 1s and 0s)
        }).reset_index()
        
        # Count balls by counting the number of deliveries per match
        ball_counts = recent_data.groupby(['match_id', 'date']).size().reset_index(name='balls')
        match_stats = match_stats.merge(ball_counts, on=['match_id', 'date'])
    
    # Calculate moving averages
    match_stats = match_stats.sort_values('date')
    
    if role == 'batsman':
        match_stats['runs_ma_5'] = match_stats['runs_batsman'].rolling(window=5, min_periods=1).mean()
        match_stats['strike_rate'] = (match_stats['runs_batsman'] / match_stats['balls']) * 100
        match_stats['strike_rate_ma_5'] = match_stats['strike_rate'].rolling(window=5, min_periods=1).mean()
    else:
        match_stats['economy'] = (match_stats['runs_total'] / match_stats['balls']) * 6
        match_stats['economy_ma_5'] = match_stats['economy'].rolling(window=5, min_periods=1).mean()
        match_stats['wickets_ma_5'] = match_stats['wicket_kind'].rolling(window=5, min_periods=1).mean()
    
    return match_stats

# Sidebar
st.sidebar.title("🏏 Dashboard Controls")
st.sidebar.markdown("---")

# Global format selection at the top level
available_formats = list(batting_data.keys()) if batting_data else list(bowling_data.keys()) if bowling_data else []
if not available_formats:
    st.error("No data available. Please load data files.")
    st.stop()

selected_format = st.sidebar.selectbox("Select Format", available_formats)

# Main navigation
main_tab = st.sidebar.radio("Select Analysis Type", 
                           ["Team & Player Stats", "Player vs Player", "Batsman vs Bowler", 
                            "Team Selection Assistant", "Players by Team", "Match Analysis",
                            "Ball-by-Ball Analysis", "Trend Analysis", "Performance Predictor"])

# Format selection for traditional stats
if main_tab == "Team & Player Stats":
    stats_type = st.sidebar.radio("Select Statistics Type", ["Batting", "Bowling", "Fielding"])
    
    if stats_type == "Batting":
        df = batting_data[selected_format]
    elif stats_type == "Bowling":
        df = bowling_data[selected_format]
    else:
        df = fielding_data[selected_format]

    # Filters
    st.sidebar.markdown("### 🔧 Filters")
    
    if stats_type == "Batting":
        min_matches = st.sidebar.slider("Minimum Matches", 1, 300, 1)
        min_runs = st.sidebar.slider("Minimum Runs", 0, 10000, 0)
        min_innings = st.sidebar.slider("Minimum Innings", 1, 300, 1)
        
        # Apply filters
        filtered_df = df[
            (df['matches'] >= min_matches) & 
            (df['runs'] >= min_runs) &
            (df['innings'] >= min_innings)
        ].copy()
        
        # Format-specific adjustments
        if selected_format == 'T20':
            sr_threshold = 100
            boundary_multiplier = 1.5
        elif selected_format == 'Test':
            sr_threshold = 50
            boundary_multiplier = 0.8
        else:  # ODI
            sr_threshold = 75
            boundary_multiplier = 1.0
            
    elif stats_type == "Bowling":
        min_matches = st.sidebar.slider("Minimum Matches", 1, 300, 1)
        min_wickets = st.sidebar.slider("Minimum Wickets", 0, 500, 0)
        min_balls = st.sidebar.slider("Minimum Balls Bowled", 0, 10000, 0)
        
        # Apply filters
        filtered_df = df[
            (df['matches'] >= min_matches) & 
            (df['wickets'] >= min_wickets) &
            (df['balls_bowled'] >= min_balls)
        ].copy()
        
        # Format-specific adjustments
        if selected_format == 'T20':
            economy_threshold = 7.0
            avg_threshold = 25
        elif selected_format == 'Test':
            economy_threshold = 3.0
            avg_threshold = 30
        else:  # ODI
            economy_threshold = 5.0
            avg_threshold = 30
            
    else:  # Fielding
        min_matches = st.sidebar.slider("Minimum Matches", 1, 300, 1)
        min_dismissals = st.sidebar.slider("Minimum Dismissals", 0, 200, 0)
        
        # Apply filters
        filtered_df = df[
            (df['matches'] >= min_matches) & 
            (df['dismissals'] >= min_dismissals)
        ].copy()

# Main content based on selected tab
if main_tab == "Team & Player Stats":
    # Main content
    header_text = f'Cricket {stats_type} Statistics Dashboard <span class="format-badge">{selected_format}</span>'
    st.markdown(f'<h1 class="main-header">{header_text}</h1>', unsafe_allow_html=True)

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    if stats_type == "Batting":
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Players", len(filtered_df))
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Average Runs", f"{filtered_df['runs'].mean():.0f}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Average Strike Rate", f"{filtered_df['strike_rate'].mean():.1f}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            total_centuries = filtered_df['centuries'].sum() if 'centuries' in filtered_df.columns else 0
            st.metric("Total Centuries", f"{total_centuries:,}")
            st.markdown('</div>', unsafe_allow_html=True)
            
    elif stats_type == "Bowling":
        with col1:
            st.markdown('<div class="metric-card bowling-metric">', unsafe_allow_html=True)
            st.metric("Total Bowlers", len(filtered_df))
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-card bowling-metric">', unsafe_allow_html=True)
            st.metric("Total Wickets", f"{filtered_df['wickets'].sum():.0f}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="metric-card bowling-metric">', unsafe_allow_html=True)
            st.metric("Average Economy", f"{filtered_df['economy'].mean():.2f}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="metric-card bowling-metric">', unsafe_allow_html=True)
            five_wickets = filtered_df['five_wickets'].sum() if 'five_wickets' in filtered_df.columns else 0
            st.metric("5-Wicket Hauls", f"{five_wickets}")
            st.markdown('</div>', unsafe_allow_html=True)
            
    else:  # Fielding
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Fielders", len(filtered_df))
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Dismissals", f"{filtered_df['dismissals'].sum():.0f}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Average Catches", f"{filtered_df['catches'].mean():.1f}" if 'catches' in filtered_df.columns else "N/A")
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Dismissals/Inning", f"{filtered_df['dismissals_per_inning'].mean():.2f}" if 'dismissals_per_inning' in filtered_df.columns else "N/A")
            st.markdown('</div>', unsafe_allow_html=True)

    # Tabs for different analyses
    if stats_type == "Batting":
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🏆 Top Performers", 
            "📊 Distributions", 
            "📈 Relationships", 
            "🎯 Player Efficiency", 
            "🔍 Player Search",
            "📋 Format Comparison"
        ])
    elif stats_type == "Bowling":
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🏆 Top Performers", 
            "📊 Distributions", 
            "📈 Relationships", 
            "🎯 Bowler Efficiency", 
            "🔍 Player Search",
            "📋 Format Comparison"
        ])
    else:  # Fielding
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏆 Top Performers", 
            "📊 Distributions", 
            "📈 Relationships", 
            "🔍 Player Search",
            "📋 Format Comparison"
        ])

    with tab1:
        if stats_type == "Batting":
            st.markdown('<h2 class="section-header">Top Batters</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Top run scorers
                top_runs = filtered_df.nlargest(10, 'runs')[['batsman', 'runs', 'matches', 'strike_rate']]
                fig = px.bar(top_runs, x='runs', y='batsman', orientation='h',
                            title=f'Top 10 Run Scorers ({selected_format})', color='runs',
                            color_continuous_scale='Viridis')
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Top strike rates
                min_runs_for_sr = 100 if selected_format == 'T20' else 500
                high_scorers = filtered_df[filtered_df['runs'] >= min_runs_for_sr]
                if not high_scorers.empty:
                    top_sr = high_scorers.nlargest(10, 'strike_rate')[['batsman', 'strike_rate', 'runs', 'matches']]
                    fig = px.bar(top_sr, x='strike_rate', y='batsman', orientation='h',
                                title=f'Top 10 Strike Rates (Min {min_runs_for_sr} runs)', color='strike_rate',
                                color_continuous_scale='Plasma')
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No players with {min_runs_for_sr}+ runs found with current filters")
            
            col3, col4 = st.columns(2)
            
            with col3:
                # Most centuries
                if 'centuries' in filtered_df.columns:
                    top_centuries = filtered_df.nlargest(10, 'centuries')[['batsman', 'centuries', 'runs', 'matches']]
                    fig = px.bar(top_centuries, x='centuries', y='batsman', orientation='h',
                                title='Top 10 Century Makers', color='centuries',
                                color_continuous_scale='Turbo')
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Century data not available")
            
            with col4:
                # Most consistent (high average)
                min_matches_consistent = 10 if selected_format == 'T20' else 20
                consistent_players = filtered_df[filtered_df['matches'] >= min_matches_consistent]
                if not consistent_players.empty:
                    consistent_players['runs_per_match'] = consistent_players['runs'] / consistent_players['matches']
                    top_consistent = consistent_players.nlargest(10, 'runs_per_match')[['batsman', 'runs_per_match', 'matches', 'runs']]
                    fig = px.bar(top_consistent, x='runs_per_match', y='batsman', orientation='h',
                                title=f'Most Consistent Batters (Min {min_matches_consistent} matches)', color='runs_per_match',
                                color_continuous_scale='Rainbow')
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No players with {min_matches_consistent}+ matches found with current filters")
                    
        elif stats_type == "Bowling":
            st.markdown('<h2 class="section-header bowling-header">Top Bowlers</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Most wickets
                top_wickets = filtered_df.nlargest(10, 'wickets')[['bowler', 'wickets', 'matches', 'economy']]
                fig = px.bar(top_wickets, x='wickets', y='bowler', orientation='h',
                            title=f'Top 10 Wicket Takers ({selected_format})', color='wickets',
                            color_continuous_scale='Viridis')
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Best economy rates
                min_wickets_for_economy = 10 if selected_format == 'T20' else 20
                qualified_bowlers = filtered_df[filtered_df['wickets'] >= min_wickets_for_economy]
                if not qualified_bowlers.empty:
                    top_economy = qualified_bowlers.nsmallest(10, 'economy')[['bowler', 'economy', 'wickets', 'matches']]
                    fig = px.bar(top_economy, x='economy', y='bowler', orientation='h',
                                title=f'Best Economy Rates (Min {min_wickets_for_economy} wickets)', color='economy',
                                color_continuous_scale='Plasma')
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No bowlers with {min_wickets_for_economy}+ wickets found with current filters")
            
            col3, col4 = st.columns(2)
            
            with col3:
                # Best bowling averages
                min_wickets_for_avg = 10 if selected_format == 'T20' else 20
                qualified_bowlers = filtered_df[filtered_df['wickets'] >= min_wickets_for_avg]
                if not qualified_bowlers.empty and 'bowling_average' in qualified_bowlers.columns:
                    top_avg = qualified_bowlers.nsmallest(10, 'bowling_average')[['bowler', 'bowling_average', 'wickets', 'economy']]
                    fig = px.bar(top_avg, x='bowling_average', y='bowler', orientation='h',
                                title=f'Best Bowling Averages (Min {min_wickets_for_avg} wickets)', color='bowling_average',
                                color_continuous_scale='Turbo')
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No bowlers with {min_wickets_for_avg}+ wickets found with current filters or bowling average data not available")
            
            with col4:
                # Most 5-wicket hauls
                if 'five_wickets' in filtered_df.columns:
                    top_five_wickets = filtered_df.nlargest(10, 'five_wickets')[['bowler', 'five_wickets', 'wickets', 'matches']]
                    fig = px.bar(top_five_wickets, x='five_wickets', y='bowler', orientation='h',
                                title='Most 5-Wicket Hauls', color='five_wickets',
                                color_continuous_scale='Rainbow')
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("5-wicket haul data not available")
                    
        else:  # Fielding
            st.markdown('<h2 class="section-header">Top Fielders</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Most dismissals
                top_dismissals = filtered_df.nlargest(10, 'dismissals')[['player', 'dismissals', 'matches', 'innings']]
                fig = px.bar(top_dismissals, x='dismissals', y='player', orientation='h',
                            title=f'Top 10 Fielders by Dismissals ({selected_format})', color='dismissals',
                            color_continuous_scale='Viridis')
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Best dismissal rate
                min_matches_fielding = 10 if selected_format == 'T20' else 20
                qualified_fielders = filtered_df[filtered_df['matches'] >= min_matches_fielding]
                if not qualified_fielders.empty and 'dismissals_per_inning' in qualified_fielders.columns:
                    top_dismissal_rate = qualified_fielders.nlargest(10, 'dismissals_per_inning')[['player', 'dismissals_per_inning', 'dismissals', 'matches']]
                    fig = px.bar(top_dismissal_rate, x='dismissals_per_inning', y='player', orientation='h',
                                title=f'Best Dismissal Rate (Min {min_matches_fielding} matches)', color='dismissals_per_inning',
                                color_continuous_scale='Plasma')
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No fielders with {min_matches_fielding}+ matches found with current filters")
            
            col3, col4 = st.columns(2)
            
            with col3:
                # Most catches
                if 'catches' in filtered_df.columns:
                    top_catches = filtered_df.nlargest(10, 'catches')[['player', 'catches', 'dismissals', 'matches']]
                    fig = px.bar(top_catches, x='catches', y='player', orientation='h',
                                title='Most Catches', color='catches',
                                color_continuous_scale='Turbo')
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Catches data not available")
            
            with col4:
                # Most stumpings
                if 'stumpings' in filtered_df.columns:
                    top_stumpings = filtered_df.nlargest(10, 'stumpings')[['player', 'stumpings', 'dismissals', 'matches']]
                    fig = px.bar(top_stumpings, x='stumpings', y='player', orientation='h',
                                title='Most Stumpings', color='stumpings',
                                color_continuous_scale='Rainbow')
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Stumpings data not available")

    with tab2:
        if stats_type == "Batting":
            st.markdown('<h2 class="section-header">Statistical Distributions</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Runs distribution
                fig = px.histogram(filtered_df, x='runs', nbins=50, 
                                  title='Distribution of Runs', marginal='box')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Strike rate distribution
                fig = px.histogram(filtered_df, x='strike_rate', nbins=50, 
                                  title='Distribution of Strike Rates', marginal='box')
                st.plotly_chart(fig, use_container_width=True)
            
            col3, col4 = st.columns(2)
            
            with col3:
                # Matches distribution
                fig = px.histogram(filtered_df, x='matches', nbins=30, 
                                  title='Distribution of Matches Played', marginal='box')
                st.plotly_chart(fig, use_container_width=True)
            
            with col4:
                # Centuries vs Fifties
                if 'centuries' in filtered_df.columns and 'fifties' in filtered_df.columns:
                    fig = px.scatter(filtered_df, x='centuries', y='fifties', 
                                    color='strike_rate', size='runs',
                                    title='Centuries vs Fifties (Size: Total Runs, Color: Strike Rate)',
                                    hover_data=['batsman'])
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Century/fifty data not available")
                
        elif stats_type == "Bowling":
            st.markdown('<h2 class="section-header bowling-header">Statistical Distributions</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Wickets distribution
                fig = px.histogram(filtered_df, x='wickets', nbins=50, 
                                  title='Distribution of Wickets', marginal='box')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Economy distribution
                fig = px.histogram(filtered_df, x='economy', nbins=50, 
                                  title='Distribution of Economy Rates', marginal='box')
                st.plotly_chart(fig, use_container_width=True)
            
            col3, col4 = st.columns(2)
            
            with col3:
                # Bowling average distribution if available
                if 'bowling_average' in filtered_df.columns:
                    fig = px.histogram(filtered_df, x='bowling_average', nbins=50, 
                                      title='Distribution of Bowling Averages', marginal='box')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Bowling average data not available")
            
            with col4:
                # Wickets vs Economy
                fig = px.scatter(filtered_df, x='economy', y='wickets', 
                                color='bowling_strike_rate' if 'bowling_strike_rate' in filtered_df.columns else 'matches', 
                                size='matches',
                                title='Economy vs Wickets (Size: Matches, Color: Strike Rate)',
                                hover_data=['bowler'])
                st.plotly_chart(fig, use_container_width=True)
                
        else:  # Fielding
            st.markdown('<h2 class="section-header">Statistical Distributions</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Dismissals distribution
                fig = px.histogram(filtered_df, x='dismissals', nbins=50, 
                                  title='Distribution of Dismissals', marginal='box')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Dismissals per inning distribution if available
                if 'dismissals_per_inning' in filtered_df.columns:
                    fig = px.histogram(filtered_df, x='dismissals_per_inning', nbins=50, 
                                      title='Distribution of Dismissals per Inning', marginal='box')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Dismissals per inning data not available")
            
            col3, col4 = st.columns(2)
            
            with col3:
                # Catches distribution if available
                if 'catches' in filtered_df.columns:
                    fig = px.histogram(filtered_df, x='catches', nbins=50, 
                                      title='Distribution of Catches', marginal='box')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Catches data not available")
            
            with col4:
                # Stumpings distribution if available
                if 'stumpings' in filtered_df.columns:
                    fig = px.histogram(filtered_df, x='stumpings', nbins=50, 
                                      title='Distribution of Stumpings', marginal='box')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Stumpings data not available")

    with tab3:
        if stats_type == "Batting":
            st.markdown('<h2 class="section-header">Relationship Analysis</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Runs vs Strike Rate
                size_col = 'balls_faced' if 'balls_faced' in filtered_df.columns else 'runs'
                fig = px.scatter(filtered_df, x='runs', y='strike_rate', 
                                color='matches', size=size_col,
                                title='Runs vs Strike Rate',
                                hover_data=['batsman'])
                # Add format-specific reference line
                fig.add_hline(y=sr_threshold, line_dash="dash", line_color="red", 
                             annotation_text=f"Format Avg: {sr_threshold}")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Experience vs Performance
                fig =px.scatter(filtered_df, x='matches', y='runs', 
                                color='strike_rate', size=size_col,
                                title='Experience (Matches) vs Total Runs',
                                hover_data=['batsman'])
                st.plotly_chart(fig, use_container_width=True)
            
            # Correlation heatmap
            st.markdown('<h3>Correlation Matrix</h3>', unsafe_allow_html=True)
            numeric_cols = ['runs', 'strike_rate', 'matches', 'innings']
            if 'balls_faced' in filtered_df.columns:
                numeric_cols.append('balls_faced')
            if 'centuries' in filtered_df.columns:
                numeric_cols.append('centuries')
            if 'fifties' in filtered_df.columns:
                numeric_cols.append('fifties')
                
            corr_matrix = filtered_df[numeric_cols].corr()
            fig = px.imshow(corr_matrix, text_auto=True, aspect='auto', 
                           color_continuous_scale='RdBu_r', title='Correlation Matrix')
            st.plotly_chart(fig, use_container_width=True)
            
        elif stats_type == "Bowling":
            st.markdown('<h2 class="section-header bowling-header">Relationship Analysis</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Wickets vs Economy
                fig = px.scatter(filtered_df, x='economy', y='wickets', 
                                color='bowling_strike_rate' if 'bowling_strike_rate' in filtered_df.columns else 'matches', 
                                size='matches',
                                title='Economy vs Wickets (Size: Matches, Color: Strike Rate)',
                                hover_data=['bowler'])
                # Add format-specific reference line
                fig.add_vline(x=economy_threshold, line_dash="dash", line_color="red", 
                             annotation_text=f"Format Avg: {economy_threshold}")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Economy vs Average if available
                if 'bowling_average' in filtered_df.columns:
                    fig = px.scatter(filtered_df, x='economy', y='bowling_average', 
                                    color='wickets', size='matches',
                                    title='Economy vs Bowling Average (Size: Matches, Color: Wickets)',
                                    hover_data=['bowler'])
                    # Add format-specific reference lines
                    fig.add_vline(x=economy_threshold, line_dash="dash", line_color="red")
                    fig.add_hline(y=avg_threshold, line_dash="dash", line_color="red")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Bowling average data not available")
            
            # Correlation heatmap
            st.markdown('<h3>Correlation Matrix</h3>', unsafe_allow_html=True)
            numeric_cols = ['wickets', 'economy', 'matches']
            if 'runs_conceded' in filtered_df.columns:
                numeric_cols.append('runs_conceded')
            if 'bowling_average' in filtered_df.columns:
                numeric_cols.append('bowling_average')
            if 'bowling_strike_rate' in filtered_df.columns:
                numeric_cols.append('bowling_strike_rate')
            if 'four_wickets' in filtered_df.columns:
                numeric_cols.append('four_wickets')
            if 'five_wickets' in filtered_df.columns:
                numeric_cols.append('five_wickets')
                
            corr_matrix = filtered_df[numeric_cols].corr()
            fig = px.imshow(corr_matrix, text_auto=True, aspect='auto', 
                           color_continuous_scale='RdBu_r', title='Correlation Matrix')
            st.plotly_chart(fig, use_container_width=True)
            
        else:  # Fielding
            st.markdown('<h2 class="section-header">Relationship Analysis</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Dismissals vs Matches
                fig = px.scatter(filtered_df, x='matches', y='dismissals', 
                                color='dismissals_per_inning' if 'dismissals_per_inning' in filtered_df.columns else 'innings', 
                                size='innings',
                                title='Matches vs Dismissals',
                                hover_data=['player'])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Catches vs Stumpings if available
                if 'catches' in filtered_df.columns and 'stumpings' in filtered_df.columns:
                    fig = px.scatter(filtered_df, x='catches', y='stumpings', 
                                    color='dismissals', size='matches',
                                    title='Catches vs Stumpings (Size: Matches, Color: Dismissals)',
                                    hover_data=['player'])
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Catches/stumpings data not available")
            
            # Correlation heatmap
            st.markdown('<h3>Correlation Matrix</h3>', unsafe_allow_html=True)
            numeric_cols = ['dismissals', 'matches', 'innings']
            if 'catches' in filtered_df.columns:
                numeric_cols.append('catches')
            if 'stumpings' in filtered_df.columns:
                numeric_cols.append('stumpings')
            if 'dismissals_per_inning' in filtered_df.columns:
                numeric_cols.append('dismissals_per_inning')
                
            corr_matrix = filtered_df[numeric_cols].corr()
            fig = px.imshow(corr_matrix, text_auto=True, aspect='auto', 
                           color_continuous_scale='RdBu_r', title='Correlation Matrix')
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        if stats_type == "Batting":
            st.markdown('<h2 class="section-header">Batter Efficiency Analysis</h2>', unsafe_allow_html=True)
            
            # Efficiency metrics
            filtered_df['runs_per_match'] = filtered_df['runs'] / filtered_df['matches']
            
            if 'balls_faced' in filtered_df.columns and 'runs' in filtered_df.columns:
                filtered_df['runs_per_100_balls'] = (filtered_df['runs'] / filtered_df['balls_faced']) * 100
                filtered_df['runs_per_100_balls'] = filtered_df['runs_per_100_balls'].replace([np.inf, -np.inf], 0).fillna(0)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Runs per match distribution
                min_matches_efficient = 10 if selected_format == 'T20' else 20
                efficient_players = filtered_df[filtered_df['matches'] >= min_matches_efficient]
                if not efficient_players.empty:
                    top_efficient = efficient_players.nlargest(15, 'runs_per_match')
                    fig = px.bar(top_efficient, x='runs_per_match', y='batsman', orientation='h',
                                title=f'Most Efficient Batters (Min {min_matches_efficient} matches)', color='runs_per_match')
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No players with {min_matches_efficient}+ matches found with current filters")
            
            with col2:
                # Runs per 100 balls if available
                if 'runs_per_100_balls' in filtered_df.columns:
                    min_balls_faced = 100 if selected_format == 'T20' else 300
                    aggressive_batters = filtered_df[filtered_df['balls_faced'] >= min_balls_faced]
                    if not aggressive_batters.empty:
                        top_aggressive = aggressive_batters.nlargest(15, 'runs_per_100_balls')
                        fig =px.bar(top_aggressive, x='runs_per_100_balls', y='batsman', orientation='h',
                                    title=f'Most Aggressive Batters (Min {min_balls_faced} balls)', color='runs_per_100_balls')
                        fig.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(f"No batters with {min_balls_faced}+ balls found with current filters")
                else:
                    st.info("Balls faced data not available")
            
            # Experience categories analysis
            st.markdown('<h3>Performance by Experience Level</h3>', unsafe_allow_html=True)
            
            # Create experience categories with format-specific bins
            if selected_format == 'T20':
                bins = [0, 10, 25, 50, 100, 200]
            elif selected_format == 'Test':
                bins = [0, 10, 25, 50, 100, 200]
            else:  # ODI
                bins = [0, 10, 25, 50, 100, 500]
            
            labels = ['Rookie', 'Developing', 'Established', 'Experienced', 'Veteran']
            filtered_df['experience'] = pd.cut(filtered_df['matches'], bins=bins, labels=labels)
            
            exp_stats = filtered_df.groupby('experience').agg({
                'runs': 'mean',
                'strike_rate': 'mean',
                'runs_per_match': 'mean',
                'batsman': 'count'
            }).rename(columns={'batsman': 'player_count'}).reset_index()
            
            fig = make_subplots(rows=2, cols=2, subplot_titles=('Average Runs', 'Average Strike Rate', 'Runs per Match', 'Player Count'))
            
            fig.add_trace(go.Bar(x=exp_stats['experience'], y=exp_stats['runs'], name='Average Runs'), 1, 1)
            fig.add_trace(go.Bar(x=exp_stats['experience'], y=exp_stats['strike_rate'], name='Strike Rate'), 1, 2)
            fig.add_trace(go.Bar(x=exp_stats['experience'], y=exp_stats['runs_per_match'], name='Runs per Match'), 2, 1)
            fig.add_trace(go.Bar(x=exp_stats['experience'], y=exp_stats['player_count'], name='Player Count'), 2, 2)
            
            fig.update_layout(height=600, title_text="Performance Metrics by Experience Level", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
        elif stats_type == "Bowling":
            st.markdown('<h2 class="section-header bowling-header">Bowler Efficiency Analysis</h2>', unsafe_allow_html=True)
            
            # Efficiency metrics
            filtered_df['wickets_per_match'] = filtered_df['wickets'] / filtered_df['matches']
            
            if 'balls_bowled' in filtered_df.columns:
                filtered_df['wickets_per_100_balls'] = (filtered_df['wickets'] / filtered_df['balls_bowled']) * 100
                filtered_df['wickets_per_100_balls'] = filtered_df['wickets_per_100_balls'].replace([np.inf, -np.inf], 0).fillna(0)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Wickets per match
                min_matches_efficient = 10 if selected_format == 'T20' else 20
                efficient_bowlers = filtered_df[filtered_df['matches'] >=min_matches_efficient]
                if not efficient_bowlers.empty:
                    top_efficient = efficient_bowlers.nlargest(15, 'wickets_per_match')
                    fig = px.bar(top_efficient, x='wickets_per_match', y='bowler', orientation='h',
                                title=f'Most Efficient Bowlers (Min {min_matches_efficient} matches)', color='wickets_per_match')
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No bowlers with {min_matches_efficient}+ matches found with current filters")
            
            with col2:
                # Wickets per 100 balls if available
                if 'wickets_per_100_balls' in filtered_df.columns:
                    min_balls_bowled = 100 if selected_format == 'T20' else 300
                    strike_bowlers = filtered_df[filtered_df['balls_bowled'] >= min_balls_bowled]
                    if not strike_bowlers.empty:
                        top_strike = strike_bowlers.nlargest(15, 'wickets_per_100_balls')
                        fig =px.bar(top_strike, x='wickets_per_100_balls', y='bowler', orientation='h',
                                    title=f'Best Strike Bowlers (Min {min_balls_bowled} balls)', color='wickets_per_100_balls')
                        fig.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(f"No bowlers with {min_balls_bowled}+ balls found with current filters")
                else:
                    st.info("Balls bowled data not available")
            
            # Experience categories analysis
            st.markdown('<h3>Performance by Experience Level</h3>', unsafe_allow_html=True)
            
            # Create experience categories with format-specific bins
            if selected_format == 'T20':
                bins = [0, 10, 25, 50, 100, 200]
            elif selected_format == 'Test':
                bins = [0, 10, 25, 50, 100, 200]
            else:  # ODI
                bins = [0, 10, 25, 50, 100, 500]
            
            labels = ['Rookie', 'Developing', 'Established', 'Experienced', 'Veteran']
            filtered_df['experience'] = pd.cut(filtered_df['matches'], bins=bins, labels=labels)
            
            exp_stats = filtered_df.groupby('experience').agg({
                'wickets': 'mean',
                'economy': 'mean',
                'wickets_per_match': 'mean',
                'bowler': 'count'
            }).rename(columns={'bowler': 'player_count'}).reset_index()
            
            fig = make_subplots(rows=2, cols=2, subplot_titles=('Average Wickets', 'Average Economy', 'Wickets per Match', 'Player Count'))
            
            fig.add_trace(go.Bar(x=exp_stats['experience'], y=exp_stats['wickets'], name='Average Wickets'), 1, 1)
            fig.add_trace(go.Bar(x=exp_stats['experience'], y=exp_stats['economy'], name='Economy'), 1, 2)
            fig.add_trace(go.Bar(x=exp_stats['experience'], y=exp_stats['wickets_per_match'], name='Wickets per Match'), 2, 1)
            fig.add_trace(go.Bar(x=exp_stats['experience'], y=exp_stats['player_count'], name='Player Count'), 2, 2)
            
            fig.update_layout(height=600, title_text="Performance Metrics by Experience Level", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
        else:  # Fielding
            st.markdown('<h2 class="section-header">Fielder Efficiency Analysis</h2>', unsafe_allow_html=True)
            
            # Efficiency metrics
            filtered_df['dismissals_per_match'] = filtered_df['dismissals'] / filtered_df['matches']
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Dismissals per match
                min_matches_efficient = 10 if selected_format == 'T20' else 20
                efficient_fielders = filtered_df[filtered_df['matches'] >= min_matches_efficient]
                if not efficient_fielders.empty:
                    top_efficient = efficient_fielders.nlargest(15, 'dismissals_per_match')
                    fig = px.bar(top_efficient, x='dismissals_per_match', y='player', orientation='h',
                                title=f'Most Efficient Fielders (Min {min_matches_efficient} matches)', color='dismissals_per_match')
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No fielders with {min_matches_efficient}+ matches found with current filters")
            
            with col2:
                # Dismissals per inning if available
                if 'dismissals_per_inning' in filtered_df.columns:
                    min_innings = 10 if selected_format == 'T20' else 20
                    consistent_fielders = filtered_df[filtered_df['innings'] >= min_innings]
                    if not consistent_fielders.empty:
                        top_consistent = consistent_fielders.nlargest(15, 'dismissals_per_inning')
                        fig =px.bar(top_consistent, x='dismissals_per_inning', y='player', orientation='h',
                                    title=f'Most Consistent Fielders (Min {min_innings} innings)', color='dismissals_per_inning')
                        fig.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(f"No fielders with {min_innings}+ innings found with current filters")
                else:
                    st.info("Dismissals per inning data not available")
            
            # Experience categories analysis
            st.markdown('<h3>Performance by Experience Level</h3>', unsafe_allow_html=True)
            
            # Create experience categories with format-specific bins
            if selected_format == 'T20':
                bins = [0, 10, 25, 50, 100, 200]
            elif selected_format == 'Test':
                bins = [0, 10, 25, 50, 100, 200]
            else:  # ODI
                bins = [0, 10, 25, 50, 100, 500]
            
            labels = ['Rookie', 'Developing', 'Established', 'Experienced', 'Veteran']
            filtered_df['experience'] = pd.cut(filtered_df['matches'], bins=bins, labels=labels)
            
            exp_stats = filtered_df.groupby('experience').agg({
                'dismissals': 'mean',
                'dismissals_per_match': 'mean',
                'player': 'count'
            }).rename(columns={'player': 'player_count'}).reset_index()
            
            fig = make_subplots(rows=2, cols=2, subplot_titles=('Average Dismissals', 'Dismissals per Match', 'Player Count', ''))
            
            fig.add_trace(go.Bar(x=exp_stats['experience'], y=exp_stats['dismissals'], name='Average Dismissals'), 1, 1)
            fig.add_trace(go.Bar(x=exp_stats['experience'], y=exp_stats['dismissals_per_match'], name='Dismissals per Match'), 1, 2)
            fig.add_trace(go.Bar(x=exp_stats['experience'], y=exp_stats['player_count'], name='Player Count'), 2, 1)
            
            fig.update_layout(height=600, title_text="Performance Metrics by Experience Level", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    if stats_type != "Fielding":  # Fielding doesn't have tab5
        with tab5:
            if stats_type == "Batting":
                st.markdown('<h2 class="section-header">Player Search and Comparison</h2>', unsafe_allow_html=True)
                
                # Player search
                player_names = filtered_df['batsman'].unique()
                
                selected_players = st.multiselect('Select players to compare:', player_names, default=player_names[:3] if len(player_names) > 0 else [])
                
                if selected_players:
                    selected_data = filtered_df[filtered_df['batsman'].isin(selected_players)].copy()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Radar chart for comparison
                        metrics = ['runs', 'strike_rate', 'matches']
                        if 'centuries' in selected_data.columns:
                            metrics.append('centuries')
                        if 'fifties' in selected_data.columns:
                            metrics.append('fifties')
                            
                        fig = go.Figure()
                        
                        for _, player in selected_data.iterrows():
                            values = [player[metric] for metric in metrics]
                            # Normalize values for radar chart
                            max_vals = safe_max_calculation(filtered_df, metrics)
                            normalized_values = [values[i] / max_vals[metric] for i, metric in enumerate(metrics)]
                            
                            fig.add_trace(go.Scatterpolar(
                                r=normalized_values,
                                theta=metrics,
                                fill='toself',
                                name=player['batsman']
                            ))
                        
                        fig.update_layout(
                            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                            title='Player Comparison (Normalized Metrics)',
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Detailed player stats
                        display_cols = ['batsman', 'matches', 'innings', 'runs', 'strike_rate']
                        if 'balls_faced' in selected_data.columns:
                            display_cols.append('balls_faced')
                        if 'centuries' in selected_data.columns:
                            display_cols.append('centuries')
                        if 'fifties' in selected_data.columns:
                            display_cols.append('fifties')
                        if 'highest_score' in selected_data.columns:
                            display_cols.append('highest_score')
                        if 'batting_average' in selected_data.columns:
                            display_cols.append('batting_average')
                        if 'ducks' in selected_data.columns:
                            display_cols.append('ducks')
                            
                        st.dataframe(
                            selected_data[display_cols].set_index('batsman'),
                            use_container_width=True
                        )
                    
                    # Performance scatter plot
                    fig = px.scatter(selected_data, x='matches', y='runs', size='strike_rate',
                                    color='batsman', 
                                    hover_data=['strike_rate'],
                                    title='Selected Players: Experience vs Performance')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Select players from the dropdown to compare")
                    
            else:  # Bowling
                st.markdown('<h2 class="section-header bowling-header">Player Search and Comparison</h2>', unsafe_allow_html=True)
                
                # Player search
                player_names = filtered_df['bowler'].unique()
                    
                selected_players = st.multiselect('select players to compare:', player_names, default=player_names[:3] if len(player_names) > 0 else [])
                
                if selected_players:
                    selected_data = filtered_df[filtered_df['bowler'].isin(selected_players)].copy()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Radar chart for comparison
                        metrics = ['wickets', 'economy', 'matches']
                        
                        # Only add these metrics if they exist in the data
                        available_metrics = [m for m in metrics if m in selected_data.columns]
                        
                        # Add additional metrics if they exist
                        if 'bowling_average' in selected_data.columns:
                            available_metrics.append('bowling_average')
                        if 'bowling_strike_rate' in selected_data.columns:
                            available_metrics.append('bowling_strike_rate')
                        if 'four_wickets' in selected_data.columns:
                            available_metrics.append('four_wickets')
                        if 'five_wickets' in selected_data.columns:
                            available_metrics.append('five_wickets')
                            
                        fig = go.Figure()
                        
                        for _, player in selected_data.iterrows():
                            values = [player[metric] for metric in available_metrics]
                            # Normalize values for radar chart (inverse for economy and average)
                            max_vals = safe_max_calculation(bowling_data[selected_format], available_metrics)
                            normalized_values = [values[i] / max_vals[metric] for i, metric in enumerate(available_metrics)]
                            
                            # For economy and average, lower is better, so we invert the normalization
                            if 'economy' in available_metrics:
                                idx = available_metrics.index('economy')
                                normalized_values[idx] = 1 - normalized_values[idx]
                            if 'bowling_average' in available_metrics:
                                idx = available_metrics.index('bowling_average')
                                normalized_values[idx] = 1 - normalized_values[idx]
                            
                            fig.add_trace(go.Scatterpolar(
                                r=normalized_values,
                                theta=available_metrics,
                                fill='toself',
                                name=player['bowler']
                            ))
                        
                        fig.update_layout(
                            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                            title='Player Comparison (Normalized Metrics)',
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Detailed player stats
                        display_cols = ['bowler', 'matches', 'wickets', 'economy']
                        if 'innings' in selected_data.columns:
                            display_cols.append('innings')
                        if 'runs_conceded' in selected_data.columns:
                            display_cols.append('runs_conceded')
                        if 'bowling_average' in selected_data.columns:
                            display_cols.append('bowling_average')
                        if 'bowling_strike_rate' in selected_data.columns:
                            display_cols.append('bowling_strike_rate')
                        if 'four_wickets' in selected_data.columns:
                            display_cols.append('four_wickets')
                        if 'five_wickets' in selected_data.columns:
                            display_cols.append('five_wickets')
                            
                        st.dataframe(
                            selected_data[display_cols].set_index('bowler'),
                            use_container_width=True
                        )
                    
                    # Performance scatter plot
                    fig = px.scatter(selected_data, x='matches', y='wickets', size='economy',
                                    color='bowler', 
                                    hover_data=['economy'],
                                    title='Selected Players: Experience vs Wickets')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Select players from the dropdown to compare")
    
    with tab6 if stats_type != "Fielding" else tab5:  # Fielding has one less tab
        if stats_type == "Batting":
            st.markdown('<h2 class="section-header">Format Comparison</h2>', unsafe_allow_html=True)
            
            if len(batting_data) > 1:
                # Combine all format data
                combined_data = pd.concat(batting_data.values(), ignore_index=True)
                
                # Format comparison metrics
                format_stats = combined_data.groupby('format').agg({
                    'runs': ['mean', 'median', 'max'],
                    'strike_rate': ['mean', 'median'],
                    'matches': ['mean', 'median'],
                    'batsman': 'count'
                }).round(1)
                
                # Add century stats if available
                if 'centuries' in combined_data.columns:
                    format_stats[('centuries', 'mean')] = combined_data.groupby('format')['centuries'].mean()
                if 'fifties' in combined_data.columns:
                    format_stats[('fifties', 'mean')] = combined_data.groupby('format')['fifties'].mean()
                
                format_stats.columns = ['_'.join(col).strip() for col in format_stats.columns.values]
                format_stats = format_stats.reset_index()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Average runs by format
                    fig = px.bar(format_stats, x='format', y='runs_mean', 
                                title='Average Runs by Format', color='format',
                                labels={'runs_mean': 'Average Runs', 'format': 'Format'})
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Average strike rate by format
                    fig = px.bar(format_stats, x='format',y='runs_mean', 
                                title='Average Strike Rate by Format', color='format',
                                labels={'strike_rate_mean': 'Average Strike Rate', 'format': 'Format'})
                    st.plotly_chart(fig, use_container_width=True)
                
                col3, col4 = st.columns(2)
                
                with col3:
                    # Average centuries by format if available
                    if 'centuries_mean' in format_stats.columns:
                        fig = px.bar(format_stats, x='format',y='centuries_mean', 
                                    title='Average Centuries by Format', color='format',
                                    labels={'centuries_mean': 'Average Centuries', 'format': 'Format'})
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Century data not available for format comparison")
                
                with col4:
                    # Player count by format
                    fig = px.bar(format_stats,x='format', y='batsman_count', 
                                title='Number of Players by Format', color='format',
                                labels={'batsman_count': 'Number of Players', 'format': 'Format'})
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show detailed comparison table
                st.markdown('</h3>Detailed Format Statistics</h3>', unsafe_allow_html=True)
                st.dataframe(format_stats, use_container_width=True)
            else:
                st.info("Need data for multiple formats to enable comparison")
                
        elif stats_type == "Bowling":
            st.markdown('<h2 class="section-header bowling-header">Format Comparison</h2>', unsafe_allow_html=True)
            
            if len(bowling_data) > 1:
                # Combine all format data
                combined_data = pd.concat(bowling_data.values(), ignore_index=True)
                
                # Format comparison metrics
                format_stats = combined_data.groupby('format').agg({
                    'wickets': ['mean', 'median', 'max'],
                    'economy': ['mean', 'median'],
                    'matches': ['mean', 'median'],
                    'bowler': 'count'
                }).round(2)
                
                # Add bowling metrics if available
                if 'bowling_average' in combined_data.columns:
                    format_stats[('bowling_average', 'mean')] = combined_data.groupby('format')['bowling_average'].mean()
                if 'bowling_strike_rate' in combined_data.columns:
                    format_stats[('bowling_strike_rate', 'mean')] = combined_data.groupby('format')['bowling_strike_rate'].mean()
                if 'five_wickets' in combined_data.columns:
                    format_stats[('five_wickets', 'mean')] = combined_data.groupby('format')['five_wickets'].mean()
                
                format_stats.columns = ['_'.join(col).strip() for col in format_stats.columns.values]
                format_stats = format_stats.reset_index()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Average wickets by format
                    fig = px.bar(format_stats, x='format', y='wickets_mean', 
                                title='Average Wickets by Format', color='format',
                                labels={'wickets_mean': 'Average Wickets', 'format': 'Format'})
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Average economy by format
                    fig = px.bar(format_stats, x='format', y='economy_mean', 
                                title='Average Economy by Format', color='format',
                                labels={'economy_mean': 'Average Economy', 'format': 'Format'})
                    st.plotly_chart(fig, use_container_width=True)
                
                col3, col4 = st.columns(2)
                
                with col3:
                    # Average bowling average by format if available
                    if 'bowling_average_mean' in format_stats.columns:
                        fig =px.bar(format_stats, x='format', y='bowling_average_mean', 
                                    title='Average Bowling Average by Format', color='format',
                                    labels={'bowling_average_mean': 'Average Bowling Average', 'format': 'Format'})
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Bowling average data not available for format comparison")
                
                with col4:
                    # Player count by format
                    fig = px.bar(format_stats, x='format', y='bowler_count', 
                                title='Number of Bowlers by Format', color='format',
                                labels={'bowler_count': 'Number of Bowlers', 'format': 'Format极'})
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show detailed comparison table
                st.markdown('<h3>Detailed Format Statistics</h3>', unsafe_allow_html=True)
                st.dataframe(format_stats, use_container_width=True)
            else:
                st.info("Need data for multiple formats to enable comparison")
                
        else:  # Fielding
            st.markdown('<h2 class="section-header">Format Comparison</h2>', unsafe_allow_html=True)
            
            if len(fielding_data) > 1:
                # Combine all format data
                combined_data = pd.concat(fielding_data.values(), ignore_index=True)
                
                # Format comparison metrics
                format_stats = combined_data.groupby('format').agg({
                    'dismissals': ['mean', 'median', 'max'],
                    'matches': ['mean', 'median'],
                    'player': 'count'
                }).round(2)
                
                # Add fielding metrics if available
                if 'catches' in combined_data.columns:
                    format_stats[('catches', 'mean')] = combined_data.groupby('format')['catches'].mean()
                if 'stumpings' in combined_data.columns:
                    format_stats[('stumpings', 'mean')] = combined_data.groupby('format')['stumpings'].mean()
                if 'dismissals_per_inning' in combined_data.columns:
                    format_stats[('dismissals_per_inning', 'mean')] = combined_data.groupby('format')['dismissals_per_inning'].mean()
                
                format_stats.columns = ['_'.join(col).strip() for col in format_stats.columns.values]
                format_stats = format_stats.reset_index()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Average dismissals by format
                    fig = px.bar(format_stats, x='format', y='dismissals_mean', 
                                title='Average Dismissals by Format', color='format',
                                labels={'dismissals_mean': 'Average Dismissals', 'format': 'Format'})
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Average dismissals per inning by format if available
                    if 'dismissals_per_inning_mean' in format_stats.columns:
                        fig = px.bar(format_stats, x='format', y='dismissals_per_inning_mean', 
                                    title='Average Dismissals per Inning by Format', color='format',
                                    labels={'dismissals_per_inning_mean': 'Average Dismissals per Inning', 'format': 'Format'})
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Dismissals per inning data not available for format comparison")
                
                col3, col4 = st.columns(2)
                
                with col3:
                    # Average catches by format if available
                    if 'catches_mean' in format_stats.columns:
                        fig =px.bar(format_stats, x='format', y='catches_mean', 
                                    title='Average Catches by Format', color='format',
                                    labels={'catches_mean': 'Average Catches', 'format': 'Format'})
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Catches data not available for format comparison")
                
                with col4:
                    # Player count by format
                    fig = px.bar(format_stats, x='format', y='player_count', 
                                title='Number of Fielders by Format', color='format',
                                labels={'player_count': 'Number of Fielders', 'format': 'Format'})
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show detailed comparison table
                st.markdown('<h3>Detailed Format Statistics</h3>', unsafe_allow_html=True)
                st.dataframe(format_stats, use_container_width=True)
            else:
                st.info("Need data for multiple formats to enable comparison")

# The rest of the code remains the same as before for other tabs...
elif main_tab == "Player vs Player":
    st.markdown('<h1 class="main-header">Player vs Player Comparison</h1>', unsafe_allow_html=True)
    
    # Use the globally selected format
    comparison_format = selected_format
    
    # Get available players from the selected format
    batting_players = batting_data[comparison_format]['batsman'].unique() if comparison_format in batting_data else []
    bowling_players = bowling_data[comparison_format]['bowler'].unique() if comparison_format in bowling_data else []
    
    st.markdown('<h3 class="section-header">Select Batsmen</h3>', unsafe_allow_html=True)
    selected_batsmen = st.multiselect("Choose batsmen to compare:", batting_players, default=batting_players[:2] if len(batting_players) > 0 else [])
    
    
    if selected_batsmen:
        st.markdown('<h3 class="section-header">Batsmen Comparison</h3>', unsafe_allow_html=True)
        
        # Get batsmen data
        batsmen_data = batting_data[comparison_format][batting_data[comparison_format]['batsman'].isin(selected_batsmen)].copy()
        
        if not batsmen_data.empty:
            # Display comparison table
            comparison_cols = ['batsman', 'matches', 'innings', 'runs', 'strike_rate']
            if 'balls_faced' in batsmen_data.columns:
                comparison_cols.append('balls_faced')
            if 'fours' in batsmen_data.columns:
                comparison_cols.append('fours')
            if 'sixes' in batsmen_data.columns:
                comparison_cols.append('sixes')
            if 'dismissals' in batsmen_data.columns:
                comparison_cols.append('dismissals')
                
            st.dataframe(batsmen_data[comparison_cols].set_index('batsman'), use_container_width=True)
            
            # Create radar chart for comparison
            metrics = ['runs', 'strike_rate', 'matches']
            if 'fours' in batsmen_data.columns:
                metrics.append('fours')
            if 'sixes' in batsmen_data.columns:
                metrics.append('sixes')
                
            fig = go.Figure()
            
            for _, player in batsmen_data.iterrows():
                values = [player[metric] for metric in metrics]
                # Normalize values for radar chart
                max_vals = safe_max_calculation(batting_data[comparison_format], metrics)
                normalized_values = [values[i] / max_vals[metric] for i, metric in enumerate(metrics)]
                
                fig.add_trace(go.Scatterpolar(
                    r=normalized_values,
                    theta=metrics,
                    fill='toself',
                    name=player['batsman']
                ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title='Batsmen Comparison (Normalized Metrics)',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for selected batsmen")
    st.markdown('<h3 class="bowling-header">Select Bowlers</h3>', unsafe_allow_html=True)
    selected_bowlers = st.multiselect("Choose bowlers to compare:", bowling_players, default=bowling_players[:2] if len(bowling_players) > 0 else [])
    
    
    if selected_bowlers:
        st.markdown('<h3 class="bowling-header">Bowlers Comparison</h3>', unsafe_allow_html=True)
        
        # Get bowlers data
        bowlers_data = bowling_data[comparison_format][bowling_data[comparison_format]['bowler'].isin(selected_bowlers)].copy()
        
        if not bowlers_data.empty:
            # Display comparison table
            comparison_cols = ['bowler', 'matches', 'wickets', 'economy']
            if 'runs_conceded' in bowlers_data.columns:
                comparison_cols.append('runs_conceded')
            if 'bowling_average' in bowlers_data.columns:
                comparison_cols.append('bowling_average')
            if 'bowling_strike_rate' in bowlers_data.columns:
                comparison_cols.append('bowling_strike_rate')
            if 'maidens' in bowlers_data.columns:
                comparison_cols.append('maidens')
                
            st.dataframe(bowlers_data[comparison_cols].set_index('bowler'), use_container_width=True)
            
            # Create radar chart for comparison
            metrics = ['wickets', 'economy', 'matches']
            if 'bowling_average' in bowlers_data.columns:
                metrics.append('bowling_average')
            if 'bowling_strike_rate' in bowlers_data.columns:
                metrics.append('bowling_strike_rate')
                
            fig = go.Figure()
            
            for _, player in bowlers_data.iterrows():
                values = [player[metric] for metric in metrics]
                # Normalize values for radar chart (inverse for economy and average)
                max_vals = safe_max_calculation(bowling_data[comparison_format], metrics)
                normalized_values = [values[i] / max_vals[metric] for i, metric in enumerate(metrics)]
                
                # For economy and average, lower is better, so we invert the normalization
                if 'economy' in metrics:
                    idx = metrics.index('economy')
                    normalized_values[idx] = 1 - normalized_values[idx]
                if 'bowling_average' in metrics:
                    idx = metrics.index('bowling_average')
                    normalized_values[idx] = 1 - normalized_values[idx]
                
                fig.add_trace(go.Scatterpolar(
                    r=normalized_values,
                    theta=metrics,
                    fill='toself',
                    name=player['bowler']
                ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title='Bowlers Comparison (Normalized Metrics)',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for selected bowlers")

elif main_tab == "Batsman vs Bowler":
    st.markdown('<h1 class="main-header">Batsman vs Bowler Analysis</h1>', unsafe_allow_html=True)
    
    if batsman_vs_bowler_data:
        # Use the globally selected format
        bvb_format = selected_format
        
        if bvb_format in batsman_vs_bowler_data:
            # Get available players from the selected format
            available_batsmen = batsman_vs_bowler_data[bvb_format]['batsman'].unique()
            available_bowlers = batsman_vs_bowler_data[bvb_format]['bowler'].unique()
            
            col1, col2 = st.columns(2)
            
            with col1:
                selected_batsman = st.selectbox("Select Batsman", available_batsmen)
            
            with col2:
                selected_bowler = st.selectbox("Select Bowler", available_bowlers)
            
            # Filter data for the selected matchup
            matchup_data = batsman_vs_bowler_data[bvb_format][
                (batsman_vs_bowler_data[bvb_format]['batsman'] == selected_batsman) & 
                (batsman_vs_bowler_data[bvb_format]['bowler'] == selected_bowler)
            ]
            
            if not matchup_data.empty:
                st.markdown(f'<h3 class="section-header">{selected_batsman} vs {selected_bowler}</h3>', unsafe_allow_html=True)
                
                # Display matchup stats
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown('<div class="metric-card player-vs-player">', unsafe_allow_html=True)
                    st.metric("Runs", int(matchup_data['runs'].iloc[0]))
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-card player-vs-player">', unsafe_allow_html=True)
                    st.metric("Balls Faced", int(matchup_data['balls'].iloc[0]))
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="metric-card player-vs-player">', unsafe_allow_html=True)
                    st.metric("Strike Rate", f"{float(matchup_data['strike_rate'].iloc[0]):.2f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col4:
                    st.markdown('<div class="metric-card player-vs-player">', unsafe_allow_html=True)
                    dismissals = int(matchup_data['dismissals'].iloc[0])
                    st.metric("Dismissals", dismissals)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Calculate average
                average = float(matchup_data['runs'].iloc[0]) / dismissals if dismissals > 0 else float('inf')
                
                col5, col6 = st.columns(2)
                
                with col5:
                    st.markdown('<div class="metric-card player-vs-player">', unsafe_allow_html=True)
                    st.metric("Average", f"{average:.2f}" if dismissals > 0 else "N/A")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col6:
                    st.markdown('<div class="metric-card player-vs-player">', unsafe_allow_html=True)
                    dominance = "Batsman" if average > 30 or float(matchup_data['strike_rate'].iloc[0]) > 120 else "Bowler" if dismissals > 0 and average < 20 else "Balanced"
                    st.metric("Dominance", dominance)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Show detailed data
                st.dataframe(matchup_data, use_container_width=True)
                
            else:
                st.info(f"No historical data available for {selected_batsman} vs {selected_bowler} in {bvb_format} format")
                
                # Show individual player stats instead
                col1, col2 = st.columns(2)
                
                with col1:
                    if bvb_format in batting_data:
                        batsman_stats = batting_data[bvb_format][batting_data[bvb_format]['batsman'] == selected_batsman]
                        if not batsman_stats.empty:
                            st.markdown(f'<h4 class="section-header">{selected_batsman} Overall Stats</h4>', unsafe_allow_html=True)
                            st.dataframe(batsman_stats, use_container_width=True)
                
                with col2:
                    if bvb_format in bowling_data:
                        bowler_stats = bowling_data[bvb_format][bowling_data[bvb_format]['bowler'] == selected_bowler]
                        if not bowler_stats.empty:
                            st.markdown(f'<h4 class="bowling-header">{selected_bowler} Overall Stats</h4>', unsafe_allow_html=True)
                            st.dataframe(bowler_stats, use_container_width=True)
        else:
            st.warning(f"No batsman vs bowler data available for {bvb_format} format")
    else:
        st.warning("Batsman vs Bowler data not available. Please make sure the CSV files exist.")

elif main_tab == "Team Selection Assistant":
    st.markdown('<h1 class="main-header">Team Selection Assistant</h1>', unsafe_allow_html=True)
    
    # Add model info
    if selected_format in model_predictor.xi_models:
        st.info("✅ Using trained machine learning model for team selection")
    else:
        st.warning("⚠️ Using fallback logic (ML model not available)")
    
    if ball_data:
        # Use the globally selected format
        team_format = selected_format
        
        if team_format in ball_data:
            # Get available teams from ball-by-ball data
            available_columns = ball_data[team_format].columns
            
            # Check if team1 and team2 columns exist
            if 'team1' in available_columns and 'team2' in available_columns:
                teams = list(ball_data[team_format]['team1'].unique()) + list(ball_data[team_format]['team2'].unique())
            elif 'batting_team' in available_columns and 'bowling_team' in available_columns:
                teams = list(ball_data[team_format]['batting_team'].unique()) + list(ball_data[team_format]['bowling_team'].unique())
            else:
                # Fallback: try to find any team-related columns
                team_columns = [col for col in available_columns if 'team' in col.lower()]
                if team_columns:
                    teams = []
                    for col in team_columns:
                        teams += list(ball_data[team_format][col].unique())
                else:
                    st.error("Could not find team information in the data. Available columns: " + str(available_columns))
                    st.stop()
            
            teams = [t for t in teams if pd.notna(t)]
            teams = sorted(set(teams))
            
            if not teams:
                st.warning("No teams found in the data")
                st.stop()
            
            selected_team = st.selectbox("Select Team", teams)
            
            # Get unique opposition teams
            opposition_teams = [t for t in teams if t != selected_team]
            selected_opposition = st.selectbox("Select Opposition Team", opposition_teams)
            
            # Get available venues
            venues = ball_data[team_format]['venue'].unique()
            venues = [v for v in venues if pd.notna(v)]
            selected_venue = st.selectbox("Select Venue", venues)
            
            # Pitch conditions
            pitch_conditions = st.selectbox("Select Pitch Conditions", 
                                          ['Batting Friendly', 'Bowling Friendly', 'Balanced', 'Spinning', 'Seaming'])
            
            if st.button("Generate Optimal Playing XI"):
                with st.spinner("Analyzing player form and match conditions..."):
                    recommended_xi = recommend_playing_xi(selected_team, selected_opposition, selected_venue, pitch_conditions, team_format)
                    
                    if not recommended_xi.empty:
                        st.success(f"Recommended Playing XI for {selected_team} vs {selected_opposition}")
                        
                        # Display the team
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Batting Lineup**")
                            batsmen = recommended_xi[recommended_xi['role'].isin(['batsman', 'all_rounder'])]
                            for i, (_, player) in enumerate(batsmen.iterrows(), 1):
                                st.write(f"{i}. {player['player']} ({player['role']}) - Score: {player['composite_score']:.1f}")
                        
                        with col2:
                            st.markdown("**Bowling Attack**")
                            bowlers = recommended_xi[recommended_xi['role'].isin(['bowler', 'all_rounder'])]
                            for i, (_, player) in enumerate(bowlers.iterrows(), 1):
                                st.write(f"{i}. {player['player']} ({player['role']}) - Score: {player['composite_score']:.1f}")
                        
                        # Show detailed analysis
                        with st.expander("View Detailed Player Analysis"):
                            for _, player in recommended_xi.iterrows():
                                st.markdown(f"### {player['player']} ({player['role']})")
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Form", f"{player['form']:.1f}/100")
                                with col2:
                                    st.metric("vs Opponent", f"{player['opponent_performance']:.1f}/100")
                                with col3:
                                    st.metric("At Venue", f"{player['venue_performance']:.1f}/100")
                    else:
                        st.warning("Could not generate playing XI recommendation. Please check if there's enough data available.")
        else:
            st.warning(f"No ball-by-ball data available for {team_format} format")
    else:
        st.warning("Ball-by-ball data not available. Please make sure the CSV files exist.")

elif main_tab == "Players by Team":
    st.markdown('<h1 class="main-header">Players by Team</h1>', unsafe_allow_html=True)
    
    if ball_data:
        # Use the globally selected format
        team_format = selected_format
        
        if team_format in ball_data:
            # Get available teams from ball-by-ball data
            available_columns = ball_data[team_format].columns
            
            # Check if team1 and team2 columns exist
            if 'team1' in available_columns and 'team2' in available_columns:
                teams = list(ball_data[team_format]['team1'].unique()) + list(ball_data[team_format]['team2'].unique())
            elif 'batting_team' in available_columns and 'bowling_team' in available_columns:
                teams = list(ball_data[team_format]['batting_team'].unique()) + list(ball_data[team_format]['bowling_team'].unique())
            else:
                # Fallback: try to find any team-related columns
                team_columns = [col for col in available_columns if 'team1' in col.lower()]
                if team_columns:
                    teams = []
                    for col in team_columns:
                        teams += list(ball_data[team_format][col].unique())
                else:
                    st.error("Could not find team information in the data. Available columns: " + str(available_columns))
                    st.stop()
            
            teams = [t for t in teams if pd.notna(t)]
            teams = sorted(set(teams))
            
            if not teams:
                st.warning("No teams found in the data")
                st.stop()
            
            selected_team = st.selectbox("Select Team", teams)
            
            # Better approach: Find players who belong to this team by analyzing match squads
            team_players = []
            
            # Get unique matches for this team
            team_matches = ball_data[team_format][
                (ball_data[team_format]['team1'] == selected_team) | 
                (ball_data[team_format]['team2'] == selected_team)
            ]
            
            # For each match, find players who were in the squad for this team
            for match_id in team_matches['match_id'].unique():
                match_data = team_matches[team_matches['match_id'] == match_id]
                
                # Determine if the team was team1 or team2 in this match
                if match_data['team1'].iloc[0] == selected_team:
                    # Team was team1, so get their batsmen and bowlers
                    team_batsmen = match_data[match_data['batting_team'] == selected_team]['batsman'].unique()
                    team_bowlers = match_data[match_data['team2'] == selected_team]['bowler'].unique()
                else:
                    # Team was team2, so get their batsmen and bowlers
                    team_batsmen = match_data[match_data['batting_team'] == selected_team]['batsman'].unique()
                    team_bowlers = match_data[match_data['team1'] == selected_team]['bowler'].unique()
                
                team_players.extend(team_batsmen)
                team_players.extend(team_bowlers)
            
            # Remove duplicates and NaN values
            team_players = list(set([p for p in team_players if pd.notna(p)]))
            
            if not team_players:
                st.warning(f"No players found for {selected_team}")
                st.stop()
            
            st.markdown(f'<h3 class="team-analysis">{selected_team} Players</h3>', unsafe_allow_html=True)
            
            # Display players
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<h4 class="section-header">Batsmen</h4>', unsafe_allow_html=True)
                if team_format in batting_data:
                    # Get batting stats only for players in this team
                    batsmen_stats = batting_data[team_format][batting_data[team_format]['batsman'].isin(team_players)]
                    
                    if not batsmen_stats.empty:
                        st.dataframe(batsmen_stats[['batsman', 'matches', 'runs', 'strike_rate']].set_index('batsman'), use_container_width=True)
                    else:
                        st.info("No batting data available for this team")
                else:
                    st.info("Batting data not available for this format")
            
            with col2:
                st.markdown('<h4 class="bowling-header">Bowlers</h4>', unsafe_allow_html=True)
                if team_format in bowling_data:
                    # Get bowling stats only for players in this team
                    bowlers_stats = bowling_data[team_format][bowling_data[team_format]['bowler'].isin(team_players)]
                    
                    if not bowlers_stats.empty:
                        st.dataframe(bowlers_stats[['bowler', 'matches', 'wickets', 'economy']].set_index('bowler'), use_container_width=True)
                    else:
                        st.info("No bowling data available for this team")
                else:
                    st.info("Bowling data not available for this format")
            
            # Show player list
            st.markdown('<h4 class="team-analysis">Full Squad</h4>', unsafe_allow_html=True)
            
            # Create a DataFrame for better display
            squad_data = []
            for player in team_players:
                # Check if player is in batting or bowling data
                is_batsman = player in ball_data[team_format]['batsman'].values
                is_bowler = player in ball_data[team_format]['bowler'].values
                
                role = ""
                if is_batsman and is_bowler:
                    role = "All-rounder"
                elif is_batsman:
                    role = "Batsman"
                elif is_bowler:
                    role = "Bowler"
                
                squad_data.append({'Player': player, 'Role': role})
            
            squad_df = pd.DataFrame(squad_data)
            
            # Display in a nice format
            col_count = 3
            cols = st.columns(col_count)
            
            for i, row in squad_df.iterrows():
                with cols[i % col_count]:
                    st.markdown(f"**{row['Player']}** ({row['Role']})")
            
            # Also show the full list as a dataframe
            with st.expander("View Full Squad as Table"):
                st.dataframe(squad_df, use_container_width=True)
        else:
            st.warning(f"No ball-by-ball data available for {team_format} format")
    else:
        st.warning("Ball-by-ball data not available. Please make sure the CSV files exist.")
elif main_tab == "Match Analysis":
    st.markdown('<h1 class="main-header">Match Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    if match_data:
        # Use the globally selected format
        match_format = selected_format
        
        if match_format in match_data:
            df = match_data[match_format]
            
            # Basic stats
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Total Matches", len(df))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Average Runs per Match", f"{df['total_runs'].mean():.0f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Average Wickets per Match", f"{df['total_wickets'].mean():.1f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Matches with Result", f"{len(df[df['winner'].notna()])}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Tabs for different analyses
            tab1, tab2, tab3, tab4 = st.tabs([
                "🏟️ Venue Analysis", 
                "📅 Timeline Analysis", 
                "🏏 Team Performance", 
                "🎯 Toss Impact"
            ])
            
            with tab1:
                st.markdown('<h2 class="section-header">Venue Analysis</h2>', unsafe_allow_html=True)
                
                # Venue statistics
                venue_stats = df.groupby('venue').agg({
                    'total_runs': ['mean', 'count'],
                    'total_wickets': 'mean',
                    'winner': lambda x: x.notna().sum()
                }).round(1)
                
                venue_stats.columns = ['avg_runs', 'matches_played', 'avg_wickets', 'completed_matches']
                venue_stats = venue_stats.reset_index()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Top venues by matches
                    top_venues = venue_stats.nlargest(10, 'matches_played')
                    fig = px.bar(top_venues, x='matches_played', y='venue', orientation='h',
                                title='Top Venues by Matches Played', color='matches_played')
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Highest scoring venues
                    min_matches = 3  # Minimum matches to qualify
                    qualified_venues = venue_stats[venue_stats['matches_played'] >= min_matches]
                    if not qualified_venues.empty:
                        high_scoring = qualified_venues.nlargest(10, 'avg_runs')
                        fig = px.bar(high_scoring, x='avg_runs', y='venue', orientation='h',
                                    title=f'Highest Scoring Venues (Min {min_matches} matches)', color='avg_runs')
                        fig.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(f"No venues with {min_matches}+ matches found")
                
                # Venue details expander
                with st.expander("View Detailed Venue Statistics"):
                    st.dataframe(venue_stats, use_container_width=True)
            
            with tab2:
                st.markdown('<h2 class="section-header">Timeline Analysis</h2>', unsafe_allow_html=True)
                
                # Extract year from date
                df['year'] = df['date'].dt.year
                
                # Year statistics
                yearly_stats =df.groupby('year').agg({
                    'total_runs': ['mean', 'sum'],
                    'total_wickets': 'mean',
                    'match_id': 'count'
                }).round(1)
                
                yearly_stats.columns = ['avg_runs', 'total_runs', 'avg_wickets', 'matches_played']
                yearly_stats = yearly_stats.reset_index()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Matches per year
                    fig =px.bar(yearly_stats, x='year', y='matches_played',
                                title='Matches Played per Year', color='matches_played')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Average runs per year
                    fig = px.line(yearly_stats, x='year', y='avg_runs',
                                title='Average Runs per Match Over Time', markers=True)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Run rate analysis if we have balls and runs
                if 'total_balls' in df.columns and 'total_runs' in df.columns:
                    df['run_rate'] = (df['total_runs'] / df['total_balls']) * 6
                    yearly_run_rate = df.groupby('year')['run_rate'].mean().reset_index()
                    
                    fig = px.line(yearly_run_rate, x='year', y='run_rate',
                                title='Average Run Rate Over Time', markers=True)
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.markdown('<h2 class="section-header">Team Performance Analysis</h2>', unsafe_allow_html=True)
                
                # Get all unique teams
                all_teams = pd.concat([df['team1'], df['team2']]).unique()
                selected_team = st.selectbox("Select Team", all_teams)
                
                # Filter matches for the selected team
                team_matches = df[(df['team1'] == selected_team) | (df['team2'] == selected_team)]
                
                # Calculate win/loss/draw stats
                team_matches['result'] = team_matches.apply(
                    lambda x: 'Win' if x['winner'] == selected_team else 
                             'Loss' if pd.notna(x['winner']) and x['winner'] != selected_team else 
                             'Draw/Tie', axis=1
                )
                
                result_stats = team_matches['result'].value_counts().reset_index()
                result_stats.columns = ['result', 'count']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Results pie chart
                    fig = px.pie(result_stats, values='count', names='result',
                                title=f'{selected_team} Match Results')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                # Team performance by venue
                    venue_performance = team_matches.groupby(['venue', 'result']).size().unstack(fill_value=0).reset_index()
    
                # Only sum numeric columns (exclude the 'venue' string column)
                    numeric_cols = venue_performance.select_dtypes(include=[np.number]).columns
                    venue_performance['total_matches'] = venue_performance[numeric_cols].sum(axis=1)
    
                # Get top venues
                    top_venues = venue_performance.nlargest(5, 'total_matches')
    
                    if not top_venues.empty:
                        fig = px.bar(top_venues, x='venue', y=['Win', 'Loss', 'Draw/Tie'],
                        title=f'{selected_team} Performance at Top Venues',
                        labels={'value': 'Matches', 'variable': 'Result'})
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Not enough venue data available")
                
                # Team batting performance over time
                team_matches['team_runs'] = team_matches.apply(
                    lambda x: x['total_runs']/2 if pd.isna(x['winner']) else  # For draws, approximate
                    x['total_runs'] if (x['winner'] == selected_team and x['team1'] == selected_team) or 
                                      (x['winner'] != selected_team and x['team2'] == selected_team) else
                    x['total_runs']/2,  # Simplified approach
                    axis=1
                )
                
                yearly_team_stats = team_matches.groupby('year').agg({
                    'team_runs': 'mean',
                    'match_id': 'count'
                }).reset_index()
                yearly_team_stats.columns = ['year', 'avg_runs', 'matches_played']
                
                fig = px.line(yearly_team_stats, x='year', y='avg_runs',
                            title=f'{selected_team} Average Runs per Match Over Time', markers=True)
                st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                st.markdown('<h2 class="section-header">Toss Impact Analysis</h2>', unsafe_allow_html=True)
                
                # Calculate toss impact
                df['toss_impact'] = df.apply(
                    lambda x: 'Won Toss & Match' if x['toss_winner'] == x['winner'] else
                             'Lost Toss & Match' if pd.notna(x['winner']) and x['toss_winner'] != x['winner'] else
                             'No Result/Unknown',
                    axis=1
                )
                
                # Toss decision impact
                toss_stats = df[df['toss_impact'] != 'No Result/Unknown']['toss_impact'].value_counts().reset_index()
                toss_stats.columns = ['result', 'count']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Toss impact pie chart
                    fig =px.pie(toss_stats, values='count', names='result',
                                title='Impact of Winning the Toss')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Toss decision analysis
                    if 'toss_decision' in df.columns:
                        decision_stats = df['toss_decision'].value_counts().reset_index()
                        decision_stats.columns = ['decision', 'count']
                        
                        fig =px.pie(decision_stats, values='count', names='decision',
                                    title='Toss Decisions Distribution')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Toss decision data not available")
                
                # Toss impact by venue
                venue_toss_impact = df.groupby(['venue', 'toss_impact']).size().unstack(fill_value=0).reset_index()
                venue_toss_impact['win_percentage'] = (venue_toss_impact.get('Won Toss & Match', 0) / 
                                                     (venue_toss_impact.get('Won Toss & Match', 0) + 
                                                      venue_toss_impact.get('Lost Toss & Match', 0))) * 100
                
                # Filter venues with sufficient data
                min_matches = 5
                qualified_venues = venue_toss_impact[
                    (venue_toss_impact.get('Won Toss & Match', 0) + 
                     venue_toss_impact.get('Lost Toss & Match', 0)) >= min_matches
                ]
                
                if not qualified_venues.empty:
                    fig = px.bar(qualified_venues.nlargest(10, 'win_percentage'), 
                                x='venue', y='win_percentage',
                                title='Venues Where Toss Has Highest Impact (Win %)',
                                labels={'win_percentage': 'Win Percentage when Winning Toss'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No venues with {min_matches}+ decisive matches found")
        else:
            st.warning(f"No match data available for {match_format} format")
    else:
        st.warning("Match data not available. Please make sure the CSV files exist.")

elif main_tab == "Ball-by-Ball Analysis":
    st.markdown('<h1 class="main-header">Ball-by-Ball Analysis</h1>', unsafe_allow_html=True)
    
    if ball_data:
        # Use the globally selected format
        ball_format = selected_format
        
        if ball_format in ball_data:
            # Get available matches
            matches = ball_data[ball_format]['match_id'].unique()
            selected_match = st.selectbox("Select Match", matches)
            
            # Get match details
            match_details = ball_data[ball_format][ball_data[ball_format]['match_id'] == selected_match].iloc[0]
            
            st.markdown(f'<h3 class="section-header">{match_details.get("team1", "Team 1")} vs {match_details.get("team2", "Team 2")} at {match_details.get("venue", "Unknown Venue")}</h3>', unsafe_allow_html=True)
            
            # Analysis type selection
            analysis_type = st.selectbox("Select Analysis Type", 
                                       ["Over-by-Over", "Partnerships", "Bowler Spells", "Phase Analysis"])
            
            if analysis_type == "Over-by-Over":
                st.markdown('<h4 class="section-header">Runs per Over Analysis</h4>', unsafe_allow_html=True)
                
                over_stats = over_by_over_analysis(selected_match, ball_format)
                if over_stats is not None:
                    fig = px.line(over_stats, x='over', y='runs_total', color='inning',
                                 title='Runs Scored per Over by Innings', markers=True)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No data available for this match")
            
            elif analysis_type == "Partnerships":
                st.markdown('<h4 class="section-header">Partnership Analysis</h4>', unsafe_allow_html=True)
                
                inning = st.selectbox("Select Inning", [1, 2])
                partnerships = partnership_analysis(selected_match, inning, ball_format)
                
                if not partnerships.empty:
                    # Display top partnerships
                    top_partnerships = partnerships.nlargest(5, 'runs')
                    st.dataframe(top_partnerships, use_container_width=True)
                    
                    # Partnership chart
                    fig = px.bar(top_partnerships, x='runs', y='batsman1', 
                                title='Top Partnerships', orientation='h',
                                hover_data=['batsman2', 'balls', 'start_over', 'end_over'])
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No partnership data available for this inning")
            
            elif analysis_type == "Bowler Spells":
                st.markdown('<h4 class="bowling-header">Bowler Spell Analysis</h4>', unsafe_allow_html=True)
                
                # Get bowlers from this match
                bowlers = ball_data[ball_format][ball_data[ball_format]['match_id'] == selected_match]['bowler'].unique()
                selected_bowler = st.selectbox("Select Bowler", bowlers)
                
                spell_data = bowler_spell_analysis(selected_match, selected_bowler, ball_format)
                
                if not spell_data.empty:
                    # Display spell data
                    st.dataframe(spell_data, use_container_width=True)
                    
                    # Spell performance chart
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    fig.add_trace(
                        go.Bar(x=spell_data['over'], y=spell_data['runs_total'], name="Runs Conceded"),
                        secondary_y=False,
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=spell_data['over'], y=spell_data['wicket_kind'], name="Wickets", mode='lines+markers'),
                        secondary_y=True,
                    )
                    
                    fig.update_layout(
                        title_text=f"{selected_bowler} Spell Analysis",
                        xaxis_title="Over",
                    )
                    
                    fig.update_yaxes(title_text="Runs Conceded", secondary_y=False)
                    fig.update_yaxes(title_text="Wickets", secondary_y=True)
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No spell data available for this bowler")
            
            elif analysis_type == "Phase Analysis":
                st.markdown('<h4 class="section-header">Phase-wise Performance</h4>', unsafe_allow_html=True)
                
                team = st.selectbox("Select Team", 
                                  [match_details.get('team1', 'Team 1'), match_details.get('team2', 'Team 2')])
                
                phase_data = phase_analysis(selected_match, team, ball_format)
                
                if not phase_data.empty:
                    # Display phase data
                    st.dataframe(phase_data, use_container_width=True)
                    
                    # Phase performance chart
                    fig = px.bar(phase_data,x='phase', y='total_runs', 
                                title=f'{team} Performance by Phase',
                                color='phase', 
                                hover_data=['avg_runs_per_over', 'wickets', 'balls'])
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No phase data available for this team")
        else:
            st.warning(f"No ball-by-ball data available for {ball_format} format")
    else:
        st.warning("Ball-by-ball data not available. Please make sure the CSV files exist.")

elif main_tab == "Trend Analysis":
    st.markdown('<h1 class="main-header">Performance Trend Analysis</h1>', unsafe_allow_html=True)
    
    if ball_data:
        # Use the globally selected format
        trend_format = selected_format
        
        if trend_format in ball_data:
            # Analysis type selection
            analysis_type = st.selectbox("Select Analysis Type", 
                                       ["Player Trends", "Team Trends"])
            
            if analysis_type == "Player Trends":
                st.markdown('<h3 class="section-header">Player Performance Trends</h3>', unsafe_allow_html=True)
                
                # Get available players
                players = list(ball_data[trend_format]['batsman'].unique()) + list(ball_data[trend_format]['bowler'].unique())
                players = [p for p in players if pd.notna(p)]
                players = sorted(set(players))
                
                selected_player = st.selectbox("Select Player", players)
                
                # Determine player role
                if selected_player in ball_data[trend_format]['batsman'].values:
                    role = 'batsman'
                else:
                    role = 'bowler'
                
                # Date range selection
                days_back = st.slider("Analysis Period (days)", 30, 365, 180)
                
                trend_data = player_performance_trends(selected_player, role, selected_format, days_back)
                
                if not trend_data.empty:
                    if role == 'batsman':
                        fig = px.line(trend_data, x='date', y='runs_ma_5', 
                                     title=f'{selected_player} Batting Form (5-match moving average)',
                                     markers=True)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        fig2 = px.line(trend_data, x='date', y='strike_rate_ma_5',
                                      title=f'{selected_player} Strike Rate Trend (5-match moving average)',
                                      markers=True)
                        st.plotly_chart(fig2, use_container_width=True)
                    else:
                        fig = px.line(trend_data, x='date', y='economy_ma_5',
                                     title=f'{selected_player} Economy Rate Trend (5-match moving average)',
                                     markers=True)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        fig2 = px.line(trend_data,x='date', y='wickets_ma_5',
                                      title=f'{selected_player} Wickets Trend (5-match moving average)',
                                      markers=True)
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    # Show recent form summary
                    recent_matches = trend_data.nlargest(5, 'date')
                    if role == 'batsman':
                        avg_runs = recent_matches['runs_batsman'].mean()
                        avg_sr = recent_matches['strike_rate'].mean()
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown('<div class="trend-card">', unsafe_allow_html=True)
                            st.metric("Recent Average Runs", f"{avg_runs:.1f}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        with col2:
                            st.markdown('<div class="trend-card">', unsafe_allow_html=True)
                            st.metric("Recent Strike Rate", f"{avg_sr:.1f}")
                            st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        avg_economy = recent_matches['economy'].mean()
                        avg_wickets = recent_matches['wicket_kind'].mean()
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown('<div class="trend-card">', unsafe_allow_html=True)
                            st.metric("Recent Economy", f"{avg_economy:.1f}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        with col2:
                            st.markdown('<div class="trend-card">', unsafe_allow_html=True)
                            st.metric("Recent Average Wickets", f"{avg_wickets:.1f}")
                            st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("No trend data available for this player")
            
            elif analysis_type == "Team Trends":
                st.markdown('<h3 class="team-analysis">Team Performance Trends</h3>', unsafe_allow_html=True)
                
                # Get available teams
                teams = list(ball_data[trend_format]['team1'].unique()) + list(ball_data[trend_format]['team2'].unique())
                teams = [t for t in teams if pd.notna(t)]
                teams = sorted(set(teams))
                
                selected_team = st.selectbox("Select Team", teams)
                
                # Date range selection
                days_back = st.slider("Analysis Period (days)", 30, 365, 180)
                
                # Get team matches
                team_matches = ball_data[trend_format][
                    (ball_data[trend_format]['team1'] == selected_team) | 
                    (ball_data[trend_format]['team2'] == selected_team)
                ]
                
                if team_matches.empty:
                    st.info("极No data available for this team")
                else:
                    # Get unique matches
                    match_results = team_matches.groupby('match_id').first().reset_index()
                    match_results['date'] = pd.to_datetime(match_results['date'])
                    match_results = match_results.sort_values('date')
                    
                    # Calculate win/loss
                    match_results['result'] = match_results.apply(
                        lambda x: 'Win' if x['winner'] == selected_team else 
                                 'Loss' if pd.notna(x['winner']) else 'Draw',
                        axis=1
                    )
                    
                    # Calculate moving win percentage
                    match_results['win'] = (match_results['result'] == '极Win').astype(int)
                    match_results['win_pct_5'] = match_results['win'].rolling(window=5, min_periods=1).mean() * 100
                    
                    # Filter by date
                    cutoff_date = pd.to_datetime('today') - timedelta(days=days_back)
                    recent_results = match_results[match_results['date'] > cutoff_date]
                    
                    if not recent_results.empty:
                        fig = px.line(recent_results,x='date', y='win_pct_5',
                                     title=f'{selected_team} Win Percentage Trend (5-match moving average)',
                                     markers=True)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Recent performance summary
                        recent_wins = recent_results['win'].sum()
                        recent_matches = len(recent_results)
                        win_pct = (recent_wins / recent_matches) * 100 if recent_matches > 0 else 0
                        
                        st.markdown('<div class="trend-card">', unsafe_allow_html=True)
                        st.metric("Recent Win Percentage", f"{win_pct:.1f}%")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.info("No recent data available for this team")
        else:
            st.warning(f"No ball-by-ball data available for {trend_format} format")
    else:
        st.warning("Ball-by-ball data not available. Please make sure the CSV files exist.")

elif main_tab == "Performance Predictor":
    st.markdown('<h1 class="main-header">Next Match Performance Predictor</h1>', unsafe_allow_html=True)
    
    # Add model info
    if selected_format in model_predictor.batting_models and selected_format in model_predictor.bowling_models:
        st.info("✅ Using trained machine learning models for predictions")
    else:
        st.warning("⚠️ Using fallback logic (ML models not available)")
    
    if ball_data:
        # Use the globally selected format
        predictor_format = selected_format
        
        if predictor_format in ball_data:
            # Get available players
            batsmen = list(ball_data[predictor_format]['batsman'].unique())
            batsmen = [p for p in batsmen if pd.notna(p)]
            batsmen = sorted(set(batsmen))
            
            bowlers = list(ball_data[predictor_format]['bowler'].unique())
            bowlers = [p for p in bowlers if pd.notna(p)]
            bowlers = sorted(set(bowlers))
            
            # Player type selection
            player_type = st.radio("Select Player Type", ["Batsman", "Bowler"])
            
            if player_type == "Batsman":
                selected_player = st.selectbox("Select Batsman", batsmen)
                role = 'batsman'
            else:
                selected_player = st.selectbox("Select Bowler", bowlers)
                role = 'bowler'
            
            # Match context
            st.markdown('<h3 class="section-header">Match Context</h3>', unsafe_allow_html=True)
            
            # Get available teams and venues
            teams = list(ball_data[predictor_format]['team1'].unique()) + list(ball_data[predictor_format]['team2'].unique())
            teams = [t for t in teams if pd.notna(t)]
            teams = sorted(set(teams))
            
            venues = ball_data[predictor_format]['venue'].unique()
            venues = [v for v in venues if pd.notna(v)]
            venues = sorted(venues)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                opponent = st.selectbox("Opponent Team", [t for t in teams if t != selected_player])
            
            with col2:
                venue = st.selectbox("Venue", venues)
            
            with col3:
                pitch_conditions = st.selectbox("Pitch Conditions", 
                                              ['Batting Friendly', 'Bowling Friendly', 'Balanced', 'Spinning', 'Seaming'])
            
            if st.button("Predict Performance"):
                prediction = predict_next_match_performance(selected_player, role, opponent, venue, pitch_conditions, predictor_format)
                
                if prediction:
                    if role == 'batsman':
                        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                        st.metric("Predicted Runs", f"{prediction['predicted_runs']}")
                        st.metric("Predicted Strike Rate", f"{prediction['predicted_strike_rate']}")
                        st.metric("Confidence Level", f"{prediction['confidence']*100:.1f}%")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                        st.metric("Predicted Wickets", f"{prediction['predicted_wickets']}")
                        st.metric("Predicted Economy", f"{prediction['predicted_economy']}")
                        st.metric("Confidence Level", f"{prediction['confidence']*100:.1f}%")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show factors affecting prediction
                    with st.expander("View Prediction Factors"):
                        # Get player form
                        form = calculate_player_form(selected_player, role, format_name=predictor_format) or 50
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Current Form", f"{form:.1f}/100")
                        
                        with col2:
                            # Get historical performance against opponent
                            if predictor_format in ball_data:
                                if role == 'batsman':
                                    opponent_matches = ball_data[predictor_format][
                                        (ball_data[predictor_format]['batsman'] == selected_player) &
                                        (ball_data[predictor_format]['team2'] == opponent)
                                    ]
                                else:
                                    opponent_matches = ball_data[predictor_format][
                                        (ball_data[predictor_format]['bowler'] == selected_player) &
                                        (ball_data[predictor_format]['batting_team'] == opponent)
                                    ]
                                
                                if not opponent_matches.empty:
                                    if role == 'batsman':
                                        runs = opponent_matches['runs_batsman'].sum()
                                        innings = len(opponent_matches['match_id'].unique())
                                        avg = runs / innings if innings > 0 else 0
                                        st.metric("Avg vs Opponent", f"{avg:.1f}")
                                    else:
                                        wickets = opponent_matches['wicket_kind'].notna().sum()
                                        innings = len(opponent_matches['match_id'].unique())
                                        avg = wickets / innings if innings > 0 else 0
                                        st.metric("Avg vs Opponent", f"{avg:.1f}")
                                else:
                                    st.metric("Avg vs Opponent", "No data")
                            else:
                                st.metric("Avg vs Opponent", "No data")
                        
                        with col3:
                            # Get historical performance at venue
                            if predictor_format in ball_data:
                                venue_matches = ball_data[predictor_format][
                                    ((ball_data[predictor_format]['batsman'] == selected_player) | 
                                     (ball_data[predictor_format]['bowler'] == selected_player)) &
                                    (ball_data[predictor_format]['venue'] == venue)
                                ]
                                
                                if not venue_matches.empty:
                                    if role == 'batsman':
                                        runs = venue_matches['runs_batsman'].sum()
                                        innings = len(venue_matches['match_id'].unique())
                                        avg = runs / innings if innings > 0 else 0
                                        st.metric("Avg at Venue", f"{avg:.1f}")
                                    else:
                                        wickets = venue_matches['wicket_kind'].notna().sum()
                                        innings = len(venue_matches['match_id'].unique())
                                        avg = wickets / innings if innings > 0 else 0
                                        st.metric("Avg at Venue", f"{avg:.1f}")
                                else:
                                    st.metric("Avg at Venue", "No data")
                            else:
                                st.metric("Avg at Venue", "No data")
                else:
                    st.error("Could not generate prediction. Please check if there's enough data available.")
        else:
            st.warning(f"No ball-by-ball data available for {predictor_format} format")
    else:
        st.warning("Ball-by-ball data not available. Please make sure the CSV files exist.")

# Footer
st.markdown("---")
st.markdown(f"**Advanced Cricket Statistics Dashboard** - Built with Streamlit | Multi-format analysis")

# Display raw data option
if st.sidebar.checkbox("Show raw data"):
    if main_tab == "Team & Player Stats":
        st.sidebar.dataframe(filtered_df)

# Download option
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

if main_tab == "Team & Player Stats":
    csv = convert_df_to_csv(filtered_df)
    st.sidebar.download_button(
        f"Download {selected_format} {stats_type.lower()} data as CSV",
        csv,
        f"{selected_format.lower()}_{stats_type.lower()}_stats.csv",
        "text/csv"
    )

# Data info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Data Info")

if main_tab == "Team & Player Stats":
    st.sidebar.info(f"**{selected_format} {stats_type} Data:** {len(filtered_df)} players (filtered)")

    if stats_type == "Batting":
        for format_name, data in batting_data.items():
            if format_name != selected_format:
                st.sidebar.info(f"**{format_name} Batting:** {len(data)} players")
    elif stats_type == "Bowling":
        for format_name, data in bowling_data.items():
            if format_name != selected_format:
                st.sidebar.info(f"**{format_name} Bowling:** {len(data)} players")
    else:
        for format_name, data in fielding_data.items():
            if format_name != selected_format:
                st.sidebar.info(f"**{format_name} Fielding:** {len(data)} players")
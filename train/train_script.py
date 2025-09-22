# train_script.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

def clean_dates(date_series):
    """Clean and convert date strings to datetime, handling invalid formats"""
    def parse_date(date_str):
        if pd.isna(date_str):
            return pd.NaT
        try:
            # Handle incomplete dates like "2021-06-"
            if isinstance(date_str, str) and date_str.endswith('-'):
                # Extract year and month, set day to 1
                parts = date_str.split('-')
                if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                    year = int(parts[0])
                    month = int(parts[1])
                    return pd.Timestamp(year=year, month=month, day=1)
            
            # Try standard parsing
            return pd.to_datetime(date_str, errors='coerce')
        except:
            return pd.NaT
    
    return pd.to_datetime(date_series.apply(parse_date), errors='coerce')

class PlayerPerformancePredictor:
    def __init__(self):
        self.batting_model = None
        self.bowling_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def prepare_batting_features(self, ball_data, match_data, player_name):
        """Prepare features for batting performance prediction"""
        player_matches = ball_data[ball_data['batsman'] == player_name]
        
        if player_matches.empty:
            return None
            
        features = []
        
        for match_id in player_matches['match_id'].unique():
            match_balls = player_matches[player_matches['match_id'] == match_id]
            
            # Get match info
            try:
                match_info = match_data[match_data['match_id'] == match_id].iloc[0]
                venue = match_info.get('venue', 'Unknown')
                
                # For batsman, opponent is the bowling team (opposite of batting_team)
                batting_team = match_balls['batting_team'].iloc[0]
                opponent = match_info['team2'] if batting_team == match_info['team1'] else match_info['team1']
                
                is_home = 1 if batting_team == match_info['team1'] else 0
                
            except (IndexError, KeyError):
                venue = 'Unknown'
                opponent = 'Unknown'
                is_home = 1
            
            # Basic match stats
            runs = match_balls['runs_batsman'].sum()
            balls_faced = len(match_balls)
            strike_rate = (runs / balls_faced * 100) if balls_faced > 0 else 0
            
            # Historical features (last 10 matches)
            if 'date' in ball_data.columns and not ball_data['date'].isna().all():
                match_date = match_balls['date'].iloc[0]
                if pd.notna(match_date):
                    prev_matches = player_matches[player_matches['date'] < match_date]
                else:
                    prev_matches = player_matches[player_matches['match_id'] != match_id]
            else:
                prev_matches = player_matches[player_matches['match_id'] != match_id]
                
            if len(prev_matches) > 0:
                last_10 = prev_matches.groupby('match_id').agg({
                    'runs_batsman': 'sum',
                    'ball_in_over': 'count'
                }).tail(10)
                
                avg_runs = last_10['runs_batsman'].mean()
                avg_sr = (last_10['runs_batsman'].sum() / last_10['ball_in_over'].sum() * 100) if last_10['ball_in_over'].sum() > 0 else 0
                form = avg_runs * 0.7 + avg_sr * 0.3
            else:
                avg_runs = 0
                avg_sr = 0
                form = 0
            
            features.append({
                'player': player_name,
                'match_id': match_id,
                'opponent': opponent,
                'venue': venue,
                'is_home': is_home,
                'prev_avg_runs': avg_runs,
                'prev_avg_sr': avg_sr,
                'form': form,
                'target_runs': runs
            })
        
        return pd.DataFrame(features)
    
    def prepare_bowling_features(self, ball_data, match_data, player_name):
        """Prepare features for bowling performance prediction"""
        player_matches = ball_data[ball_data['bowler'] == player_name]
        
        if player_matches.empty:
            return None
            
        features = []
        
        for match_id in player_matches['match_id'].unique():
            match_balls = player_matches[player_matches['match_id'] == match_id]
            
            # Get match info
            try:
                match_info = match_data[match_data['match_id'] == match_id].iloc[0]
                venue = match_info.get('venue', 'Unknown')
                
                # For bowler, opponent is the batting team
                opponent = match_balls['batting_team'].iloc[0]
                bowling_team = match_info['team2'] if opponent == match_info['team1'] else match_info['team1']
                is_home = 1 if bowling_team == match_info['team1'] else 0
                
            except (IndexError, KeyError):
                venue = 'Unknown'
                opponent = 'Unknown'
                is_home = 1
            
            # Basic match stats
            wickets = match_balls['wicket_kind'].notna().sum()
            runs_conceded = match_balls['runs_total'].sum()
            balls_bowled = len(match_balls)
            economy = (runs_conceded / balls_bowled * 6) if balls_bowled > 0 else 0
            
            # Historical features
            if 'date' in ball_data.columns and not ball_data['date'].isna().all():
                match_date = match_balls['date'].iloc[0]
                if pd.notna(match_date):
                    prev_matches = player_matches[player_matches['date'] < match_date]
                else:
                    prev_matches = player_matches[player_matches['match_id'] != match_id]
            else:
                prev_matches = player_matches[player_matches['match_id'] != match_id]
                
            if len(prev_matches) > 0:
                last_10 = prev_matches.groupby('match_id').agg({
                    'wicket_kind': lambda x: x.notna().sum(),
                    'runs_total': 'sum',
                    'ball_in_over': 'count'
                }).tail(10)
                
                avg_wickets = last_10['wicket_kind'].mean()
                avg_economy = (last_10['runs_total'].sum() / last_10['ball_in_over'].sum() * 6) if last_10['ball_in_over'].sum() > 0 else 0
                form = avg_wickets * 0.7 - avg_economy * 0.3
            else:
                avg_wickets = 0
                avg_economy = 0
                form = 0
            
            features.append({
                'player': player_name,
                'match_id': match_id,
                'opponent': opponent,
                'venue': venue,
                'is_home': is_home,
                'prev_avg_wickets': avg_wickets,
                'prev_avg_economy': avg_economy,
                'form': form,
                'target_wickets': wickets,
                'target_economy': economy
            })
        
        return pd.DataFrame(features)
    
    def train_batting_model(self, ball_data, match_data, players):
        """Train batting performance prediction model"""
        all_features = []
        
        for player in players[:100]:
            features = self.prepare_batting_features(ball_data, match_data, player)
            if features is not None and not features.empty:
                all_features.append(features)
        
        if not all_features:
            print("No batting features found for training")
            return None
            
        df = pd.concat(all_features, ignore_index=True)
        
        # Filter out unknown values
        df = df[df['opponent'] != 'Unknown']
        df = df[df['venue'] != 'Unknown']
        
        if df.empty:
            print("No valid batting data after filtering unknowns")
            return None
        
        print(f"Training batting model with {len(df)} samples")
        
        # Encode categorical features
        for col in ['opponent', 'venue']:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[f'batting_{col}'] = le
        
        # Features and target
        X = df[['opponent', 'venue', 'is_home', 'prev_avg_runs', 'prev_avg_sr', 'form']]
        y = df['target_runs']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Batting Model - MAE: {mae:.2f}, R²: {r2:.3f}, Samples: {len(X_train)}")
        
        self.batting_model = model
        return model
    
    def train_bowling_model(self, ball_data, match_data, players):
        """Train bowling performance prediction model"""
        all_features = []
        
        for player in players[:100]:
            features = self.prepare_bowling_features(ball_data, match_data, player)
            if features is not None and not features.empty:
                all_features.append(features)
        
        if not all_features:
            print("No bowling features found for training")
            return None
            
        df = pd.concat(all_features, ignore_index=True)
        
        # Filter out unknown values
        df = df[df['opponent'] != 'Unknown']
        df = df[df['venue'] != 'Unknown']
        
        if df.empty:
            print("No valid bowling data after filtering unknowns")
            return None
        
        print(f"Training bowling model with {len(df)} samples")
        
        # Encode categorical features
        for col in ['opponent', 'venue']:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[f'bowling_{col}'] = le
        
        # Features for wickets prediction
        X_wickets = df[['opponent', 'venue', 'is_home', 'prev_avg_wickets', 'prev_avg_economy', 'form']]
        y_wickets = df['target_wickets']
        
        # Train wickets model
        X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(X_wickets, y_wickets, test_size=0.2, random_state=42)
        X_train_w_scaled = self.scaler.fit_transform(X_train_w)
        X_test_w_scaled = self.scaler.transform(X_test_w)
        
        model_wickets = RandomForestRegressor(n_estimators=100, random_state=42)
        model_wickets.fit(X_train_w_scaled, y_train_w)
        
        # Evaluate wickets model
        y_pred_w = model_wickets.predict(X_test_w_scaled)
        mae_w = mean_absolute_error(y_test_w, y_pred_w)
        
        print(f"Bowling Wickets Model - MAE: {mae_w:.2f}, Samples: {len(X_train_w)}")
        
        self.bowling_model = model_wickets
        return model_wickets
    
    def save_models(self, path):
        """Save trained models and encoders"""
        joblib.dump({
            'batting_model': self.batting_model,
            'bowling_model': self.bowling_model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders
        }, path)
    
    def load_models(self, path):
        """Load trained models and encoders"""
        data = joblib.load(path)
        self.batting_model = data['batting_model']
        self.bowling_model = data['bowling_model']
        self.scaler = data['scaler']
        self.label_encoders = data['label_encoders']

class PlayingXISelector:
    def __init__(self):
        self.selector_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def get_team_players(self, ball_data, team):
        """Get all players who have played for a team"""
        team_matches = ball_data[
            (ball_data['team1'] == team) | 
            (ball_data['team2'] == team)
        ]
        
        players = set()
        players.update(team_matches['batsman'].unique())
        players.update(team_matches['bowler'].unique())
        return [p for p in players if pd.notna(p)]
    
    def train_model(self, ball_data, match_data):
        """Train playing XI selection model"""
        print("Training playing XI model...")
        
        # Get all players and their basic stats
        all_players = set(ball_data['batsman'].unique()) | set(ball_data['bowler'].unique())
        all_players = [p for p in all_players if pd.notna(p)]
        
        training_data = []
        
        for player in all_players[:200]:
            # Basic player stats
            batting_data = ball_data[ball_data['batsman'] == player]
            bowling_data = ball_data[ball_data['bowler'] == player]
            
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
            
            # Calculate performance metrics
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
            
            # Calculate selection score (0-100)
            if role == 'batsman':
                selection_score = min(100, (avg_runs * 1.5) + (avg_sr * 0.3) + (matches_batted * 0.2))
            elif role == 'bowler':
                selection_score = min(100, (avg_wickets * 15) + ((10 - avg_economy) * 3) + (matches_bowled * 0.2))
            else:  # all_rounder
                selection_score = min(100, (avg_runs * 0.8) + (avg_sr * 0.15) + (avg_wickets * 8) + ((10 - avg_economy) * 1.5) + ((matches_batted + matches_bowled) * 0.1))
            
            training_data.append({
                'player': player,
                'role': role,
                'avg_runs': avg_runs,
                'avg_sr': avg_sr,
                'avg_wickets': avg_wickets,
                'avg_economy': avg_economy,
                'total_runs': total_runs,
                'total_wickets': total_wickets,
                'matches_batted': matches_batted,
                'matches_bowled': matches_bowled,
                'selection_score': selection_score
            })
        
        if not training_data:
            print("No training data available")
            return None
        
        df = pd.DataFrame(training_data)
        print(f"Training with {len(df)} players")
        
        # Encode role
        le = LabelEncoder()
        df['role_encoded'] = le.fit_transform(df['role'])
        self.label_encoders['role'] = le
        
        # Features and target (binary: selected if score > 50)
        feature_cols = ['role_encoded', 'avg_runs', 'avg_sr', 'avg_wickets', 'avg_economy', 
                       'total_runs', 'total_wickets', 'matches_batted', 'matches_bowled']
        
        X = df[feature_cols]
        y = (df['selection_score'] > 50).astype(int)  # Selected if score > 50
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Playing XI Model - Accuracy: {accuracy:.3f}, Samples: {len(X_train)}")
        print(classification_report(y_test, y_pred))
        
        self.selector_model = model
        return model
    
    def predict_playing_xi(self, team, ball_data):
        """Predict optimal playing XI for a team"""
        if self.selector_model is None:
            return None
        
        # Get team players
        team_players = self.get_team_players(ball_data, team)
        
        if not team_players:
            print(f"No players found for team {team}")
            return None
        
        predictions = []
        
        for player in team_players:
            # Get player data
            batting_data = ball_data[ball_data['batsman'] == player]
            bowling_data = ball_data[ball_data['bowler'] == player]
            
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
            
            # Prepare features for prediction
            role_encoded = self.label_encoders['role'].transform([role])[0]
            features = [[role_encoded, avg_runs, avg_sr, avg_wickets, avg_economy, 
                        total_runs, total_wickets, matches_batted, matches_bowled]]
            
            probability = self.selector_model.predict_proba(features)[0][1]
            
            predictions.append({
                'player': player,
                'role': role,
                'selection_probability': probability,
                'avg_runs': avg_runs,
                'avg_sr': avg_sr,
                'avg_wickets': avg_wickets,
                'avg_economy': avg_economy,
                'total_runs': total_runs,
                'total_wickets': total_wickets
            })
        
        # Select top 11
        predictions_df = pd.DataFrame(predictions)
        if len(predictions_df) >= 11:
            optimal_xi = predictions_df.nlargest(11, 'selection_probability')
        else:
            optimal_xi = predictions_df
        
        return optimal_xi.sort_values('selection_probability', ascending=False)
    
    def save_model(self, path):
        """Save trained model"""
        joblib.dump({
            'selector_model': self.selector_model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders
        }, path)
    
    def load_model(self, path):
        """Load trained model"""
        data = joblib.load(path)
        self.selector_model = data['selector_model']
        self.scaler = data['scaler']
        self.label_encoders = data['label_encoders']

def train_all_models():
    formats = ['t20', 'odi', 'test']
    
    for format_name in formats:
        print(f"\n=== Training models for {format_name} ===")
        
        try:
            # Load data
            ball_file = f'{format_name}cricsheet_ball_by_ball.csv'
            match_file = f'{format_name}cricsheet_match_summary.csv'
            
            print(f"Loading {ball_file}...")
            ball_data = pd.read_csv(ball_file)
            print(f"Loading {match_file}...")
            match_data = pd.read_csv(match_file)
            
            # Clean and convert dates with error handling
            print("Cleaning dates...")
            ball_data['date'] = clean_dates(ball_data['date'])
            match_data['date'] = clean_dates(match_data['date'])
            
            print(f"Ball data dates: {ball_data['date'].notna().sum()} valid dates")
            print(f"Match data dates: {match_data['date'].notna().sum()} valid dates")
            
            # Train performance predictor
            batsmen = [p for p in ball_data['batsman'].unique() if pd.notna(p)]
            bowlers = [p for p in ball_data['bowler'].unique() if pd.notna(p)]
            
            print(f"Found {len(batsmen)} batsmen and {len(bowlers)} bowlers")
            
            predictor = PlayerPerformancePredictor()
            
            if batsmen:
                print("Training batting model...")
                predictor.train_batting_model(ball_data, match_data, batsmen)
            else:
                print("No batsmen found for training")
            
            if bowlers:
                print("Training bowling model...")
                predictor.train_bowling_model(ball_data, match_data, bowlers)
            else:
                print("No bowlers found for training")
            
            predictor.save_models(f'player_performance_models_{format_name}.pkl')
            print(f"✓ Performance models saved for {format_name}")
            
            # Train playing XI selector
            selector = PlayingXISelector()
            selector.train_model(ball_data, match_data)
            selector.save_model(f'playing_xi_model_{format_name}.pkl')
            print(f"✓ Playing XI model saved for {format_name}")
            
        except Exception as e:
            print(f"✗ Error training {format_name} models: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    train_all_models()
Advanced Cricket Statistics Dashboard
A comprehensive, interactive dashboard for analyzing cricket statistics across multiple formats (Test, ODI, T20) with advanced machine learning capabilities for performance prediction and team selection.

🏏 Features:
📊 Multi-Format Statistical Analysis
Batting Statistics: Runs, averages, strike rates, centuries, fifties
Bowling Statistics: Wickets, economy rates, averages, strike rates
Fielding Statistics: Dismissals, catches, stumpings, dismissal rates
Format Comparison: Compare player performances across Test, ODI, and T20 formats

🤖 Machine Learning Capabilities:
Player Performance Prediction: Predict runs, strike rates, wickets, and economy for upcoming matches
Optimal Team Selection: AI-powered playing XI recommendations based on form, opponent, and venue
Form Analysis: Calculate player form using recent performance data

🔍 Advanced Analytics:
Player vs Player Comparison: Head-to-head performance analysis
Batsman vs Bowler Analysis: Historical matchup statistics
Match Analysis: Over-by-over, partnership, and phase-wise analysis
Trend Analysis: Performance trends over time for players and teams
Ball-by-Ball Analysis: Detailed delivery-level insights

📈 Interactive Visualizations:
Interactive charts and graphs using Plotly
Radar charts for player comparisons
Moving averages and trend lines
Correlation matrices and statistical distributions

🚀 Installation
Prerequisites:
Python 3.8 or higher
pip (Python package manager)


Required Packages
The application requires the following Python packages:
text
streamlit==1.28.0
pandas==2.0.0
numpy==1.24.0
matplotlib==3.7.0
seaborn==0.12.0
plotly==5.14.0
scikit-learn==1.3.0
joblib==1.2.0

Step 3: Prepare Data Files
Place your cricket data CSV files in the project directory. The application expects the following file structure:

Required Data Files:
text
ODI.csv                          # ODI batting statistics
t20.csv                          # T20 batting statistics  
test.csv                         # Test batting statistics

Bowling_ODI.csv                  # ODI bowling statistics
Bowling_t20.csv                  # T20 bowling statistics
Bowling_test.csv                 # Test bowling statistics

Fielding_ODI.csv                 # ODI fielding statistics
Fielding_t20.csv                 # T20 fielding statistics
Fielding_test.csv                # Test fielding statistics

batsman_vs_bowler_odi.csv        # ODI batsman vs bowler data
batsman_vs_bowler_t20.csv        # T20 batsman vs bowler data
batsman_vs_bowler_test.csv       # Test batsman vs bowler data

playing_xi_selection_odi.csv     # ODI playing XI data
playing_xi_selection_t20.csv     # T20 playing XI data  
playing_xi_selection_test.csv    # Test playing XI data

testcricsheet_match_summary.csv  # Test match summaries
odicricsheet_match_summary.csv   # ODI match summaries
t20cricsheet_match_summary.csv   # T20 match summaries

testcricsheet_ball_by_ball.csv   # Test ball-by-ball data
odicricsheet_ball_by_ball.csv    # ODI ball-by-ball data
t20cricsheet_ball_by_ball.csv    # T20 ball-by-ball data
Optional ML Model Files (for enhanced predictions):
text
player_performance_models_odi.pkl    # ODI performance ML models
player_performance_models_t20.pkl    # T20 performance ML models  
player_performance_models_test.pkl   # Test performance ML models

playing_xi_model_odi.pkl            # ODI team selection model
playing_xi_model_t20.pkl            # T20 team selection model
playing_xi_model_test.pkl           # Test team selection model

To run the Application:
streamlit run dashboard.py
The dashboard will open in your default web browser at http://localhost:8501

📁 Project Structure
cricket-analyticss-dashboard/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies 
├── README.md                       # Project documentation
│
├── data/                          # Data directory (optional)
│   ├── ODI.csv
│   ├── t20.csv
│   ├── test.csv
│   └── ... (other CSV files)
│
└── models/                        # ML models directory (optional)
    ├── player_performance_models_odi.pkl
    ├── player_performance_models_t20.pkl
    └── ... (other model files)
    
🎯 Usage Guide
1. Team & Player Stats:
Select cricket format (Test/ODI/T20)
Choose statistics type (Batting/Bowling/Fielding)
Apply filters based on matches, runs, wickets, etc.
Explore top performers, distributions, and relationships

2. Player vs Player Comparison:
Compare multiple batsmen or bowlers
View normalized radar charts for easy comparison
Analyze strengths and weaknesses across metrics

3. Batsman vs Bowler Analysis:
Select specific batsman-bowler matchups
View historical performance data
Analyze dominance and strike rates

4. Team Selection Assistant:
Select team, opposition, and venue
Get AI-powered playing XI recommendations
View player form and venue-specific performance

5. Match Analysis:
Analyze individual matches ball-by-ball
View over-by-over progression
Study partnerships and bowler spells

6. Trend Analysis:
Track player performance trends over time
Analyze team win percentages
Identify form patterns and improvements

7. Performance Predictor:
Predict player performance for upcoming matches
Consider opponent, venue, and pitch conditions
Get confidence levels for predictions


📊 Data Sources
The dashboard is designed to work with data from:

ESPN Cricinfo statistics
Cricsheet ball-by-ball data
Custom cricket databases
Any CSV files with compatible structure


🤝 Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues for:

New analysis features
Data visualization improvements
Machine learning enhancements
Bug fixes and optimizations



🙏 Acknowledgments
Cricket data providers and statisticians
Streamlit community for the excellent framework
Plotly for interactive visualizations
Scikit-learn for machine learning capabilities

Note: This dashboard is designed for cricket analysis and statistical purposes. Actual team selection and player evaluation should consider additional factors beyond statistical analysis.
For questions or support, please open an issue in the GitHub repository.

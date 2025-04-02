import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the master file
try:
    master_df = pd.read_excel('nba_master_file.xlsx')
    print("Loaded nba_master_file.xlsx successfully")
except FileNotFoundError as e:
    print(f"Master file not found: {e}")
    exit()
except Exception as e:
    print(f"An error occurred while loading master file: {e}")
    exit()

# Load the schedule file
try:
    schedule_df = pd.read_excel('AprilSchedule.xlsx')
    print("Loaded schedule.xlsx successfully")
except FileNotFoundError as e:
    print(f"Schedule file not found: {e}")
    exit()
except Exception as e:
    print(f"An error occurred while loading schedule: {e}")
    exit()

# Print the column names of the schedule file to debug
print("Columns in schedule.xlsx:", schedule_df.columns.tolist())

# Define the column names for home and away teams based on VistorSchedule.xlsx
home_team_col = 'Home'
away_team_col = 'Visitor'

# Ensure the schedule has the required columns
required_schedule_cols = [home_team_col, away_team_col]
if not all(col in schedule_df.columns for col in required_schedule_cols):
    print(f"Error: Schedule file must contain columns: {required_schedule_cols}")
    print("Please update 'home_team_col' and 'away_team_col' to match the actual column names.")
    exit()

# Select features for logistic regression (exclude non-numeric and outcome columns)
feature_cols = [col for col in master_df.columns if col not in ['Team', 'Games Played', 'W', 'L'] and master_df[col].dtype in [np.float64, np.int64]]
print("Features used for prediction:", feature_cols)

# Simulate training data by creating hypothetical matchups
teams = master_df['Team'].tolist()
np.random.seed(42)  # For reproducibility
num_simulated_games = 1000  # Number of hypothetical games to simulate

simulated_data = []
simulated_labels = []

for _ in range(num_simulated_games):
    # Randomly select two teams
    home_team = np.random.choice(teams)
    away_team = np.random.choice([t for t in teams if t != home_team])
    
    # Get the stats for both teams
    home_stats = master_df[master_df['Team'] == home_team][feature_cols].iloc[0]
    away_stats = master_df[master_df['Team'] == away_team][feature_cols].iloc[0]
    
    # Compute the difference (home - away)
    diff_stats = home_stats - away_stats
    simulated_data.append(diff_stats.values)
    
    # Simulate the outcome: If home team has more wins, they are more likely to win
    home_wins = master_df[master_df['Team'] == home_team]['W'].iloc[0]
    away_wins = master_df[master_df['Team'] == away_team]['W'].iloc[0]
    prob_home_win = (home_wins / (home_wins + away_wins)) + np.random.uniform(-0.2, 0.2)  # Add noise
    home_wins_game = 1 if prob_home_win > 0.5 else 0
    simulated_labels.append(home_wins_game)

# Convert simulated data to a DataFrame
X_simulated = pd.DataFrame(simulated_data, columns=feature_cols)
y_simulated = np.array(simulated_labels)

# Split the simulated data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_simulated, y_simulated, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the logistic regression model
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train_scaled, y_train)

# Evaluate the model on the test set
train_accuracy = logreg.score(X_train_scaled, y_train)
test_accuracy = logreg.score(X_test_scaled, y_test)
print(f"Training accuracy: {train_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

# Prepare the schedule data for prediction
predictions = []
for idx, row in schedule_df.iterrows():
    home_team = row[home_team_col]
    away_team = row[away_team_col]
    
    # Check if both teams exist in the master file
    if home_team not in master_df['Team'].values or away_team not in master_df['Team'].values:
        print(f"Warning: Game {idx} skipped - Team(s) not found: {home_team} vs {away_team}")
        continue
    
    # Get the stats for both teams
    home_stats = master_df[master_df['Team'] == home_team][feature_cols].iloc[0]
    away_stats = master_df[master_df['Team'] == away_team][feature_cols].iloc[0]
    
    # Compute the difference (home - away) and keep it as a DataFrame
    diff_stats = pd.DataFrame([home_stats - away_stats], columns=feature_cols)
    diff_stats_scaled = scaler.transform(diff_stats)  # Pass DataFrame to scaler
    
    # Predict the probability of home team winning
    prob_home_win = logreg.predict_proba(diff_stats_scaled)[0][1]  # Probability of class 1 (home win)
    
    # Determine the predicted winner
    predicted_winner = home_team if prob_home_win >= 0.5 else away_team
    
    # Store the prediction
    predictions.append({
        'Date': row.get('Date', 'N/A'),
        'Home Team': home_team,
        'Away Team': away_team,
        'Probability Home Wins': prob_home_win,
        'Predicted Winner': predicted_winner
    })

# Convert predictions to a DataFrame
predictions_df = pd.DataFrame(predictions)

# Save predictions to an Excel file
predictions_df.to_excel('nba_game_predictions.xlsx', index=False)
print("Game predictions saved as 'nba_game_predictions.xlsx'")

# Display the predictions
print("\nGame Predictions:")
print(predictions_df)

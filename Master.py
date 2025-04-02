import pandas as pd
import os

# Print current working directory and folder contents
print("Current Working Directory:", os.getcwd())
print("Files in Stats for Teams:", os.listdir('.\\Stats for Teams\\'))

# Delete the previous master file if it exists
master_file_path = 'nba_master_file.xlsx'
if os.path.exists(master_file_path):
    os.remove(master_file_path)
    print(f"Deleted previous master file: {master_file_path}")
else:
    print("No previous master file found.")

# Load each document into a DataFrame
try:
    df1 = pd.read_excel('.\\Stats for Teams\\PerGame.xlsx')  # First document (Per Game stats)
    print("Loaded PerGame.xlsx successfully")
    df2 = pd.read_excel('.\\Stats for Teams\\Per 100 Poss.xlsx')  # Second document (Per 100 stats)
    print("Loaded Per 100 Poss.xlsx successfully")
    df3 = pd.read_excel('.\\Stats for Teams\\Totals.xlsx')  # Third document (Totals)
    print("Loaded Totals.xlsx successfully")
    # Load Advanced Stats, with header on the second row (index 1), and select only "Team", "W", "L"
    df4 = pd.read_excel('.\\Stats for Teams\\Advanced Stats.xlsx', header=1)
    df4 = df4[['Team', 'W', 'L']]  # Keep only "Team", "W", and "L" columns
    print("Loaded Advanced Stats.xlsx successfully (only W and L columns)")
except FileNotFoundError as e:
    print(f"File not found: {e}")
    exit()
except Exception as e:
    print(f"An error occurred: {e}")
    exit()

# List of DataFrames with their source file names (without .xlsx extension)
dfs = [
    (df1, "PerGame"),
    (df2, "Per100Poss"),
    (df3, "Totals"),
    (df4, "AdvancedStats")
]

# Clean each DataFrame: Remove "Rk" columns, drop "Unnamed" columns, handle "G", and ensure "Team" is present
for i, (df, source_name) in enumerate(dfs):
    # Drop columns starting with "Unnamed"
    df.drop(columns=[col for col in df.columns if 'Unnamed' in col], inplace=True)
    # Drop any column named "Rk"
    df.drop(columns=[col for col in df.columns if col == 'Rk'], inplace=True, errors='ignore')
    # Ensure "Team" column exists
    if "Team" not in df.columns:
        print(f"Error: 'Team' column not found in DataFrame {i+1}")
        continue
    # Remove any rows that are not team-specific (e.g., "League Average")
    df = df[df['Team'].str.contains('League Average') == False].copy()
    # Handle "G" column: Keep it only in the first DataFrame (PerGame), rename to "Games Played", drop from others
    if i == 0:  # First DataFrame (PerGame)
        if 'G' in df.columns:
            df.rename(columns={'G': 'Games Played'}, inplace=True)
    else:  # All other DataFrames
        df.drop(columns=['G'], inplace=True, errors='ignore')
    # Update the DataFrame in the list
    dfs[i] = (df, source_name)

# Rename columns to include the source file name, except for AdvancedStats
for i, (df, source_name) in enumerate(dfs):
    if source_name == "AdvancedStats":
        # Skip renaming for AdvancedStats columns (keep "W" and "L" as is)
        pass
    else:
        # Rename columns for other DataFrames, excluding "Team" and "Games Played"
        df.columns = [
            col if col in ['Team', 'Games Played'] else f"{col}_{source_name}"
            for col in df.columns
        ]
    dfs[i] = (df, source_name)  # Update the DataFrame in the list

# Extract just the DataFrames from the list
dfs = [df for df, _ in dfs]

# Merge all DataFrames on "Team" column using an outer join
master_df = dfs[0]
for df in dfs[1:]:
    master_df = master_df.merge(df, on="Team", how="outer")

# Sort the DataFrame by "Team" in alphabetical order
master_df = master_df.sort_values(by='Team', ascending=True)

# Reset index after sorting
master_df.reset_index(drop=True, inplace=True)

# Reorder columns: "Team", "Games Played", "W", "L", then all other columns
# First, ensure these columns exist in the DataFrame
required_columns = ['Team', 'Games Played', 'W', 'L']
missing_columns = [col for col in required_columns if col not in master_df.columns]
if missing_columns:
    print(f"Warning: The following required columns are missing: {missing_columns}")
else:
    # Get all columns except the required ones
    other_columns = [col for col in master_df.columns if col not in required_columns]
    # New column order: required columns first, then others
    new_column_order = required_columns + other_columns
    # Reorder the DataFrame
    master_df = master_df[new_column_order]

# Check for duplicate columns
duplicate_columns = master_df.columns[master_df.columns.duplicated()]
if len(duplicate_columns) > 0:
    print("Warning: Duplicate columns found after merging:", duplicate_columns)
else:
    print("No duplicate columns found after merging.")

# Save to an Excel file
master_df.to_excel('nba_master_file.xlsx', index=False)
print("Merged dataset saved as 'nba_master_file.xlsx'")

# Display the first few rows to verify
print(master_df.head())
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
except FileNotFoundError as e:
    print(f"File not found: {e}")
    exit()
except Exception as e:
    print(f"An error occurred: {e}")
    exit()

# List of DataFrames with their source file names (without .xlsx extension)
dfs = [
    (df1, "PerGame"),
    (df2, "Per100Poss"),  # Remove spaces for cleaner column names
    (df3, "Totals")
]

# Clean each DataFrame: Remove "Rk" columns, drop "Unnamed" columns, and ensure "Team" is present
for i, (df, _) in enumerate(dfs):
    # Drop columns starting with "Unnamed"
    df.drop(columns=[col for col in df.columns if 'Unnamed' in col], inplace=True)
    # Drop any column named "Rk"
    df.drop(columns=[col for col in df.columns if col == 'Rk'], inplace=True, errors='ignore')
    # Ensure "Team" column exists
    if "Team" not in df.columns:
        print(f"Error: 'Team' column not found in DataFrame {i+1}")
        continue
    # Remove any rows that are not team-specific (e.g., "League Average")
    dfs[i] = (df[df['Team'].str.contains('League Average') == False].copy(), dfs[i][1])

# Rename columns to include the source file name (e.g., "FG" -> "FG_PerGame")
for i, (df, source_name) in enumerate(dfs):
    # Skip renaming the "Team" column
    df.columns = [f"{col}_{source_name}" if col != "Team" else "Team" for col in df.columns]
    dfs[i] = (df, source_name)  # Update the DataFrame in the list

# Extract just the DataFrames from the list (discard the source names now that they're in the column names)
dfs = [df for df, _ in dfs]

# Merge all DataFrames on "Team" column using an outer join
master_df = dfs[0]
for df in dfs[1:]:
    master_df = master_df.merge(df, on="Team", how="outer")

# Sort the DataFrame by "Team" in alphabetical order
master_df = master_df.sort_values(by='Team', ascending=True)

# Reset index after sorting
master_df.reset_index(drop=True, inplace=True)

# Since we added source file names to columns, there should be no duplicates
# But let's check for safety
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
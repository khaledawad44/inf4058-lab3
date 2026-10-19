import pandas as pd
import numpy as np

DATA_FILE = "yellow_tripdata_2021-01.csv.gz"
CHUNK_SIZE = 100_000

# ==============================================================================
# STEP 1: LOOK (First 100 rows)
# ==============================================================================
print("=" * 60)
print("STEP 1: LOOK (FIRST 100 ROWS)")
print("=" * 60)

df_100 = pd.read_csv(DATA_FILE, nrows=100)

print("\n--- df.head() ---")
print(df_100.head())

print("\n--- df.dtypes ---")
print(df_100.dtypes)

# ==============================================================================
# STEP 2: PROFILE (Read whole file in chunks of 100,000 rows)
# ==============================================================================
print("\n" + "=" * 60)
print("STEP 2: PROFILE (CHUNKED 100,000 ROWS)")
print("=" * 60)

total_rows = 0
null_counts = None
dtypes = None
min_dict = {}
max_dict = {}

for chunk in pd.read_csv(DATA_FILE, chunksize=CHUNK_SIZE, low_memory=False):
    total_rows += len(chunk)
    
    if dtypes is None:
        dtypes = chunk.dtypes
        null_counts = chunk.isnull().sum()
    else:
        null_counts += chunk.isnull().sum()
        
    for col in chunk.columns:
        valid_vals = chunk[col].dropna()
        if not valid_vals.empty:
            try:
                c_min = valid_vals.min()
                c_max = valid_vals.max()
                if col not in min_dict or c_min < min_dict[col]:
                    min_dict[col] = c_min
                if col not in max_dict or c_max > max_dict[col]:
                    max_dict[col] = c_max
            except Exception:
                pass

profile_table = pd.DataFrame({
    'dtype': dtypes,
    'null_count': null_counts,
    'null_pct': ((null_counts / total_rows) * 100).round(2),
    'min': pd.Series(min_dict),
    'max': pd.Series(max_dict)
})

print(f"\nTotal Rows Profiled: {total_rows}")
print("\n--- PROFILING RESULTS ---")
print(profile_table.to_string())

# ==============================================================================
# STEP 3: FIND PROBLEMS (Count occurrences across the dataset)
# ==============================================================================
print("\n" + "=" * 60)
print("STEP 3: FIND PROBLEMS")
print("=" * 60)

problem_counts = {
    'pickups_outside_jan_2021': 0,
    'trip_distance_zero_or_negative': 0,
    'fare_amount_zero_or_negative': 0,
    'passenger_count_zero': 0,
    'duplicate_rows': 0
}

for chunk in pd.read_csv(DATA_FILE, chunksize=CHUNK_SIZE, low_memory=False):
    pickup_dt = pd.to_datetime(chunk['tpep_pickup_datetime'], errors='coerce')
    
    outside_jan = (pickup_dt < '2021-01-01') | (pickup_dt > '2021-01-31 23:59:59') | pickup_dt.isna()
    problem_counts['pickups_outside_jan_2021'] += int(outside_jan.sum())
    
    problem_counts['trip_distance_zero_or_negative'] += int((chunk['trip_distance'] <= 0).sum())
    problem_counts['fare_amount_zero_or_negative'] += int((chunk['fare_amount'] <= 0).sum())
    problem_counts['passenger_count_zero'] += int((chunk['passenger_count'] == 0).sum())
    problem_counts['duplicate_rows'] += int(chunk.duplicated().sum())

problems_table = pd.DataFrame(
    list(problem_counts.items()), 
    columns=['Problem', 'Count']
)
print("\n--- PROBLEMS IDENTIFIED ---")
print(problems_table.to_string(index=False))

# ==============================================================================
# STEP 4 & 5: DECIDE & CLEAN
# ==============================================================================
print("\n" + "=" * 60)
print("STEP 4 & 5: CLEAN FUNCTION & EXECUTION")
print("=" * 60)

def clean(df: pd.DataFrame):
    """
    Cleans taxi trip data according to defined data quality rules.
    Does NOT modify the original dataframe.
    Returns: (cleaned_df, rules_removed_dict)
    """
    df_clean = df.copy()
    rules_removed = {}
    
    # Rule 1: Remove duplicate rows
    dups = df_clean.duplicated()
    rules_removed['duplicate_rows'] = int(dups.sum())
    df_clean = df_clean[~dups]
    
    # Rule 2: Remove pickups outside January 2021
    pickup_dt = pd.to_datetime(df_clean['tpep_pickup_datetime'], errors='coerce')
    valid_dates = (pickup_dt >= '2021-01-01') & (pickup_dt <= '2021-01-31 23:59:59')
    rules_removed['pickups_outside_jan_2021'] = int((~valid_dates).sum())
    df_clean = df_clean[valid_dates]
    
    # Rule 3: Remove trip distance <= 0
    valid_distance = df_clean['trip_distance'] > 0
    rules_removed['trip_distance_zero_or_negative'] = int((~valid_distance).sum())
    df_clean = df_clean[valid_distance]
    
    # Rule 4: Remove fare amount <= 0
    valid_fare = df_clean['fare_amount'] > 0
    rules_removed['fare_amount_zero_or_negative'] = int((~valid_fare).sum())
    df_clean = df_clean[valid_fare]
    
    # Rule 5: Fix passenger_count == 0 or null -> impute with 1 (default solo passenger)
    zero_passengers = (df_clean['passenger_count'] == 0) | (df_clean['passenger_count'].isna())
    rules_removed['passenger_count_imputed_to_1'] = int(zero_passengers.sum())
    df_clean.loc[zero_passengers, 'passenger_count'] = 1
    
    return df_clean, rules_removed

# Run clean() on a test chunk
test_sample = pd.read_csv(DATA_FILE, nrows=100_000, low_memory=False)
cleaned_sample, removal_summary = clean(test_sample)

print(f"\nRows before cleaning: {len(test_sample)}")
print(f"Rows after cleaning:  {len(cleaned_sample)}")
print(f"Total rows dropped:   {len(test_sample) - len(cleaned_sample)}")
print("\n--- REMOVAL SUMMARY PER RULE ---")
summary_table = pd.DataFrame(list(removal_summary.items()), columns=['Rule', 'Rows Affected/Removed'])
print(summary_table.to_string(index=False))

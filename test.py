import pandas as pd
import itertools
import time
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output

# Load the dataset
file_path = r"datasets\joined_di_hdi.csv"  # Change this if needed
df = pd.read_csv(file_path)

# Define the range of p, d, q values to test
p = range(0, 3)
d = range(0, 2)
q = range(0, 3)
pdq_combinations = list(itertools.product(p, d, q))

# Dictionary to store the best (p,d,q) counts
best_pdq_counts = {}

# Track progress
start_time = time.time()
total_countries = len(df['country'].unique())
processed = 0

# Iterate through each country in the dataset
for country in df['country'].unique():
    country_df = df[df['country'] == country].sort_values(by='year')

    if len(country_df) > 10:  # Ensure enough data points
        hdi_series = country_df['human_development_index']
        di_series = country_df['democracy_index']

        best_pdq = None
        best_aic = float("inf")

        # Iterate through all (p,d,q) combinations to find the best model
        for pdq in pdq_combinations:
            try:
                model = ARIMA(hdi_series, order=pdq, exog=di_series)
                model_fit = model.fit()
                aic = model_fit.aic

                if aic < best_aic:
                    best_pdq = pdq
                    best_aic = aic

            except Exception:
                continue  # Skip failed models

        # Store the count of each best (p,d,q)
        if best_pdq:
            best_pdq_counts[best_pdq] = best_pdq_counts.get(best_pdq, 0) + 1

    # Print progress and current (p,d,q) counts every 5 countries
    processed += 1
    if processed % 5 == 0 or processed == total_countries:
        elapsed_time = time.time() - start_time
        print(f"Processed {processed}/{total_countries} countries - Elapsed Time: {elapsed_time:.2f}s")
        print("Current (p,d,q) counts:")
        for key, value in sorted(best_pdq_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"{key}: {value}")
        print("-" * 50)

# Convert results to DataFrame and save as CSV
best_pdq_counts_df = pd.DataFrame(list(best_pdq_counts.items()), columns=['Best (p,d,q)', 'Count'])
best_pdq_counts_df = best_pdq_counts_df.sort_values(by='Count', ascending=False)

# Save the results
output_path = "best_pdq_counts.csv"
best_pdq_counts_df.to_csv(output_path, index=False)
print(f"Best ARIMA (p,d,q) results saved to {output_path}")

from nsepython import index_history
import pandas as pd

# Indexes you want (official NSE names)
nse_indexes = [
    "NIFTY 50",
    "NIFTY BANK",
    "NIFTY IT",
    "NIFTY FMCG",
    "NIFTY AUTO",
    "NIFTY PHARMA",
    "NIFTY ENERGY",
    "NIFTY METAL",
    "NIFTY REALTY",
    "NIFTY INFRA"
]

start_date = "01-01-2000"  # DD-MM-YYYY format
end_date = "01-01-2025"

all_data = []

print("Downloading NSE Index Data...")

for idx in nse_indexes:
    try:
        print(f"Fetching {idx}...")
        df = index_history(idx, start_date, end_date)
        if df.empty:
            print(f"❌ No data for {idx}")
            continue
        df["Index"] = idx
        all_data.append(df)
    except Exception as e:
        print(f"❌ Failed for {idx}: {e}")

# Merge all into one DataFrame
final_df = pd.concat(all_data, axis=0, ignore_index=True)

# Save to CSV
output_file = "nse_indexes_2000_2025.csv"
final_df.to_csv(output_file, index=False)

print(f"\n✅ Index data saved to {output_file}")




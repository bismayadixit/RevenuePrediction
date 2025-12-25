import pandas as pd

# 1. Load dataset
df = pd.read_csv("dataset/movies_raw.csv")

# 2. Clean column names (important)
df.columns = df.columns.str.strip()

print("COLUMNS IN DATASET:")
print(df.columns.tolist())

# 3. Select required columns
df = df[['$Worldwide', '$Domestic', '$Foreign', 'Genres', 'Year']]

# 4. Rename columns (standard names)
df.rename(columns={
    '$Worldwide': 'worldwide_revenue',
    '$Domestic': 'domestic_revenue',
    '$Foreign': 'foreign_revenue',
    'Genres': 'genre',
    'Year': 'release_year'
}, inplace=True)

# 5. Remove $ and commas from revenue columns
for col in ['worldwide_revenue', 'domestic_revenue', 'foreign_revenue']:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace('$', '', regex=False)
        .str.replace(',', '', regex=False)
    )

# 6. Convert to numeric
df['worldwide_revenue'] = pd.to_numeric(df['worldwide_revenue'], errors='coerce')
df['domestic_revenue'] = pd.to_numeric(df['domestic_revenue'], errors='coerce')
df['foreign_revenue'] = pd.to_numeric(df['foreign_revenue'], errors='coerce')

# 7. Remove missing / invalid rows
df.dropna(inplace=True)
df = df[df['worldwide_revenue'] > 0]

# 8. Final dataset columns
df = df[['domestic_revenue', 'foreign_revenue',
         'genre', 'release_year', 'worldwide_revenue']]

# 9. Save cleaned dataset
df.to_csv("dataset/movies_cleaned.csv", index=False)

print("✅ Dataset preparation completed successfully!")
print("📁 Saved as dataset/movies_cleaned.csv")

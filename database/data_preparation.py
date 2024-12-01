import pandas as pd
import numpy as np
import sqlite3

# Load and clean dataset
df = pd.read_csv(
    'database/openfoodfacts.csv',
    delimiter=';',
    encoding='ISO-8859-1',
    dtype={'code': str},
    low_memory=False
)

# Keep and clean relevant columns
columns_to_keep = [
    'code', 'price', 'product_name_fr', 'quantity', 'labels', 
    'ingredients_text_fr', 'sugars_value', 'off:nutriscore_grade', 'off:ecoscore_grade'
]
df_cleaned = df[columns_to_keep].dropna(subset=['code', 'price', 'product_name_fr']).fillna({
    'sugars_value': 0,
    'off:nutriscore_grade': 'unknown',
    'off:ecoscore_grade': 'unknown'
})

# Rename columns for consistency
df_cleaned.rename(columns={
    'product_name_fr': 'product_name',
    'ingredients_text_fr': 'ingredients',
    'sugars_value': 'sugars',
    'off:nutriscore_grade': 'nutriscore_grade',
    'off:ecoscore_grade': 'ecoscore_grade'
}, inplace=True)

# Convert 'price' and 'sugars' to float, replacing commas and handling errors
df_cleaned['price'] = pd.to_numeric(df_cleaned['price'].str.replace(',', '.'), errors='coerce')
df_cleaned['sugars'] = pd.to_numeric(df_cleaned['sugars'], errors='coerce')

# Remove rows with invalid or missing price values
df_cleaned = df_cleaned.dropna(subset=['price'])

# Add derived 'palm_oil' column
df_cleaned['palm_oil'] = df_cleaned['ingredients'].apply(
    lambda x: 'with' if isinstance(x, str) and 'huile de palme' in x.lower() else 'without'
)

# Add 'price_per_kg' column, handling various quantity formats
def calculate_price_per_kg(row):
    try:
        if 'g' in str(row['quantity']):
            grams = float(row['quantity'].replace('g', '').strip())
            return float(row['price']) / (grams / 1000)
        elif 'kg' in str(row['quantity']):
            kilograms = float(row['quantity'].replace('kg', '').strip())
            return float(row['price']) / kilograms
        else:
            return None
    except:
        return None

df_cleaned['price_per_kg'] = df_cleaned.apply(calculate_price_per_kg, axis=1)

# Add 'recommended_by' column with multiple associations for 7 random products
df_cleaned['recommended_by'] = ''
specific_indices = df_cleaned.sample(7).index
associations = ['Foodwatch', 'Greenpeace', '60 millions de consommateurs', 'Ufc Que Choisir']
for index in specific_indices:
    df_cleaned.at[index, 'recommended_by'] = ', '.join(
        np.random.choice(associations, size=np.random.randint(2, 4), replace=False)
    )

# Remove duplicates based on 'code'
df_cleaned.drop_duplicates(subset=['code'], inplace=True)

# Save to SQLite database
conn = sqlite3.connect('products.db')
cursor = conn.cursor()

# Create or replace the 'products' table with updated schema
cursor.execute('''
CREATE TABLE IF NOT EXISTS products (
    code TEXT PRIMARY KEY, -- Barcodes stored as text
    price REAL,
    product_name TEXT,
    quantity TEXT,
    labels TEXT,
    ingredients TEXT,
    sugars REAL,
    nutriscore_grade TEXT,
    ecoscore_grade TEXT,
    palm_oil TEXT,
    price_per_kg REAL,
    recommended_by TEXT
)
''')
df_cleaned.to_sql('products', conn, if_exists='replace', index=False)
conn.close()

print("Database initialized and data saved to 'products.db'")

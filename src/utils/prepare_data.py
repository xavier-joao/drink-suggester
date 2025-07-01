import pandas as pd
from datasets import load_dataset
import unicodedata

def normalize_text(text):
    if not isinstance(text, str):
        return text
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join([c for c in nfkd if not unicodedata.combining(c)]).lower().strip()

# Load your adapted drinks list
drinks_df = pd.read_csv('src/database/drinks_list.csv')
drinks_df['name_norm'] = drinks_df['name'].apply(normalize_text)

# Load the original HuggingFace dataset
original_ds = load_dataset('erwanlc/cocktails_recipe')
original_df = original_ds["train"].to_pandas()
original_df['title_norm'] = original_df['title'].apply(normalize_text)

# Merge on normalized name/title
merged_df = pd.merge(
    drinks_df,
    original_df[['title_norm', 'ingredients', 'recipe', 'garnish', 'glass']],
    left_on='name_norm',
    right_on='title_norm',
    how='left',
    suffixes=('', '_original')
)

# Rename columns for clarity
merged_df = merged_df.rename(columns={
    'ingredients': 'ingredients_adapted',
    'ingredients_original': 'ingredients',
})

# Reorder columns for readability
final_columns = [
    'name', 'ingredients_adapted', 'ingredients', 'recipe', 'garnish', 'glass'
]
merged_df = merged_df[final_columns]

# Save to new CSV
merged_df.to_csv('src/database/drinks_list_full.csv', index=False)

print("Merged and saved to src/database/drinks_list_full.csv")
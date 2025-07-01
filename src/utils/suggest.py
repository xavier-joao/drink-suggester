import pandas as pd
import random
import re
import unicodedata
import json
import os  
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

script_dir = os.path.dirname(os.path.abspath(__file__))

json_path = os.path.join(script_dir, '..', 'database', 'flavor_map.json')

try:
    with open(json_path, 'r') as f:
        FLAVOR_MAP = json.load(f)
except FileNotFoundError:
    print(f"Error: Could not find 'flavor_map.json' at expected path: {json_path}")
    exit()


def flavor_probabilities(ingredients):
    norm_ingredients = [i.lower().strip() for i in ingredients]
    flavor_counts = {flavor: 0 for flavor in FLAVOR_MAP}
    for ingr in norm_ingredients:
        for flavor, flavor_ings in FLAVOR_MAP.items():
            if any(f in ingr for f in flavor_ings):
                flavor_counts[flavor] += 1
    total = sum(flavor_counts.values())
    if total == 0:
        return {flavor: 0.0 for flavor in FLAVOR_MAP}
    return {flavor: round(count / total, 3) for flavor, count in flavor_counts.items()}

def normalize_text(text):
    if not isinstance(text, str):
        return text
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join([c for c in nfkd if not unicodedata.combining(c)]).lower()

def remove_parentheses(text):
    text = re.sub(r'\s*\([^)]*\)', '', text).strip()
    return normalize_text(text)

def normalize_ingredients(ingredients):
    return [remove_parentheses(i) for i in ingredients]

def prepare_from_stringified_list(path):
    df = pd.read_csv(path)
    df['ingredients_list'] = df['ingredients'].apply(
        lambda ingr: [remove_parentheses(i.strip()) for i in ingr.split(',')]
    )
    df['ingredients_str'] = df['ingredients_list'].apply(
        lambda x: ', '.join(sorted([i.strip().lower() for i in x]))
    )
    df['name'] = df['name'].apply(normalize_text)
    df = df[df['ingredients_list'].apply(lambda x: len(x) > 0)]

    output_csv_path = os.path.join(script_dir, '..', 'database', 'drinks_list.csv')
    df[['name', 'ingredients_str']].rename(
        columns={'ingredients_str': 'ingredients'}
    ).to_csv(output_csv_path, index=False)
    
    return df

def train_classifier():
    db_path = os.path.join(script_dir, '..', 'database', 'drinks_list.csv')
    drinks_df = pd.read_csv(db_path)
    
    positives = drinks_df['ingredients'].tolist()
    negatives = generate_negative_samples(drinks_df, n_samples=len(positives))
    X = positives + negatives
    y = [1] * len(positives) + [0] * len(negatives)
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(', '), token_pattern=None)
    X_vec = vectorizer.fit_transform(X)
    clf = LogisticRegression()
    clf.fit(X_vec, y)
    drinks_vec = X_vec[:len(positives)]
    return clf, vectorizer, drinks_vec, drinks_df

def find_similar_drinks(ingredients, vectorizer, drinks_vec, drinks_df, top_n, min_similarity):
    norm_ingredients = normalize_ingredients(ingredients)
    input_str = ', '.join(sorted(norm_ingredients))
    input_vec = vectorizer.transform([input_str])
    similarities = cosine_similarity(input_vec, drinks_vec).flatten()
    results = []
    for idx in similarities.argsort()[::-1]:
        score = similarities[idx]
        if score < min_similarity:
            break
        results.append({
            'name': drinks_df.iloc[idx]['name'],
            'ingredients': drinks_df.iloc[idx]['ingredients'].split(', '),
            'similarity': round(float(score), 4)
        })
        if len(results) >= top_n:
            break
    return results

def predict_drink(ingredients, clf, vectorizer):
    norm_ingredients = normalize_ingredients(ingredients)
    user_ingredients = ', '.join(sorted(norm_ingredients))
    user_vec = vectorizer.transform([user_ingredients])
    return clf.predict_proba(user_vec)[0][1]

def generate_negative_samples(drinks_df, n_samples=1000):
    all_ingredients = list(set(
        ingredient.strip()
        for ingr_list in drinks_df['ingredients']
        for ingredient in ingr_list.split(', ')
    ))
    
    negatives = set()
    existing_combinations = set(drinks_df['ingredients'])
    
    while len(negatives) < n_samples:
        n_ingredients = random.randint(2, 5)
        sample = ', '.join(sorted(random.sample(all_ingredients, n_ingredients)))
        
        if sample not in existing_combinations:
            negatives.add(sample)
    
    return list(negatives)
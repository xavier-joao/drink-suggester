import pandas as pd
import random
import re
import unicodedata
import json
import os
import numpy as np
from fuzzywuzzy import fuzz
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

from fuzzywuzzy import fuzz
import numpy as np
from functools import lru_cache

# Cache the classifier training to avoid recomputing
@lru_cache(maxsize=1)
def get_classifier():
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

def find_similar_drinks(ingredients, vectorizer, drinks_vec, drinks_df, 
                       top_n=20, min_similarity=0.2, 
                       fuzzy_threshold=50, use_fuzzy_weight=0.4):
    """
    Optimized version with better fuzzy matching and early termination
    """
    norm_ingredients = normalize_ingredients(ingredients)
    input_str = ', '.join(sorted(norm_ingredients))
    
    # Early return if no ingredients
    if not norm_ingredients:
        return []
    
    # Vectorize input
    input_vec = vectorizer.transform([input_str])
    
    # Get base cosine similarities
    cosine_sims = cosine_similarity(input_vec, drinks_vec).flatten()
    
    # Pre-calculate ingredient sets for fuzzy matching
    drinks_ingredients = [set(ing.split(', ')) for ing in drinks_df['ingredients']]
    
    # Calculate fuzzy matches with early termination
    fuzzy_scores = np.zeros(len(drinks_df))
    for i, drink_ingredients in enumerate(drinks_ingredients):
        total_score = 0
        matched = 0
        
        for user_ingr in norm_ingredients:
            best_score = 0
            # Only check for best match if cosine similarity is promising
            if cosine_sims[i] > min_similarity/2:  
                best_score = max(
                    fuzz.partial_ratio(user_ingr, ingr)  # More forgiving than token_set_ratio
                    for ingr in drink_ingredients
                )
                if best_score >= fuzzy_threshold:
                    total_score += best_score
                    matched += 1
        
        # Only calculate score if we found at least one match
        if matched > 0:
            fuzzy_scores[i] = total_score / (100 * matched)
    
    # Combine scores
    combined_scores = (1 - use_fuzzy_weight) * cosine_sims + use_fuzzy_weight * fuzzy_scores
    
    # Get top results
    top_indices = np.argpartition(-combined_scores, min(top_n, len(combined_scores)))[:top_n]
    top_indices = top_indices[combined_scores[top_indices] >= min_similarity]
    
    # Sort the top results
    top_indices = top_indices[np.argsort(-combined_scores[top_indices])]
    
    # Prepare results
    results = []
    for idx in top_indices:
        drink_ingredients = drinks_df.iloc[idx]['ingredients'].split(', ')
        ingredient_matches = []
        
        for user_ingr in norm_ingredients:
            best_match = max(
                [(ingr, fuzz.partial_ratio(user_ingr, ingr)) 
                for ingr in drink_ingredients],
                key=lambda x: x[1]
            )
            ingredient_matches.append({
                'user_ingredient': user_ingr,
                'matched_ingredient': best_match[0],
                'score': best_match[1]
            })
        
        results.append({
            'name': drinks_df.iloc[idx]['name'],
            'ingredients': drink_ingredients,
            'similarity': round(float(combined_scores[idx]), 4),
            'ingredient_matches': ingredient_matches
        })
    
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
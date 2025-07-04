import pandas as pd
import random
import re
import unicodedata
import json
import os
import numpy as np
import logging
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from functools import lru_cache

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

script_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(script_dir, '..', 'database', 'flavor_map.json')

try:
    with open(json_path, 'r') as f:
        FLAVOR_MAP = json.load(f)
except FileNotFoundError:
    logger.error(f"Could not find 'flavor_map.json' at path: {json_path}")
    exit()

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

def train_classifier():
    db_path = os.path.join(script_dir, '..', 'database', 'drinks_list.csv')
    drinks_df = pd.read_csv(db_path)
    positives = drinks_df['ingredients'].tolist()
    negatives = generate_negative_samples(drinks_df, n_samples=len(positives))
    X = positives + negatives
    y = [1] * len(positives) + [0] * len(negatives)
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(', '), token_pattern=None)
    X_vec = vectorizer.fit_transform(X)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_vec, y)
    all_ingredients_vocab = list(vectorizer.vocabulary_.keys())
    return clf, vectorizer, drinks_df, all_ingredients_vocab

@lru_cache(maxsize=1)
def get_classifier():
    return train_classifier()

def autocorrect_ingredients(ingredients, vocab, threshold=85):
    norm_ingredients = normalize_ingredients(ingredients)
    corrected_ingredients = []
    for ing in norm_ingredients:
        best_match, score, _ = process.extractOne(ing, vocab, scorer=fuzz.WRatio)
        if score >= threshold:
            logger.info(f"Autocorrecting '{ing}' to '{best_match}' with WRatio score {score}")
            corrected_ingredients.append(best_match)
        else:
            corrected_ingredients.append(ing)
    return corrected_ingredients

def _find_similar_drinks_internal(user_ingredients, drinks_df):
    norm_user_ingredients = normalize_ingredients(user_ingredients)
    if not norm_user_ingredients:
        return []

    all_results = []
    max_possible_score = len(norm_user_ingredients) * 100.0

    for index, row in drinks_df.iterrows():
        drink_ingredients = row['ingredients'].split(', ')
        total_score = 0
        ingredient_matches = []

        for user_ingr in norm_user_ingredients:
            best_match = process.extractOne(user_ingr, drink_ingredients, scorer=fuzz.WRatio)
            if best_match:
                matched_ingredient, score = best_match[0], best_match[1]
                total_score += score
                if score > 60:
                    ingredient_matches.append({
                        'user_ingredient': user_ingr,
                        'matched_ingredient': matched_ingredient,
                        'score': score
                    })
        
        if max_possible_score > 0:
            final_similarity = total_score / max_possible_score
        else:
            final_similarity = 0.0

        if final_similarity > 0.2:
            all_results.append({
                'name': row['name'],
                'ingredients': drink_ingredients,
                'similarity': round(final_similarity, 4),
                'ingredient_matches': ingredient_matches
            })

    all_results.sort(key=lambda x: x['similarity'], reverse=True)
    return all_results[:20]

def predict_drink(ingredients, clf, vectorizer):
    norm_ingredients = normalize_ingredients(ingredients)
    user_ingredients_str = ', '.join(sorted(norm_ingredients))
    if not user_ingredients_str:
        return 0.0
    user_vec = vectorizer.transform([user_ingredients_str])
    return clf.predict_proba(user_vec)[0][1]

def get_drink_recommendations(user_ingredients):
    clf, vectorizer, drinks_df, vocab = get_classifier()
    
    results = _find_similar_drinks_internal(user_ingredients, drinks_df)
    
    if not results:
        logger.warning("Direct search failed. Falling back to search with autocorrected ingredients.")
        plausible_ingredients = autocorrect_ingredients(user_ingredients, vocab)
        if sorted(plausible_ingredients) != sorted(normalize_ingredients(user_ingredients)):
             results = _find_similar_drinks_internal(plausible_ingredients, drinks_df)

    if results:
        top_drink_ingredients = results[0]['ingredients']
        final_prob = predict_drink(top_drink_ingredients, clf, vectorizer)
    else:
        final_prob = 0.0

    return {
        'probability': round(float(final_prob), 4),
        'similar_drinks': results
    }

def generate_negative_samples(drinks_df, n_samples=1000):
    all_ingredients = list(set(ing for ing_list in drinks_df['ingredients'] for ing in ing_list.split(', ')))
    negatives = set()
    existing_combinations = set(drinks_df['ingredients'])
    while len(negatives) < n_samples:
        n_ingredients = random.randint(2, 5)
        sample = ', '.join(sorted(random.sample(all_ingredients, n_ingredients)))
        if sample not in existing_combinations:
            negatives.add(sample)
    return list(negatives)
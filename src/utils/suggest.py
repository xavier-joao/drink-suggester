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
from sklearn.metrics.pairwise import cosine_similarity
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
    drinks_vec = X_vec[:len(positives)]
    all_ingredients_vocab = list(vectorizer.vocabulary_.keys())
    return clf, vectorizer, drinks_vec, drinks_df, all_ingredients_vocab

@lru_cache(maxsize=1)
def get_classifier():
    return train_classifier()

def expand_ingredients(ingredients, vocab):
    final_expanded_set = set()
    norm_ingredients = normalize_ingredients(ingredients)
    for user_ing in norm_ingredients:
        pattern = r'\b' + re.escape(user_ing) + r'\b'
        current_expansions = set()
        for vocab_ing in vocab:
            if re.search(pattern, vocab_ing):
                current_expansions.add(vocab_ing)
        if current_expansions:
            final_expanded_set.update(current_expansions)
        else:
            final_expanded_set.add(user_ing)
    logger.info(f"Expanded search terms from '{ingredients}' to '{list(final_expanded_set)}'")
    return list(final_expanded_set)

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

def _find_similar_drinks_internal(search_ingredients, original_ingredients, vectorizer, drinks_vec, drinks_df):
    norm_search_ingredients = normalize_ingredients(search_ingredients)
    input_str = ', '.join(sorted(norm_search_ingredients))
    if not norm_search_ingredients:
        return []

    input_vec = vectorizer.transform([input_str])
    cosine_sims = cosine_similarity(input_vec, drinks_vec).flatten()
    
    combined_scores = cosine_sims
    top_n = 20
    min_similarity = 0.1
    top_indices = np.argpartition(-combined_scores, min(top_n, len(combined_scores)-1))[:top_n]
    top_indices = top_indices[combined_scores[top_indices] >= min_similarity]
    top_indices = top_indices[np.argsort(-combined_scores[top_indices])]

    results = []
    for idx in top_indices:
        drink_ingredients = drinks_df.iloc[idx]['ingredients'].split(', ')
        ingredient_matches = []
        for user_ingr in normalize_ingredients(original_ingredients):
            best_match = max([(ingr, fuzz.WRatio(user_ingr, ingr)) for ingr in drink_ingredients], key=lambda x: x[1])
            if best_match[1] > 50:
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
    user_ingredients_str = ', '.join(sorted(norm_ingredients))
    if not user_ingredients_str:
        return 0.0
    user_vec = vectorizer.transform([user_ingredients_str])
    return clf.predict_proba(user_vec)[0][1]

def get_drink_recommendations(user_ingredients):
    clf, vectorizer, drinks_vec, drinks_df, vocab = get_classifier()
    
    plausible_ingredients = autocorrect_ingredients(user_ingredients, vocab)
    final_prob = predict_drink(plausible_ingredients, clf, vectorizer)
    
    search_ingredients = expand_ingredients(user_ingredients, vocab)
    results = _find_similar_drinks_internal(search_ingredients, user_ingredients, vectorizer, drinks_vec, drinks_df)
    
    if not results and sorted(plausible_ingredients) != sorted(normalize_ingredients(user_ingredients)):
        logger.warning("Broad search failed. Falling back to search with autocorrected ingredients.")
        results = _find_similar_drinks_internal(plausible_ingredients, plausible_ingredients, vectorizer, drinks_vec, drinks_df)

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
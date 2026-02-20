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

MIN_INGREDIENT_SCORE = 75
FINAL_SIMILARITY_THRESHOLD = 0.5
AUTOCORRECT_MIN_SCORE = 90

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

def get_drinks_dataframe():
    """Get drinks DataFrame from database."""
    from src.models import Drink
    try:
        drinks = Drink.query.all()
        if not drinks:
            logger.warning("No drinks found in database")
            return pd.DataFrame(columns=['name', 'ingredients'])
        data = []
        for drink in drinks:
            data.append({
                'name': drink.name,
                'ingredients': drink.ingredients
            })
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error querying database: {e}")
        return pd.DataFrame(columns=['name', 'ingredients'])

def train_classifier():
    drinks_df = get_drinks_dataframe()
    
    if drinks_df.empty:
        logger.warning("Empty drinks DataFrame, cannot train classifier")
        from sklearn.pipeline import Pipeline
        clf = LogisticRegression(max_iter=1000)
        vectorizer = CountVectorizer(tokenizer=lambda x: x.split(', '), token_pattern=None)
        return clf, vectorizer, drinks_df, []
    
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

def get_classifier():
    return train_classifier()

def autocorrect_ingredients(ingredients, vocab, threshold=50):
    norm_ingredients = normalize_ingredients(ingredients)
    corrected_ingredients = []
    
    if not vocab:
        logger.warning("Empty vocabulary, returning normalized ingredients")
        return norm_ingredients
    
    for ing in norm_ingredients:
        result = process.extractOne(ing, vocab, scorer=fuzz.WRatio)
        if result is None:
            corrected_ingredients.append(ing)
        else:
            best_match, score, _ = result
            if score >= threshold:
                corrected_ingredients.append(best_match)
            else:
                corrected_ingredients.append(ing)
    return corrected_ingredients

def _get_custom_score(str1, str2):
    wratio_score = fuzz.WRatio(str1, str2)
    len1, len2 = len(str1), len(str2)
    length_similarity = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0
    final_score = wratio_score * (length_similarity ** 0.5)
    return final_score

def _find_similar_drinks_internal(user_ingredients, drinks_df):
    norm_user_ingredients = normalize_ingredients(user_ingredients)
    if not norm_user_ingredients:
        return []

    all_results = []
    for index, row in drinks_df.iterrows():
        drink_ingredients = row['ingredients'].split(', ')
        total_score = 0
        num_matched = 0
        ingredient_matches = []

        for user_ingr in norm_user_ingredients:
            best_match = max(
                [(ingr, _get_custom_score(user_ingr, ingr)) for ingr in drink_ingredients],
                key=lambda x: x[1]
            )
            
            if best_match:
                matched_ingredient, score = best_match[0], best_match[1]
                if score >= MIN_INGREDIENT_SCORE:
                    total_score += score
                    num_matched += 1
                    ingredient_matches.append({
                        'user_ingredient': user_ingr,
                        'matched_ingredient': matched_ingredient,
                        'score': score
                    })

        if not norm_user_ingredients:
            continue

        if num_matched == 0:
            continue

        avg_score = total_score / num_matched
        match_ratio = num_matched / len(norm_user_ingredients)
        final_similarity = (avg_score / 100) * match_ratio

        if final_similarity >= FINAL_SIMILARITY_THRESHOLD:
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
    try:
        user_vec = vectorizer.transform([user_ingredients_str])
        return clf.predict_proba(user_vec)[0][1]
    except Exception as e:
        logger.warning(f"Error predicting drink probability: {e}, returning 0.0")
        return 0.0

def get_drink_recommendations(user_ingredients):
    clf, vectorizer, drinks_df, vocab = get_classifier()

    norm_user_ingredients = normalize_ingredients(user_ingredients)
    best_guess_ingredients = []
    corrections = []

    vocab_set = set(vocab) if vocab else set()

    if vocab:
        for orig in norm_user_ingredients:
            if orig in vocab_set:
                best_guess_ingredients.append(orig)
                continue

            res = process.extractOne(orig, vocab, scorer=fuzz.WRatio)
            if res:
                match, score, _ = res
                is_substring = orig in match or match in orig
                if (score >= AUTOCORRECT_MIN_SCORE
                        and match != orig
                        and match not in norm_user_ingredients
                        and not is_substring):
                    best_guess_ingredients.append(match)
                    corrections.append({'original': orig, 'corrected': match, 'score': int(score)})
                else:
                    best_guess_ingredients.append(orig)
            else:
                best_guess_ingredients.append(orig)
    else:
        best_guess_ingredients = list(norm_user_ingredients)

    logger.info(f"PROB CHECK: Using best-guess list for probability: {best_guess_ingredients}")
    final_prob = predict_drink(best_guess_ingredients, clf, vectorizer)

    results = _find_similar_drinks_internal(norm_user_ingredients, drinks_df)

    results_corrected = []
    if corrections:
        results_corrected = _find_similar_drinks_internal(best_guess_ingredients, drinks_df)

        corrected_map = {r['name']: r.get('ingredient_matches', []) for r in results_corrected}

        for res in results:
            corr_list = corrected_map.get(res['name'], [])
            for corr in corrections:
                orig = corr['original']
                corr_val = corr['corrected']
                for m in res.get('ingredient_matches', []):
                    if m.get('user_ingredient') == orig:
                        m['original_user_ingredient'] = orig
                        m['user_ingredient'] = corr_val

                for m in corr_list:
                    if m.get('user_ingredient') == corr_val:
                        exists = any((mm.get('matched_ingredient') == m.get('matched_ingredient') and mm.get('user_ingredient') == m.get('user_ingredient')) for mm in res.get('ingredient_matches', []))
                        if not exists:
                            appended = m.copy()
                            appended['original_user_ingredient'] = orig
                            res.setdefault('ingredient_matches', []).append(appended)

    if not results and results_corrected:
        logger.info("Direct search failed. Falling back to search using autocorrected ingredients.")
        for res in results_corrected:
            for match in res.get('ingredient_matches', []):
                try:
                    idx = next((i for i, v in enumerate(best_guess_ingredients) if v == match['user_ingredient']), None)
                    if idx is not None:
                        match['user_ingredient'] = norm_user_ingredients[idx]
                except Exception:
                    pass
        results = results_corrected

    return {
        'probability': round(float(final_prob), 4),
        'similar_drinks': results,
        'best_guess': best_guess_ingredients,
        'corrections': corrections
    }

def get_exact_match_drinks(user_ingredients):
    """Find drinks using ONLY your available ingredients.
    
    Shows:
    - Drinks using ALL your ingredients (exact match, highest priority)
    - Drinks using some of your ingredients (e.g., vodka + lime juice but no sugar syrup)
    - Drinks using just one ingredient
    
    Does NOT show drinks that require ingredients you don't have.
    """
    norm_user_ingredients = normalize_ingredients(user_ingredients)
    user_ing_set = set(norm_user_ingredients)
    
    _, _, drinks_df, _ = get_classifier()
    
    if drinks_df.empty:
        return {
            'probability': 0.0,
            'similar_drinks': []
        }
    
    exact_matches = []
    
    for index, row in drinks_df.iterrows():
        drink_ingredients = row['ingredients'].split(', ')
        norm_drink_ingredients = normalize_ingredients(drink_ingredients)
        drink_ing_set = set(norm_drink_ingredients)
        if drink_ing_set.issubset(user_ing_set) and drink_ing_set:
            match_count = len(drink_ing_set)
            match_ratio = match_count / len(user_ing_set)
            
            exact_matches.append({
                'name': row['name'],
                'ingredients': drink_ingredients,
                'similarity': round(match_ratio, 4),
                'ingredient_matches': []
            })
    
    exact_matches.sort(key=lambda x: x['similarity'], reverse=True)
    
    return {
        'probability': 0.0,  # No probability calculation for exact matches
        'similar_drinks': exact_matches
    }

def suggest_next_ingredient(current_ingredients, top_n=5):
    """Given the current ingredient list, suggest the best next ingredient to add.

    For each candidate ingredient in the vocabulary (not already selected),
    scores the combined list using the classifier and also finds which drinks
    would match, with ingredient-level match details.

    Returns a list of dicts:
      {
        'ingredient': str,
        'score': float,          # classifier probability for current + candidate
        'matching_drinks': [     # drinks that match current + candidate
          {
            'name': str,
            'ingredients': [str],
            'similarity': float,
            'ingredient_matches': [...]
          }
        ]
      }
    """
    clf, vectorizer, drinks_df, vocab = get_classifier()

    norm_current = normalize_ingredients(current_ingredients)
    current_set = set(norm_current)

    if not vocab:
        return []

    current_matches = _find_similar_drinks_internal(norm_current, drinks_df)
    current_match_names = {d['name'] for d in current_matches}

    candidates = []
    for ingredient in vocab:
        if ingredient in current_set:
            continue

        test_ingredients = norm_current + [ingredient]
        score = predict_drink(test_ingredients, clf, vectorizer)

        # Among drinks that already match the current ingredients,
        # keep only those that also fuzzy-match the candidate ingredient.
        all_matching = _find_similar_drinks_internal(test_ingredients, drinks_df)
        matching_drinks = [
            d for d in all_matching
            if d['name'] in current_match_names
            and any(
                m.get('user_ingredient') == ingredient
                for m in d.get('ingredient_matches', [])
            )
        ]

        candidates.append({
            'ingredient': ingredient,
            'score': round(float(score), 4),
            'matching_drinks': matching_drinks
        })

    # Only return candidates that have real matching drinks — the ML score alone
    # can rank ingredients highly even when no actual drink in the database
    # combines them with the user's current selection.
    candidates = [c for c in candidates if c['matching_drinks']]
    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates[:top_n]


def generate_negative_samples(drinks_df, n_samples=1000):
    if drinks_df.empty:
        return []
    
    all_ingredients = list(set(ing for ing_list in drinks_df['ingredients'] for ing in ing_list.split(', ')))
    
    if not all_ingredients:
        return []
    
    negatives = set()
    existing_combinations = set(drinks_df['ingredients'])
    max_attempts = n_samples * 10  # Prevent infinite loop
    attempts = 0
    
    while len(negatives) < n_samples and attempts < max_attempts:
        attempts += 1
        n_ingredients = min(random.randint(2, 5), len(all_ingredients))
        try:
            sample = ', '.join(sorted(random.sample(all_ingredients, n_ingredients)))
            if sample not in existing_combinations:
                negatives.add(sample)
        except ValueError:
            # Not enough ingredients to sample from
            break
    
    return list(negatives)
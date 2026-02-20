import pandas as pd
import random
import re
import unicodedata
import json
import os
import numpy as np
import logging
from rapidfuzz import fuzz, process, utils
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

MIN_INGREDIENT_SCORE = 80
FINAL_SIMILARITY_THRESHOLD = 0.5
AUTOCORRECT_THRESHOLD = 80

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class DrinkRecommender:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DrinkRecommender, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return
        
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.flavor_map = {}
        self.drinks_data = []
        self.clf = None
        self.vectorizer = None
        self.all_ingredients = set()
        self.inverted_index = {}
        self.vocab_list = []
        self.initialized = False

    def ensure_loaded(self):
        if self.initialized:
            return

        self.flavor_map = self._load_flavor_map()
        self.drinks_data = self._load_drinks_from_db()
        self.clf, self.vectorizer = self._train_classifier(self.drinks_data)
        
        self.all_ingredients = set()
        self.inverted_index = {}
        
        for idx, drink in enumerate(self.drinks_data):
            norm_ingredients = [self.normalize_text(i) for i in drink['ingredients']]
            drink['norm_ingredients'] = norm_ingredients
            drink['norm_set'] = set(norm_ingredients)
            
            for ing in norm_ingredients:
                self.all_ingredients.add(ing)
                if ing not in self.inverted_index:
                    self.inverted_index[ing] = set()
                self.inverted_index[ing].add(idx)
        
        self.vocab_list = sorted(list(self.all_ingredients))
        self.initialized = True

    def _load_flavor_map(self):
        json_path = os.path.join(self.script_dir, '..', 'database', 'flavor_map.json')
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def _load_drinks_from_db(self):
        from src.models import Drink
        try:
            drinks = Drink.query.all()
            if not drinks:
                return []
            return [{'name': d.name, 'ingredients': d.ingredients.split(', ')} for d in drinks]
        except Exception:
            return []

    def _train_classifier(self, drinks_data):
        if not drinks_data:
            return LogisticRegression(), CountVectorizer()
        
        positives = [', '.join(d['ingredients']) for d in drinks_data]
        all_ings = list(set(ing for d in drinks_data for ing in d['ingredients']))
        negatives = []
        existing_combos = set(positives)
        
        target_neg = len(positives)
        attempts = 0
        while len(negatives) < target_neg and attempts < target_neg * 5:
            attempts += 1
            k = random.randint(2, 5)
            sample = ', '.join(sorted(random.sample(all_ings, min(k, len(all_ings)))))
            if sample not in existing_combos:
                negatives.append(sample)
        
        X = positives + negatives
        y = [1] * len(positives) + [0] * len(negatives)
        
        vectorizer = CountVectorizer(tokenizer=lambda x: x.split(', '), token_pattern=None)
        X_vec = vectorizer.fit_transform(X)
        clf = LogisticRegression(max_iter=500, C=1.0, solver='liblinear')
        clf.fit(X_vec, y)
        return clf, vectorizer

    def normalize_text(self, text):
        if not isinstance(text, str): return ""
        text = re.sub(r'\s*\([^)]*\)', '', text).strip()
        nfkd = unicodedata.normalize('NFKD', text)
        return ''.join([c for c in nfkd if not unicodedata.combining(c)]).lower()

    def predict_prob(self, ingredients):
        self.ensure_loaded()
        if not ingredients: return 0.0
        try:
            guess_str = ', '.join(sorted(ingredients))
            vec = self.vectorizer.transform([guess_str])
            return self.clf.predict_proba(vec)[0][1]
        except Exception:
            return 0.0

engine = DrinkRecommender()

def normalize_ingredients(ingredients):
    return [engine.normalize_text(i) for i in ingredients]

def autocorrect_input(user_ingredients):
    engine.ensure_loaded()
    norm_user_ingredients = normalize_ingredients(user_ingredients)
    best_guess = []
    corrections = []
    
    vocab = engine.vocab_list
    vocab_set = engine.all_ingredients
    
    for orig in norm_user_ingredients:
        if orig in vocab_set:
            best_guess.append(orig)
            continue
            
        res = process.extractOne(orig, vocab, scorer=fuzz.ratio, score_cutoff=AUTOCORRECT_THRESHOLD)
        
        if res:
            match, score, _ = res
            if match != orig and match not in best_guess:
                best_guess.append(match)
                corrections.append({'original': orig, 'corrected': match, 'score': score})
            else:
                best_guess.append(orig)
        else:
            best_guess.append(orig)
            
    return best_guess, corrections

def calculate_drink_matches(search_ingredients, user_inputs_map=None):
    engine.ensure_loaded()
    if not search_ingredients:
        return []

    candidate_indices = set()
    expanded_search_ingredients = set(search_ingredients)
    
    for db_ing in engine.vocab_list:
        for search_ing in search_ingredients:
            if search_ing in db_ing or db_ing in search_ing: 
                 if fuzz.ratio(search_ing, db_ing) > 60:
                    expanded_search_ingredients.add(db_ing)
                    if db_ing in engine.inverted_index:
                        candidate_indices.update(engine.inverted_index[db_ing])

    if not candidate_indices:
        return []

    results = []
    
    for idx in candidate_indices:
        drink = engine.drinks_data[idx]
        drink_ingredients = drink['norm_set']
        
        matches = list(drink_ingredients.intersection(expanded_search_ingredients))
        if not matches:
            continue
            
        num_matched = len(matches)
        match_details = []
        total_score = 0
        
        for matched_ing in matches:
            best_u_ing = None
            best_u_score = 0
            
            for u_ing in search_ingredients:
                s = fuzz.ratio(u_ing, matched_ing)
                if s > best_u_score:
                    best_u_score = s
                    best_u_ing = u_ing
            
            original_text = None
            if user_inputs_map and best_u_ing in user_inputs_map:
                original_text = user_inputs_map[best_u_ing]

            match_details.append({
                'user_ingredient': best_u_ing,
                'matched_ingredient': matched_matched_ing_display_name(idx, matched_ing),
                'original_user_ingredient': original_text,
                'score': best_u_score
            })
            total_score += best_u_score

        avg_score = total_score / num_matched if num_matched > 0 else 0
        match_ratio = num_matched / len(search_ingredients)
        final_similarity = (avg_score / 100) * match_ratio

        if final_similarity >= FINAL_SIMILARITY_THRESHOLD:
            # --- NEW: Calculate Flavors for this drink ---
            flavors = flavor_probabilities(drink['ingredients'])
            
            results.append({
                'name': drink['name'],
                'ingredients': drink['ingredients'],
                'similarity': round(final_similarity, 4),
                'ingredient_matches': match_details,
                'flavors': flavors # <--- Added this line
            })

    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results[:20]

def matched_matched_ing_display_name(drink_idx, norm_name):
    engine.ensure_loaded()
    drink = engine.drinks_data[drink_idx]
    for raw_ing in drink['ingredients']:
        if engine.normalize_text(raw_ing) == norm_name:
            return raw_ing
    return norm_name

def get_drink_recommendations(user_ingredients):
    engine.ensure_loaded()
    best_guess, corrections = autocorrect_input(user_ingredients)
    correction_map = {c['corrected']: c['original'] for c in corrections}
    prob = engine.predict_prob(best_guess)
    results = calculate_drink_matches(best_guess, correction_map)
    
    return {
        'probability': round(float(prob), 4),
        'similar_drinks': results,
        'best_guess': best_guess,
        'corrections': corrections
    }

def get_exact_match_drinks(user_ingredients):
    engine.ensure_loaded()
    best_guess, _ = autocorrect_input(user_ingredients)
    user_ing_set = set(best_guess)
    
    if not user_ing_set:
        return {'probability': 0.0, 'similar_drinks': []}

    exact_matches = []
    
    for idx, drink in enumerate(engine.drinks_data):
        drink_ing_set = drink['norm_set']
        
        if drink_ing_set and drink_ing_set.issubset(user_ing_set):
            
            flavors = flavor_probabilities(drink['ingredients'])
            
            matches = []
            for ing in drink_ing_set:
                matches.append({
                    'user_ingredient': ing,
                    'matched_ingredient': matched_matched_ing_display_name(idx, ing),
                    'score': 100,
                    'original_user_ingredient': None
                })

            match_ratio = len(drink_ing_set) / len(user_ing_set)
            
            exact_matches.append({
                'name': drink['name'],
                'ingredients': drink['ingredients'],
                'similarity': round(match_ratio, 4),
                'ingredient_matches': matches,
                'flavors': flavors 
            })
            
    exact_matches.sort(key=lambda x: len(x['ingredients']), reverse=True)
    
    return {'probability': 0.0, 'similar_drinks': exact_matches}

def suggest_next_ingredient(current_ingredients, top_n=5):
    engine.ensure_loaded()
    best_guess, _ = autocorrect_input(current_ingredients)
    current_set = set(best_guess)

    compatible_indices = []
    for idx, drink in enumerate(engine.drinks_data):
        if current_set.issubset(drink['norm_set']):
            compatible_indices.append(idx)

    if not compatible_indices:
        for idx, drink in enumerate(engine.drinks_data):
            if not current_set.isdisjoint(drink['norm_set']):
                compatible_indices.append(idx)

    ingredient_counts = {}
    for idx in compatible_indices:
        drink = engine.drinks_data[idx]
        missing = drink['norm_set'] - current_set
        for ing in missing:
            if ing not in ingredient_counts:
                ingredient_counts[ing] = {'count': 0, 'indices': []}
            ingredient_counts[ing]['count'] += 1
            ingredient_counts[ing]['indices'].append(idx)

    sorted_suggestions = sorted(ingredient_counts.items(), key=lambda x: x[1]['count'], reverse=True)

    results = []
    for norm_ing, data in sorted_suggestions[:top_n]:
        display_name = norm_ing
        if data['indices']:
            display_name = matched_matched_ing_display_name(data['indices'][0], norm_ing)

        example_drinks = []
        for idx in data['indices'][:5]:
            d = engine.drinks_data[idx]
            flavors = flavor_probabilities(d['ingredients'])
            
            matches = []
            for existing in current_set:
                matches.append({
                    'user_ingredient': existing,
                    'matched_ingredient': matched_matched_ing_display_name(idx, existing),
                    'score': 100
                })
            
            matches.append({
                'user_ingredient': display_name,
                'matched_ingredient': matched_matched_ing_display_name(idx, norm_ing),
                'score': 100,
                'original_user_ingredient': "(Suggested)"
            })

            example_drinks.append({
                'name': d['name'],
                'ingredients': d['ingredients'],
                'similarity': 1.0,
                'ingredient_matches': matches,
                'flavors': flavors
            })

        total_found = len(compatible_indices)
        score = data['count'] / total_found if total_found > 0 else 0

        results.append({
            'ingredient': display_name,
            'score': round(score, 2),
            'matching_drinks': example_drinks
        })

    return results

def flavor_probabilities(ingredients):
    engine.ensure_loaded()
    norm_ingredients = normalize_ingredients(ingredients)
    flavor_counts = {flavor: 0 for flavor in engine.flavor_map}
    for ingr in norm_ingredients:
        for flavor, flavor_ings in engine.flavor_map.items():
            if any(f in ingr for f in flavor_ings):
                flavor_counts[flavor] += 1
    total = sum(flavor_counts.values())
    if total == 0:
        return {flavor: 0.0 for flavor in engine.flavor_map}
    return {flavor: round(count / total, 3) for flavor, count in flavor_counts.items()}

def predict_drink(ingredients, clf=None, vectorizer=None):
    return engine.predict_prob(ingredients)
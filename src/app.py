from flask import Flask, request, jsonify, render_template
import logging
import os
import pandas as pd

from src.models import db, Drink
from src.utils.suggest import (
    get_drink_recommendations,
    flavor_probabilities,
    get_exact_match_drinks,
    suggest_next_ingredient
)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
instance_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'instance')
os.makedirs(instance_path, exist_ok=True)
db_path = os.path.join(instance_path, 'drinks.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

_db_initialized = False

def init_db():
    """Initialize database and load drinks from CSV if empty."""
    global _db_initialized
    if _db_initialized:
        return
    
    with app.app_context():
        try:
            db.create_all()
            
            # Check if database has drinks
            drink_count = Drink.query.count()
            if drink_count > 0:
                logger.info(f"Database already populated with {drink_count} drinks")
                _db_initialized = True
                return
            
            logger.info("Database is empty. Loading drinks from CSV...")
            script_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(script_dir, 'database', 'drinks_list.csv')
            
            logger.info(f"CSV path: {csv_path}")
            logger.info(f"CSV exists: {os.path.exists(csv_path)}")
            
            if not os.path.exists(csv_path):
                logger.error(f"CSV file not found at {csv_path}")
                _db_initialized = True
                return
            
            drinks_df = pd.read_csv(csv_path)
            logger.info(f"CSV loaded: {len(drinks_df)} drinks ready to import")
            
            if drinks_df.empty:
                logger.warning("CSV file is empty")
                _db_initialized = True
                return
            
            # Batch insert for better performance
            batch_size = 100
            for i in range(0, len(drinks_df), batch_size):
                batch = drinks_df.iloc[i:i+batch_size]
                for _, row in batch.iterrows():
                    drink = Drink(
                        name=row['name'],
                        ingredients=row['ingredients']
                    )
                    db.session.add(drink)
                db.session.commit()
                logger.info(f"Inserted batch {i//batch_size + 1} ({min(i+batch_size, len(drinks_df))}/{len(drinks_df)} drinks)")
            
            final_count = Drink.query.count()
            logger.info(f"Successfully loaded all {final_count} drinks into database")
            
        except Exception as e:
            logger.error(f"Error during database initialization: {e}", exc_info=True)
            db.session.rollback()
        finally:
            _db_initialized = True

@app.before_request
def before_request():
    """Initialize database on first request."""
    init_db()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/suggest', methods=['POST'])
def suggest():
    data = request.get_json()
    ingredients = data.get('ingredients', [])
    exact_match = data.get('exact_match', False)
    
    if not ingredients:
        return jsonify({'error': 'No ingredients provided'}), 400
    
    if exact_match:
        # Exact match mode: only show drinks with exactly these ingredients
        response_data = get_exact_match_drinks(ingredients)
    else:
        # Fuzzy match mode: show similar drinks (default)
        response_data = get_drink_recommendations(ingredients)

    for drink in response_data['similar_drinks']:
        drink['flavor_profile'] = flavor_probabilities(drink['ingredients'])

    return jsonify(response_data)

@app.route('/suggest_next', methods=['POST'])
def suggest_next():
    data = request.get_json()
    ingredients = data.get('ingredients', [])
    top_n = data.get('top_n', 5)

    if not ingredients:
        return jsonify({'error': 'No ingredients provided'}), 400

    suggestions = suggest_next_ingredient(ingredients, top_n=top_n)

    # Attach flavor profile for each drink inside each suggestion
    for suggestion in suggestions:
        for drink in suggestion.get('matching_drinks', []):
            drink['flavor_profile'] = flavor_probabilities(drink['ingredients'])

    return jsonify({'suggestions': suggestions})

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000)
from flask import Flask, request, jsonify, render_template
import logging

from src.utils.suggest import (
    get_drink_recommendations,
    flavor_probabilities,
    get_all_ingredients
)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    final_result = None
    all_ingredients = get_all_ingredients()
    
    if request.method == 'POST':
        # Handle both comma-separated string (legacy/text input) and multiple select values
        ingredients = request.form.getlist('ingredients')
        
        # If we got a single item that contains commas, split it (legacy behavior)
        if len(ingredients) == 1 and ',' in ingredients[0]:
             ingredients = [i.strip() for i in ingredients[0].split(',') if i.strip()]
        
        # Filter out empty strings
        ingredients = [i for i in ingredients if i.strip()]
        
        if ingredients:
            response_data = get_drink_recommendations(ingredients)
            
            for drink in response_data['similar_drinks']:
                drink['flavor_profile'] = flavor_probabilities(drink['ingredients'])

            final_result = {
                'probability': response_data['probability'],
                'similar_drinks': response_data['similar_drinks'],
                'searched_ingredients': ingredients
            }
    
    return render_template('index.html', result=final_result, available_ingredients=all_ingredients)

@app.route('/suggest', methods=['POST'])
def suggest():
    data = request.get_json()
    ingredients = data.get('ingredients', [])
    
    if not ingredients:
        return jsonify({'error': 'No ingredients provided'}), 400
    
    response_data = get_drink_recommendations(ingredients)

    for drink in response_data['similar_drinks']:
        drink['flavor_profile'] = flavor_probabilities(drink['ingredients'])

    return jsonify(response_data)

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000)  # Uncomment for production use
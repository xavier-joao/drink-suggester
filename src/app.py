from flask import Flask, request, jsonify, render_template
import logging

from src.utils.suggest import (
    get_drink_recommendations,
    flavor_probabilities
)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    final_result = None
    if request.method == 'POST':
        ingredients_str = request.form.get('ingredients', '')
        ingredients = [i.strip() for i in ingredients_str.split(',') if i.strip()]
        
        if ingredients:
            response_data = get_drink_recommendations(ingredients)
            
            for drink in response_data['similar_drinks']:
                drink['flavor_profile'] = flavor_probabilities(drink['ingredients'])

            final_result = {
                'probability': response_data['probability'],
                'similar_drinks': response_data['similar_drinks'],
                'searched_ingredients': ingredients
            }
    
    return render_template('index.html', result=final_result)

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
from flask import Flask, request, jsonify, render_template

from src.utils.suggest import (
    train_classifier, 
    predict_drink, 
    find_similar_drinks,
    flavor_probabilities
)

app = Flask(__name__)

clf, vectorizer, drinks_vec, drinks_df = train_classifier()

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        ingredients_str = request.form.get('ingredients', '')
        ingredients = [i.strip() for i in ingredients_str.split(',') if i.strip()]
        
        if ingredients:
            prob = predict_drink(ingredients, clf, vectorizer)
            similar_drinks = find_similar_drinks(
                ingredients, vectorizer, drinks_vec, drinks_df, top_n=10, min_similarity=0.3
            )
            
            for drink in similar_drinks:
                drink['flavor_profile'] = flavor_probabilities(drink['ingredients'])

            result = {
                'probability': round(float(prob), 4),
                'similar_drinks': similar_drinks
            }
    return render_template('index.html', result=result)

@app.route('/suggest', methods=['POST'])
def suggest():
    data = request.get_json()
    ingredients = data.get('ingredients', [])
    
    if not ingredients:
        return jsonify({'error': 'No ingredients provided'}), 400
    
    prob = predict_drink(ingredients, clf, vectorizer)
    
    similar_drinks = find_similar_drinks(
        ingredients,
        vectorizer,
        drinks_vec,
        drinks_df,
        top_n=5,
        min_similarity=0.3
    )

    for drink in similar_drinks:
        drink['flavor_profile'] = flavor_probabilities(drink['ingredients'])

    return jsonify({
        'probability': float(prob),  
        'similar_drinks': similar_drinks
    })

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, request, jsonify
from src.utils.suggest import train_classifier, predict_drink, find_similar_drinks

app = Flask(__name__)

clf, vectorizer, drinks_vec, drinks_df = train_classifier()

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
    
    return jsonify({
        'probability': float(prob),  
        'similar_drinks': similar_drinks
    })

if __name__ == '__main__':
    app.run(debug=True)
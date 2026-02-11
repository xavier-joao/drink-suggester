from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Drink(db.Model):
    __tablename__ = 'drinks'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), unique=True, nullable=False, index=True)
    ingredients = db.Column(db.Text, nullable=False)  # Stored as comma-separated string
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'ingredients': [ing.strip() for ing in self.ingredients.split(',')]
        }

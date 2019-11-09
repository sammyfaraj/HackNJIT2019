from flask import Flask, jsonify, request, Response
from flask_cors import CORS, cross_origin
import numpy as np
from better_recipe_generator import Generator


app = Flask(__name__)
cors = CORS(app, resources={r"/foo": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'


# Load model
checkpoint_dir = './model_data/'
recipe_generator = Generator(checkpoint_dir)


# Initialize the Flask application
app = Flask(__name__)

@app.route('/get_recipe', methods=['POST'])
def get_recipes():
    try:
        data = request.json
        recipe = recipe_generator.predict(data['ingredients'])
        return {'recipe': recipe}
    except:
        return {'error': 'error'}

if __name__ == '__main__':
    app.run(host='0.0.0.0')

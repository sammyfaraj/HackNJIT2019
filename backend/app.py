from flask import Flask, jsonify, request, Response
from flask_cors import CORS, cross_origin
import numpy as np
from better_recipe_generator import Generator
import json
import csv
import os
# for clarafai
from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage
import requests

# Load system variables with dotenv
from dotenv import load_dotenv
load_dotenv()

uniques = []
csv_reader = []

app = Flask(__name__)
cors = CORS(app, resources={r"/foo": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

with open('ingredients.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            #print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            #print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
            line_count += 1
        uniques.append(row[0])
    print('Processed ' + str(line_count) + ' lines.')


# Load model
checkpoint_dir = './model_data/'
recipe_generator = Generator(checkpoint_dir)


# Initialize the Flask application
app = Flask(__name__)

@app.route('/get_recipe', methods=['POST'])
def get_recipes():
    try:
        data = request.json
        print(str(data['ingredients']))
        recipe = recipe_generator.predict(str(data['ingredients']), temperature=None)
        return {'recipe': recipe}
    except Exception as e:
        return {'error': e}


@app.route("/ingredients")                   # at the end point /
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])
def hello():                      # call method hello
    
    dairyList = [
        'eggs', 'egg whites', 'swiss cheese', 'mozzarella', 'greek yogurt', 'cheddar cheese', 'milk',
        'cream', 'butter', 'alfredo sauce', 'ice cream', 'french onion dip'
        ]
    vegeList = [
        'broccoli', 'celery', 'tomato', 'spinach', 'asparagus', 'lettuce', 'potato', 'garlic', 'carrot',
        'onion', 'eggplant', 'kale', 'cucumber', 'peas', 'radish'
        ]
    fruitList = [
        'apple', 'pear', 'kiwi', 'orange', 'banana', 'avocado', 'mango', 'coconut', 'fig', 'lemon',
        'lime', 'peach', 'watermelon', 'blueberry', 'blackberry', 'cherry', 'apricot'
    ]
    grainList = [
        'oats', 'barley', 'rice', 'white rice', 'brown rice', 'quinoa', 'pasta', 'sphagetti', 'couscous',
        'millet', 'buckwheat', 'buckyeet', 'corn', 'soybeans', 'wheat'
    ]
    proteinList = [
        'chicken', 'beef', 'shrimp', 'fish', 'salmon', 'tuna', 'burger', 'beef burger', 'chicken wings',
        'chicke thighs', 'beef ribs', 'pork', 'ham', 'cold cuts', 'chicken breast'
    ]

    dairyList.sort()
    vegeList.sort()
    fruitList.sort()
    grainList.sort()
    proteinList.sort()

    #print(uniques)
    obj = {
        'ingredients': uniques,
        'dairy': dairyList,
        'vegetables': vegeList,
        'fruits': fruitList,
        'grains': grainList,
        'proteins': proteinList
    }

    return json.dumps(obj)

@app.route("/classifyImage", methods=['POST']))
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])
def classify():

    badLabels = ['sandwich', 'vegetable', 'meat', 'sauce', 'citrus', 'sweet', 'cocktail', 'juice',
        'pasture', 'aliment', 'condiment']

    capp = ClarifaiApp(api_key=os.environ['CLARIFAI_API_KEY'])
    model = capp.models.get('food-items-v1.0')
    
    base64passed = json.loads(request.data)['base64']
    image2 = capp.inputs.create_image_from_base64(base64_bytes=base64passed)
    
    #image = ClImage(url='https://samples.clarifai.com/food.jpg')
    #resp = model.predict([image])
    resp2 = model.predict([image2])

    #print(resp2['outputs'][0]['data']['concepts'])

    classifications = resp2['outputs'][0]['data']['concepts']

    respList = []

    for item in classifications:
        if item['value'] > .50 and item['name'] not in badLabels:
            respList.append(item['name'])
    
    return json.dumps({'items':respList})


'''
 ---------------------   MS AZURE BELOW   ---------------------
'''

@app.route("/shoppingList", methods=['POST'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])
def msAzureImageRecog():
    print('Classifying shopping list with msAzure')
    endpoint = 'https://hacknjitvision1.cognitiveservices.azure.com/vision/v2.0/recognizeText?mode=Handwritten'
    URL = json.loads(request.data)['url']
    URL = unicode(URL, "utf-8")
    print('url:')
    print(URL)
    PARAMS = {}
    HEADERS = {
        'content-type': 'application/json',
        'Ocp-Apim-Subscription-Key': 'efd3cdebf4da478e98e8c17b5364224c'
    }
    r = requests.post(url = URL, params = PARAMS, headers = HEADERS)
    print(r.text)
    return json.dumps({})

'''
 ---------------------   MS AZURE END   ---------------------
'''

if __name__ == '__main__':
    recipe_generator.predict("['rice']")
    app.run(host='0.0.0.0', port=80)

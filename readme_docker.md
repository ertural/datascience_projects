## Explaining how to run your code using Docker

We can deploy our model by following the steps below:

1. First of all we can wrap our model with a restful web service in order to make it accessible. For that purpose we can use Flask. 
2. Then we can containerize the web service with docker. This enables the fault tolerance by allowing us to spin up new containers very quickly should one break down.
3. We can now deploy our model into a kubernetes cluster.
4. Finally we should save the model in a google cloud storage so we can easily replace older models with newer versions.

### 1. Flask 
Flask is a microframework for python web development. It allows us to use what we need for our application. In this case we’ll be using flask to built the restful web service. Below is a simple flask application that loads our model into memory, makes predictions with our model and returns predictions a JSON object.

```from flask import Flask, request, jsonify
import utils
app = Flask(__name__)
model = utils.load_model("<path_to_model>")
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.data['input']
        
        model_input = utils.preprocessing(data)
        results = model.predict(model_input)
    return jsonify({"prediction": results})
```

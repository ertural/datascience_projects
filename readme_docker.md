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

### 2.Containerizing web service with Docker
Docker is an application that allows us to package the application in a container with all dependencies so we can run the applications anywhere. The main advantage of the docker is to ensure that our application works the way we want it to work and ensure scalability by starting up new containers as we get more requests, and maintaining fault tolerance by being able to replace faulty containers with new ones in no time. We can dockerize your application by adding a Dockerfile to your app folder. 

```
# Pull Base Image
FROM ubuntu:16.04
# copy code into image and set as working directory
COPY . /application
WORKDIR /application
# install dependencies
RUN sudo apt-get -y update && \
    pip install pipenv && \
    pipenv install --system --deploy
EXPOSE 5000
ENTRYPONIT ["gunicorn"]
CMD ["server:app"]
```
### 3.Deploy using Kubernetes
No we can deploy our image anymore. We can use a service on Google Cloud Platform that allows us to deploy and manage containers called Kubernetes.

1. First we have to push docker image to google container registry
```gcloud docker --push gcr.io/<your-project-id>/<image-name>
```
2. Create a container cluster on google cloud platform
```gcloud container clusters create <cluster-name> --num-nodes=3
```
3. Run app image inside the cluster we just created
```kubectl run <deployment_name> --image=<your_image_in_container_registry> --port 8080
```
4. Expose application to the internet
```kubectl expose deployment <deployment_name> --type=LoadBalancer --port 80 --target-port 8080
```
Now anymore we are able to build our model and wrap a web service around it, containerized it and deployed it with kubernetes!

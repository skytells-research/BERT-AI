from flask import Flask,request,jsonify
from flask_cors import CORS
from bert import QA

app = Flask(__name__)
CORS(app)

@app.route("/predict",methods=['POST', 'GET'])
def predict():
    document = request.json["document"]
    question = request.json["question"]
    try:
     
        out = QA.getAnswer(question, document)
        return jsonify(out)
    except Exception as e:
        print(e)
        return jsonify({"result":"Model Failed"})

@app.route("/generate",methods=['POST', 'GET'])
def generate():
    context = request.json["context"]
    
    max_length = 50
    do_sample = False
    if max_length in request.json:
        max_length = request.json["max_length"]
    else:
        max_length = 50
    if do_sample in request.json:
        do_sample = request.json["do_sample"]
    else:
        do_sample = False
    try:
        out = QA.generateText(context, max_length, do_sample)
        return jsonify(out)
    except Exception as e:
        print(e)
        return jsonify({"result":"Model Failed"})


if __name__ == "__main__":
    app.run('0.0.0.0',port=8000)

import os
from flask import Flask,request,jsonify,render_template
from flask_cors import CORS
from bert import QA
import gc

app = Flask(__name__, template_folder='views')
CORS(app)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template("prediction.html")


@app.route("/predict",methods=['POST', 'GET'])
def predict():
    gc.collect()
    document = request.json["document"]
    question = request.json["question"]
    try:
        out = QA.getAnswer(question, document)
        gc.collect()
        return jsonify(out)
    except Exception as e:
        print(e)
        return jsonify({"result":"Model Failed"})

@app.route("/generate",methods=['POST', 'GET'])
def generate():
    context = request.json["context"]
    
    max_length = 20
    do_sample = False
    if max_length in request.json:
        max_length = request.json["max_length"]
    else:
        max_length = 20
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
    port = int(os.environ.get("PORT", 5000))
    app.run('0.0.0.0',port=port)

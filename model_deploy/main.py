import pickle
import re
from flask import Flask, request, jsonify, escape
# from summarizer_model import summarize_text
from transformers import pipeline


app = Flask('app')

@app.route("/")
def index():
    text = request.args.get("text", "")
    if text:
        text = cleaning(text)
        summary_cnn = summarize_cnn(text)
        summary_xsum = summarize_xsum(text)
    else:
        summary_cnn = ""
        summary_xsum = ""

    return (
        """<form action="" method="get">
                <b>Enter text to summarize:</b> <br> <input style="width:600px" type="text" id="tbuser" name="text">
                <input type="submit" value="Summarize"><br><br>
                <textarea style="font-size: 9pt" rows="5" cols="80" id="TITLE"></textarea>
            </form>
            """
        + "<b>Summary using BART-cnn:</b> <br>" + summary_cnn + '<br><br>' + "<b>Summary using BART-xsum:</b> <br>" + summary_xsum)

def cleaning(text):
    new = re.sub(r'[,.;@#?!&$\'\"]+', '', text, flags=re.IGNORECASE)
    new = re.sub(r'[^a-zA-Z0-9]', " ", new, flags=re.VERBOSE)
    new = " ".join(wd for wd in new.split())
    new = re.sub("\n|\r", "", new)
    return new

def summarize_cnn(text):
    if len(text.split())<60:
       maxl = len(text.split())
    elif len(text.split())>900:
       text = ' '.join(text.split()[0:900])
       maxl = 120
    else:
       maxl = 120

    with open('./model-cnn.bin', 'rb') as f_in:
        model = pickle.load(f_in)
        f_in.close()
    try:
        summary = model(text, max_length=maxl, min_length=1)[0]['summary_text']
        return summary
    except ValueError:
        return "invalid input"

def summarize_xsum(text):
    if len(text.split())<60:
       maxl = len(text.split())
    elif len(text.split())>900:
       text = ' '.join(text.split()[0:900])
       maxl = 60
    else:
       maxl = 60

    with open('./model-xsum.bin', 'rb') as f_in:
        model = pickle.load(f_in)
        f_in.close()
    try:
        summary = model(text, max_length=maxl, min_length=1)[0]['summary_text']
        return summary
    except ValueError:
        return "invalid input"


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
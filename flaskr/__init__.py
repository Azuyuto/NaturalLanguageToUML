import os
import pickle

import joblib
from flask import Flask, render_template, request, jsonify
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

import numpy as np
import re

current_output = ''

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.static_folder = 'static'
    app.debug = True
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # a simple page that says hello
    @app.route('/hello')
    def hello():
        return 'Hello, World!'

    @app.route('/get_output')
    def get_output():
        global current_output
        print(current_output)
        return jsonify(text=current_output)

    @app.route('/', methods=['GET', 'POST'])
    def main():
        global current_output
        # If a form is submitted
        if request.method == "POST":
            current_output = ''

            model = load_model('flaskr/my_model7.h5')
            input_text = request.form.get("text")

            prediction = text_to_diagram(model, input_text)
            prediction = "flowchart TD\n" + prediction
            prediction = prediction.replace(" [", "[")
            prediction = prediction.replace("\\?", "?")
            prediction = prediction.replace("- -", "--")
            print(prediction)
        else:
            prediction = ""

        return render_template("website.html", output=prediction)

    return app




def text_to_diagram(model, input_text):
    global current_output
    with open('flaskr/tokenizers_and_info.pickle', 'rb') as f:
        input_tokenizer, output_tokenizer, max_input_seq_len, max_output_seq_len = pickle.load(f)
    input_seq = input_tokenizer.texts_to_sequences([input_text])[0]

    input_seq = pad_sequences([input_seq], maxlen=max_input_seq_len, padding='post')

    decoder_input = np.zeros(shape=(len(input_seq), max_output_seq_len))
    decoder_input[:, 0] = output_tokenizer.word_index['<sos>']
    for i in range(1, max_output_seq_len):
        predictions = model.predict([input_seq, decoder_input], verbose=0).argmax(axis=2)
        decoder_input[:, i] = predictions[:, i - 1]
        index = int(decoder_input[0, i])
        if index == 0:
            continue
        word = output_tokenizer.index_word[index]
        current_output = current_output + word + ' '
        if word == '<eos>':
            break
        # print(word, end=' ')

    # print(decoder_input)

    # Convert the output sequence to text
    output_text = ''
    for i in range(max_output_seq_len):
        # print(int(decoder_input[0,i]))
        # print(output_tokenizer.index_word)
        if output_tokenizer.index_word[int(decoder_input[0, i])] == '<sos>':
            continue
        if output_tokenizer.index_word[int(decoder_input[0, i])] == '<eos>':
            break
        else:
            output_text += output_tokenizer.index_word[int(decoder_input[0, i])] + ' '

    return output_text

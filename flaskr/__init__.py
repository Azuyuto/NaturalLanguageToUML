import os
import joblib
from flask import Flask, render_template, request
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

import numpy as np
import re

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
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

    @app.route('/', methods=['GET', 'POST'])
    def main():
        
        # If a form is submitted
        if request.method == "POST":
            
            model = load_model('flaskr/my_model3.h5')
            input_text = request.form.get("text")
            print(123)
            with open("flaskr/Data_modified_2.txt", "r") as f:
                data = f.read()
            cleaned_data = data
            cleaned_data = re.sub(r' {4}', ' \t ', cleaned_data)
            cleaned_data = re.sub(r'\[', ' [ ', cleaned_data)
            cleaned_data = re.sub(r'\]', ' ] ', cleaned_data)
            cleaned_data = re.sub(r'\(', ' ( ', cleaned_data)
            cleaned_data = re.sub(r'\)', ' ) ', cleaned_data)
            cleaned_data = re.sub(r'\n', ' \n ', cleaned_data)
            cleaned_data = re.sub(r' {2,}', ' ', cleaned_data)

            input_texts = []
            output_texts = []
            for line in cleaned_data.split('/'):
                sentence_parts = line.strip().split("|")
                input_texts.append(sentence_parts[0].strip())
                output_texts.append(sentence_parts[1].strip())
            pd.set_option('display.max_colwidth', None)
            pd.DataFrame({'input': input_texts, 'output': output_texts})

            # Define the input and output tokenizers
            input_tokenizer = Tokenizer(filters='', char_level=False)
            output_tokenizer = Tokenizer(filters='', char_level=False)

            # Define the maximum sequence length for the input and output sequences
            max_input_seq_len = max(len(seq) for seq in input_texts)
            max_output_seq_len = max(len(seq) for seq in output_texts)

            # Fit the input tokenizer on the preprocessed input sequences
            input_tokenizer.fit_on_texts(input_texts)

            # Fit the output tokenizer on the preprocessed output sequences
            output_tokenizer.fit_on_texts(output_texts)

            SOS_token = '<sos>'
            EOS_token = '<eos>'

            # Add the special tokens to the output tokenizer
            output_tokenizer.word_index[SOS_token] = len(output_tokenizer.word_index) + 1
            output_tokenizer.word_index[EOS_token] = len(output_tokenizer.word_index) + 1

            output_tokenizer.index_word[len(output_tokenizer.index_word) + 1] = SOS_token
            output_tokenizer.index_word[len(output_tokenizer.index_word) + 1] = EOS_token

            input_text = request.form.get("text")

            # Tokenize the input sentence
            input_seq = input_tokenizer.texts_to_sequences([input_text])[0]

            # Pad the input sequence
            input_seq = pad_sequences([input_seq], maxlen=max_input_seq_len, padding='post')

            # Generate the output sequence using the trained model
            decoder_input = np.zeros(shape=(len(input_seq), max_output_seq_len))
            decoder_input[:, 0] = output_tokenizer.word_index['<sos>']
            for i in range(1, max_output_seq_len):
                predictions = model.predict([input_seq, decoder_input]).argmax(axis=2)
                decoder_input[:, i] = predictions[:, i-1]

            # print(decoder_input)

            # Convert the output sequence to text
            output_text = ''
            for i in range(max_output_seq_len):
                # print(int(decoder_input[0,i]))
                # print(output_tokenizer.index_word)
                if output_tokenizer.index_word[int(decoder_input[0,i])] == '<sos>':
                    continue
                if output_tokenizer.index_word[int(decoder_input[0,i])] == '<eos>':
                    break
                else:
                    output_text += output_tokenizer.index_word[int(decoder_input[0,i])] + ' '

            prediction = output_text
            prediction = prediction.replace("flowchart td", "flowchart TD")
            prediction = prediction.replace("td td", "flowchart TD")  
            prediction = prediction.replace(" [", "[")                        
        else:
            prediction = ""
            
        return render_template("website.html", output = prediction)

    return app

from django.shortcuts import render
import tensorflow as tf
import numpy as np
import json
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# Create your views here.
# myapp/views.py
from rest_framework.decorators import api_view
from rest_framework.response import Response
from myapp.utils import load_model_and_tokenizer

model, tokenizer = load_model_and_tokenizer('myapp/twitter_sentiment_model.h5', "tokenizer.json")

@api_view(['POST'])
def predict_sentiment(request):
    data = request.data
    sentences = data.get('sentences', [])

    # Tokenize and pad the sequences
    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, padding='post')


    
    padded_array = np.array(padded)

    # Assuming padded is a NumPy array
    padded_array = np.array(padded)

    # Check if 1 is present in the first row of the padded array
    if 1 in padded_array[0]:
        # Find the index (position) of 1 in the first row
        positions = np.where(padded_array[0] == 1)[0]

        words_at_positions = [sentences[0].split()[position] for position in positions]
        print(f"Out of vocabulary words present at position {positions}{words_at_positions} in the first row")
    else:
        words_at_positions="Null"
        print("Value 1 is not present in the first row of the padded array")
    # Make predictions
    predictions = model.predict(padded)

    # Format predictions
    results = []
    class_labels = ['Negative', 'Neutral', 'Positive']
    for sentence, prediction in zip(sentences, predictions):
        max_prob_index = tf.argmax(prediction)
        result = {"sentence": sentence, "predicted_class": class_labels[max_prob_index], "OOV":words_at_positions}
        results.append(result)

    return Response({"prediction": results})


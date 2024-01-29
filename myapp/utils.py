import tensorflow as tf
import json
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# import tensorflow as tf
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

import os
print("Current Working Directory:", os.getcwd())


print("linne       1")
new_model = tf.keras.models.load_model('myapp/twitter_sentiment_model.keras')
print("linne       2")


# Show the model architecture
# new_model.summary()

def load_model_and_tokenizer(model_path, tokenizer_path):
    # Load the Keras model
    print(model_path)
    # return
    model = load_model(model_path)
    print("here")


    # Load the tokenizer
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer_config = f.read()
        # print((tokenizer_config))

     
        tokenizer=tf.keras.preprocessing.text.tokenizer_from_json(
                tokenizer_config
        )
     # Deserialize the tokenizer config
        # tokenizer = tf.keras.preprocessing.text.Tokenizer.from_config(tf.keras.utils.deserialize_keras_object(tokenizer_config))


    return model, tokenizer

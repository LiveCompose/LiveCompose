# To evaulate photos with inception_resnet model 
# without command line statement
import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tensorflow.keras.utils import load_img, img_to_array
import tensorflow as tf
from utils.score_utils import mean_score, std_score

def nima_score(img_paths, weights_path='weights/inception_resnet_weights.h5', target_size=(224, 224)):
    with tf.device('/CPU:0'):
        base_model = InceptionResNetV2(input_shape=(None, None, 3), include_top=False, pooling='avg', weights=None)
        x = Dropout(0.75)(base_model.output)
        x = Dense(10, activation='softmax')(x)
        model = Model(base_model.input, x)
        model.load_weights(weights_path)

        results = []
        for img_path in img_paths:
            img = load_img(img_path, target_size=target_size)
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            scores = model.predict(x, batch_size=1, verbose=0)[0]
            mean = mean_score(scores)
            std = std_score(scores)
            results.append((img_path, mean, std))
        return results

if __name__ == "__main__":
  
    img_list = [
        "./images/animal-1.jpg",
        "./images/cat-1562468.jpg",
        "./images/cat-4834800.jpg",
        "./images/goodBird.jpg",
        "./images/bad3.jpg"
    ]
    results = nima_score(img_list)
    for img_path, mean, std in results:
        print(f"{img_path}: NIMA Score = {mean:.3f} ± ({std:.3f})")
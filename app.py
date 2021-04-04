import numpy as np
import librosa
from flask import Flask, request
from flask_restful import Api, Resource
from tensorflow import keras
import pandas as pd 

app = Flask(__name__)
api = Api(app)

original = ['female_angry','female_calm','female_fearful','female_happy','female_sad','male_angry','male_calm','male_fearful','male_happy','male_sad']
opt = keras.optimizers.RMSprop(lr=0.00001, decay=1e-6)
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = keras.models.model_from_json(loaded_model_json)
loaded_model.load_weights("Emotion_Voice_Detection_Model.h5")
loaded_model.compile(loss='categorical_crossentropy',
                     optimizer=opt, metrics=['accuracy'])
print("Loaded model from disk")


class Predict(Resource):
    def get(self):
        return "app running"
    def post(self):
        X, sample_rate = librosa.load(request.files['file'], res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
        sample_rate = np.array(sample_rate)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
        featurelive = mfccs
        livedf2 = featurelive
        livedf2= pd.DataFrame(data=livedf2)
        livedf2 = livedf2.stack().to_frame().T
        twodim= np.expand_dims(livedf2, axis=2)
        livepreds = loaded_model.predict(twodim, batch_size=32,verbose=1)
        livepreds1=livepreds.argmax(axis=1)
        liveabc = livepreds1.astype(int).flatten()
        print(original[liveabc[0]])
        return str(original[liveabc[0]].split("_")[1])

api.add_resource(Predict, "/")

if __name__ == '__main__':
    app.run()

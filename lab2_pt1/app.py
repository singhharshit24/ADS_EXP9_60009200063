from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

model = tf.keras.models.load_model("model.h5")
upload_folder = os.path.join('static', 'uploads')
app.config['UPLOAD'] = upload_folder

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "imageData" not in request.files:
            return redirect("/")
        img = request.files["imageData"]
        print(img)
        pil_img = Image.open(img)
        img_data = np.array(pil_img)
        prediction = np.argmax(model.predict(np.expand_dims(img_data, axis=0)))

        filename = secure_filename(img.filename)
        img.save(os.path.join(app.config['UPLOAD'], filename))
        img = os.path.join(app.config['UPLOAD'], filename)
        return render_template("predicted.html", prediction=prediction, img=img)


if __name__ == "__main__":
    app.debug = True
    app.run()

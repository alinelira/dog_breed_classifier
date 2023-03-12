import base64
import io
import tensorflow as tf

from pathlib import Path
from flask import Flask, render_template, request, flash
from PIL import Image
from config_app import *


ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

app = Flask(__name__)
app.secret_key = "dog_breed_amol"

graph = tf.get_default_graph()  # Initialize default graph

@app.route("/")
@app.route("/index")
def index():
    return render_template("master.html")

@app.route("/get_prediction", methods=["POST"])

def get_prediction():
    """ Get prediction of image provided by user """
    
    img_file = request.files["img_provided"]
    
    if Path(img_file.filename).suffix not in ALLOWED_EXTENSIONS:
        flash("Error: file not in allowed extensions")
        return render_template("master.html")
    
    img_content = img_file.read() # Get file content
    img = Image.open(io.BytesIO(img_content))
    
    global graph  # Use the default graph in this function
    with graph.as_default():
        prediction = classify_image(io.BytesIO(img_content))

    if prediction:
        display_image = io.BytesIO()
        img.save(display_image, "PNG")
        display_image.seek(0)
        display_image = base64.b64encode(display_image.getvalue()).decode("ascii")
        return render_template("master.html", data=[prediction, display_image])
    else:
        return render_template("master.html")

def main():
    app.run(host='127.0.0.1', port=5000, debug=True)

if __name__ == '__main__':
    main()

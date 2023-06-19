from flask import Flask, request, json
import tensorflow as tf
import urllib.parse
from matplotlib.image import imread
import os
import flask_cors

# Flask server.
app = Flask(__name__)
flask_cors.CORS(app)

# Load pretrained model into program.
loaded_model = tf.keras.models.load_model(
    os.path.join(os.path.dirname(__file__), "image_model")
)

# Load image info.
# This file contains a list that correlates to the model output, mapping the indexes
# of model output array indexes to the category names.
with open(
    os.path.join(os.path.dirname(__file__), "image_info.json"), "r"
) as image_info:
    image_info = json.load(image_info)


# Gets the models prediction of an image.
@app.route("/api/predict", methods=["POST"])
def index():
    # Get image url and load image.
    url = urllib.parse.unquote(request.json["image"])
    test_image = imread(url)

    # Reshape image. Expects (#, #, #, #);
    test_image = test_image.reshape((1, 256, 256, 3))
    results = (loaded_model.predict(test_image) * 100).tolist()[0]

    # Get highest prediction.
    first_highest_percent = max(results)
    first_highest = results.index(first_highest_percent)
    results[first_highest] = -1

    # Get second highest prediction.
    second_highest_percent = max(results)
    second_highest = results.index(second_highest_percent)

    # Get predictions index values in the images_info file.
    map_first = image_info["map"][first_highest]
    map_second = image_info["map"][second_highest]

    # Get each predction's full category name.
    category_first = (
        image_info["categories"][map_first[0]]["name"]
        + " "
        + image_info["categories"][map_first[0]]["diseases"][map_first[1]]["name"]
    )
    category_second = (
        image_info["categories"][map_second[0]]["name"]
        + " "
        + image_info["categories"][map_second[0]]["diseases"][map_second[1]]["name"]
    )

    return (
        {
            "results": 2,
            "status": "success",
            "data": {
                "predictions": [
                    {"name": category_first, "percent": first_highest_percent},
                    {"name": category_second, "percent": second_highest_percent},
                ]
            },
        },
        200,
        {"Content-Type": "application/json", "Allow-Control-Allow-Origin": "*"},
    )


# Load model infomation files.
with open(
    os.path.join(os.path.dirname(__file__), "history.json"), "r"
) as model_history:
    model_history = json.load(model_history)

with open(
    os.path.join(os.path.dirname(__file__), "history_fine_tune.json"), "r"
) as history_fine_tune:
    history_fine_tune = json.load(history_fine_tune)

with open(
    os.path.join(os.path.dirname(__file__), "image_info.json"), "r"
) as image_info:
    image_info = json.load(image_info)

with open(
    os.path.join(os.path.dirname(__file__), "test_results.json"), "r"
) as test_results:
    test_results = json.load(test_results)


# Returns infomation about model training history and dataset.
@app.route("/api/stats", methods=["GET"])
def imageStats():
    return (
        {
            "status": "success",
            "data": {
                "training_stats": {"init": model_history, "tuning": history_fine_tune},
                "image_stats": image_info,
                "test_results": test_results,
            },
        },
        200,
        {"Content-Type": "application/json", "Allow-Control-Allow-Origin": "*"},
    )


if __name__ == "__main__":
    app.run(host="172.28.91.13", port=3002, debug=True)

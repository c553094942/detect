import flask
import detect_img

app = flask.Flask(__name__)
model = None
use_gpu = True


@app.route("/predict", methods=["POST"])
def predict():
    # Initialize the data dictionary that will be returned from the view.
    data = {"Detect": False,
            "Video": False}
    # Ensure an image was properly uploaded to our endpoint.
    if flask.request.method == 'POST':
        m = str(flask.request.data).split("'")[1]
        form = m.split('.')[1]
        if form in ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'dng']:
            # stream = m
            model = detect_img.detector()
            info = model.detect(m)
            data["Detect"] = True
            data["info"] = info
            # data['info'] = info
            return flask.jsonify(data)
        elif form in ['mov', 'avi', 'mp4']:
            model = detect_img.detector()
            model.detect(m)
            data["Video"] = True
            # data['info'] = info
            return flask.jsonify(data)
    # Return the data dictionary as a JSON response.


if __name__ == '__main__':
    print("Loading PyTorch model and Flask starting server ...")
    print("Please wait until server has fully started")
    app.run()

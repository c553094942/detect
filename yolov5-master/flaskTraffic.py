import flask
import threadPush
import detect_traffic


app = flask.Flask(__name__)
model = None
use_gpu = True


@app.route("/traffic", methods=["POST"])
def predict():
    # Initialize the data dictionary that will be returned from the view.
    data = {"Traffic": False}
    # Ensure an image was properly uploaded to our endpoint.
    if flask.request.method == 'POST':
        m = str(flask.request.data).split("'")[1]
        model = detect_traffic.detector()
        info = model.detect(m)
        data["Traffic"] = True
        data["info"] = info
    # Return the data dictionary as a JSON response.
    return flask.jsonify(data)


if __name__ == '__main__':
    print("Loading PyTorch model and Flask starting server ...")
    print("Please wait until server has fully started")
    app.run()
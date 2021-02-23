import requests
import json
PyTorch_REST_API_URL = 'http://127.0.0.1:5000/predict'


def predict_result(stream="rtsp://admin:hxy123456@192.168.1.64:554/h264/ch1/main", cam_id=1,
                   rtmpurl='rtmp://192.168.1.4:1935/live/test'):
    # Initialize image path
    # image = open(stream, 'rb').read()
    payload = {'detect': stream, 'cam_ID': cam_id}
    payload = {'stream': stream, 'cam_ID': cam_id, 'rtmpurl': rtmpurl}
    print('payload', payload)
    # Submit the request.
    r = requests.post(PyTorch_REST_API_URL, files=payload).json()
    # print('r', r.text)
    # r = json.loads(r.text)
    # print('r', r)
    # Ensure the request was successful.
    if r['success']:
        # Loop over the predictions and display them.
        # for (i, result) in enumerate(r['predictions']):
        #     print('{}. {}: {:.4f}'.format(i + 1, result['label'],
        #                                   result['probability']))
        print('success')
    # Otherwise, the request failed.
    else:
        print('Request failed')


predict_result()

# url = "http://127.0.0.1:5000/predict"
#
# image_path = "/media/yons/DATA/PycharmProjects/yolov5-master/dog.jpeg"
# image = open(image_path, 'rb').read()
# payload = {'image': image}
# headers = {
#     'cache-control': "no-cache",
#     'Postman-Token': "dc1d0671-d060-45ab-86b5-6bf6d8fc9efd"
#     }
#
# response = requests.request("POST", url, data=payload, headers=headers)
#
# print(response.text)

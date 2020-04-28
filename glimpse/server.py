"""Server Module."""
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO

from PIL import Image

from object_detection.model import Model
import cv2

# device's IP address
SERVER_HOST = "localhost"
SERVER_PORT = 10000
# receive 4096 bytes each time
BUFFER_SIZE = 4096  # 4KB


class Server(HTTPServer):
    """Implementation of Glimpse Server."""

    def __init__(self, hostname, port, model_path, device, save_folder):
        # listen to client
        super().__init__((hostname, port), RequestHandler)
        self.model = Model(model_path, device)


class RequestHandler(BaseHTTPRequestHandler):
    """Handle request from glimpse client."""

    def do_POST(self):
        """Process frame received from client and send back the detections."""
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        self.send_response(200)
        self.end_headers()
        response = BytesIO()
        print(len(body))
        img = Image.open(io.BytesIO(body))
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
        detections, _ = self.server.model.infer(img)
        # parse detections to user-friendly format
        parsed_detections = []
        for label, box, score in zip(detections['detection_classes'],
                                     detections['detection_boxes'],
                                     detections['detection_scores']):
            parsed_detections.append(
                [float(box[0]), float(box[1]), float(box[2]),
                 float(box[3]), int(label), float(score)])

        response.write(bytes(json.dumps(parsed_detections), 'ascii'))
        self.wfile.write(response.getvalue())


def main():
    """Simle Test."""
    # model_path = '/data/zxxia/models/research/object_detection/ssd_mobilenet_v2_coco_2018_03_29'
    model_path = '/data/zxxia/models/research/object_detection/faster_rcnn_resnet101_coco_2018_01_28'
    server = Server(SERVER_HOST, SERVER_PORT, model_path, 0, '.')
    server.serve_forever()
    # server.run()


if __name__ == "__main__":
    main()

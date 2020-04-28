"""Server Module."""
import argparse
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO

import cv2
from PIL import Image

from object_detection.model import Model

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
        detections, t_used = self.server.model.infer(img)
        # parse detections to user-friendly format
        parsed_detections = []
        for label, box, score in zip(detections['detection_classes'],
                                     detections['detection_boxes'],
                                     detections['detection_scores']):
            parsed_detections.append(
                [float(box[0]), float(box[1]), float(box[2]),
                 float(box[3]), int(label), float(score)])

        response.write(bytes(json.dumps([parsed_detections, t_used]), 'ascii'))
        self.wfile.write(response.getvalue())


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Glimpse Client.")
    parser.add_argument("--hostname", type=str, default="localhost",
                        help="Hostname to connect to.")
    parser.add_argument("--port", type=int, default=10000,
                        help="Port to connect to.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to video file.")
    args = parser.parse_args()
    return args


def main():
    """Run Glimpse server."""
    args = parse_args()
    server = Server(args.hostname, args.port, args.model_path, 0, '.')
    server.serve_forever()


if __name__ == "__main__":
    main()

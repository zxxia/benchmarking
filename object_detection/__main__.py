"""Object detection module main function."""
from object_detection.parser import parse_args
from object_detection.infer import infer

if __name__ == '__main__':
    args = parse_args()
    infer(args.input_path, args.output_path, args.device,
          args.model, args.width, args.height, args.qp, args.crop)

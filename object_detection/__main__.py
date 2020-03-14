"""Inference module main function."""
from inference.parser import parse_args
from inference.infer import infer

if __name__ == '__main__':
    args = parse_args()
    infer(args)

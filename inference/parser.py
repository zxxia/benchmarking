"""Inferece module parser."""
import argparse


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Inference model.")
    parser.add_argument("--device", type=int, required=True, help="GPU ID")
    parser.add_argument("--input_path", type=str, required=True,
                        help="input path which contains images to infer")
    parser.add_argument("--model", type=str, required=True, help="model path")
    parser.add_argument("--output_path", type=str, required=True,
                        help="output path where output files will be saved to")
    # parser.add_argument("--batch_size", type=int, defaul=1,
    # help="batch size")
    args = parser.parse_args()
    return args

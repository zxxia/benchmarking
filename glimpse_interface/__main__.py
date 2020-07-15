"""Glimpse module main function."""
from glimpse_interface.parser import parse_args
from glimpse_interface.run import run


if __name__ == '__main__':
    args = parse_args()
    run(args)

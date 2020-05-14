"""Glimpse module main function."""
from glimpse.parser import parse_args
from glimpse.run import run


if __name__ == '__main__':
    args = parse_args()
    run(args)

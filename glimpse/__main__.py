"""Glimpse module main function."""
from glimpse.parser import parse_args
from videostorm.run import run


if __name__ == '__main__':
    args = parse_args()
    run(args)

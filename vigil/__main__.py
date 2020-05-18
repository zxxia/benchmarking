"""Vigil module main function."""
from vigil.parser import parse_args
from vigil.run import run


if __name__ == '__main__':
    args = parse_args()
    run(args)

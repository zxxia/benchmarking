"""AWStream module main function."""
from awstream.parser import parse_args
from awstream.run import run


if __name__ == '__main__':
    args = parse_args()
    run(args)

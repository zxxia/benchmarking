"""AWStream Interface module main function."""
from awstream_interface.parser import parse_args
from awstream_interface.run import run


if __name__ == '__main__':
    args = parse_args()
    run(args)

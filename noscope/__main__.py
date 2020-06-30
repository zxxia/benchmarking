"""NoScope module main function."""
from noscope.parser import parse_args
from noscope.run import run


if __name__ == '__main__':
    args = parse_args()
    run(args)

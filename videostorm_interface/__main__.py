"""VideoStorm Interface module main function."""
from videostorm_interface.parser import parse_args
from videostorm_interface.run import run


if __name__ == '__main__':
    args = parse_args()
    run(args)

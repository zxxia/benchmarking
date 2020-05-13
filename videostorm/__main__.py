"""VideoStorm module main function."""
from videostorm.parser import parse_args
from videostorm.run import run


if __name__ == '__main__':
    args = parse_args()
    run(args)

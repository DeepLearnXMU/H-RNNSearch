#!/usr/bin/python
import sys


def main():
    for line in sys.stdin:
        sys.stdout.write('%s\n' % (''.join(line.split())))


if __name__ == '__main__':
    main()


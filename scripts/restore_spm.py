#!/usr/bin/python
# coding=utf-8
import sys


def main():
    for line in sys.stdin:
        line = ''.join(line.split())
        line = ' '.join(line.split('▁'))
        sys.stdout.write('%s\n' % line)


if __name__ == "__main__":
    main()

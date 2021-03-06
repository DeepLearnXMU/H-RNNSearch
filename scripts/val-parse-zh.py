#!/usr/bin/python
import sys
import re


def main(enc):
    if enc is None:
        enc = 'utf-8'
    pattern = re.compile('(([^\\s\\d\\w])|(\\d\\s)+|(\\w+))')
    for line in sys.stdin:
        line = line.strip().decode(enc)
        output = pattern.sub(r' \1 ', line).strip().encode(enc)
        output = ' '.join(output.split())
        sys.stdout.write('%s\n' % output)


if __name__ == '__main__':
    main(None)


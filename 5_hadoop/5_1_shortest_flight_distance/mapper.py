#!/usr/bin/env python
import sys

"""
Disclaimer:
This implementation makes use of code produced for
the homework assignment 5.
"""

# This is implemented to skip the first line (the header)
header = True
for line in sys.stdin:
    if header is True:
        header = False
        continue

    # Remove whitespace and split up the lines into data (space as delimiter)

    line = line.strip()
    data = line.split(',')

    # Save the airline_id and the distance for each flight
    airline_id = data[1]
    distance = data[10]

    # Stream the two values
    print('%s\t%s' % (airline_id, distance))

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

    # Save the airline_id and the arrival delay for each flight
    airline_id = data[1]
    arrival_delay = data[8]
    if arrival_delay == "":
        arrival_delay = 0

    # Stream the two values
    print('%s\t%s' % (airline_id, arrival_delay))

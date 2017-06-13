#!/usr/bin/env python
import sys

# Where do these lines come from?
# This is done by Hadoop Streaming ...
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

    print('%s\t%s' % (airline_id, distance))

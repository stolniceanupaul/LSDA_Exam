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

    # Save the airline_id and the arrival delay for each fligh
    airline_id = data[1]
    arrival_delay = data[8]
    if arrival_delay == "":
        arrival_delay = 0

    print('%s\t%s' % (airline_id, arrival_delay))

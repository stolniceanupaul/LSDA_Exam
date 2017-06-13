#!/usr/bin/env python
import sys

min_dist = None
old_airline = None

# Input from stdin (handled via Hadoop Streaming)
for line in sys.stdin:

    # Remove whitespace and split up lines
    line = line.strip()
    line = line.split('\t')
    if len(line) != 2:
        continue

    # Get airline_id and the distance
    airline_id, distance = line

    try:
        distance = float(distance)
    except ValueError:
        continue

    # This if-statement only works because Hadoop sorts
    # the output of the mapping phase by key (here, by
    # airline_id) before it is passed to the reducers. Each
    # reducer gets all the values for a given key. Each
    # reducer might get the values for MULTIPLE keys.
    if (old_airline is not None) and (old_airline != airline_id):
        print('%s\t%s' % (old_airline, min_dist))
        min_dist = None

    if min_dist > distance or min_dist is None:
        min_dist = distance

    old_airline = airline_id

# We have to output the shortest distance for the last airline_id!
if old_airline is not None:
    print('%s\t%s' % (old_airline, min_dist))

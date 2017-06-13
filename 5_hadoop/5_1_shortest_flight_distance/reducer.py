#!/usr/bin/env python
import sys

"""
Disclaimer:
This implementation makes use of code produced for
the homework assignment 5.
"""

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

    # This usses the assumption that the pairs (key, value)
    # are sorted after the mapping phase and therefore
    # the input for the reducer is sorted by key.

    if (old_airline is not None) and (old_airline != airline_id):
        print('%s\t%s' % (old_airline, min_dist))
        min_dist = None

    if min_dist > distance or min_dist is None:
        min_dist = distance

    old_airline = airline_id

# Stream the output for the last airline_id
if old_airline is not None:
    print('%s\t%s' % (old_airline, min_dist))

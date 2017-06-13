#!/usr/bin/env python
import sys

"""
Disclaimer:
This implementation makes use of code produced for
the homework assignment 5.
"""

total_flights = 0
delayed_flights = 0.0
old_airline = None

# Input from stdin (handled via Hadoop Streaming)
for line in sys.stdin:

    # Remove whitespace and split up lines
    line = line.strip()
    line = line.split('\t')
    if len(line) != 2:
        continue

    # Get airline_id and the arrival_delay
    airline_id, arrival_delay = line

    try:
        arrival_delay = float(arrival_delay)
    except ValueError:
        continue

    # This usses the assumption that the pairs (key, value)
    # are sorted after the mapping phase and therefore
    # the input for the reducer is sorted by key.

    if (old_airline is not None) and (old_airline != airline_id):
        print('%s\t%s\t%s\t%s' % (old_airline, total_flights, delayed_flights, delayed_flights * 100 / total_flights))
        total_flights = 0
        delayed_flights = 0.0

    if arrival_delay > 0:
        delayed_flights += 1

    total_flights += 1
    old_airline = airline_id

# Stream the output for the last airline_id
if old_airline is not None:
    print('%s\t%s\t%s\t%s' % (old_airline, total_flights, delayed_flights, delayed_flights * 100 / total_flights))

#!/usr/bin/env python
import sys

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

    # Get airline_id and the distance
    airline_id, arrival_delay = line

    try:
        arrival_delay = float(arrival_delay)
    except ValueError:
        continue

    # This if-statement only works because Hadoop sorts
    # the output of the mapping phase by key (here, by
    # airline_id) before it is passed to the reducers. Each
    # reducer gets all the values for a given key. Each
    # reducer might get the values for MULTIPLE keys.
    if (old_airline is not None) and (old_airline != airline_id):
        print('%s\t%s\t%s\t%s' % (old_airline, total_flights, delayed_flights, delayed_flights * 100 / total_flights))
        total_flights = 0
        delayed_flights = 0.0

    if arrival_delay > 0:
        delayed_flights += 1

    total_flights += 1
    old_airline = airline_id

# We have to output the shortest distance for the last airline_id!
if old_airline is not None:
    print('%s\t%s\t%s\t%s' % (old_airline, total_flights, delayed_flights, delayed_flights * 100 / total_flights))

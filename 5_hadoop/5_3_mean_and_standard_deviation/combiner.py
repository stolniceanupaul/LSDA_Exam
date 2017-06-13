#!/usr/bin/env python
import sys

"""
Disclaimer:
This implementation makes use of code produced for
the homework assignment 5.
"""

number_of_flights = 0
sum_of_delays = 0
squared_sum_of_delays = 0
old_airline_id = None

# Input from stdin (handled via Hadoop Streaming)
for line in sys.stdin:

    # Remove whitespace and split up lines
    line = line.strip()
    line = line.split('\t')
    if len(line) != 2:
        continue

    # Get airline_id and the arrival delay
    airline_id, arrival_delay = line

    try:
        arrival_delay = float(arrival_delay)
    except ValueError:
        continue

    # This usses the assumption that the pairs (key, value)
    # are sorted after the mapping phase (per document) and therefore
    # the input for the combiner is sorted by key as well.

    if (old_airline_id is not None) and (old_airline_id != airline_id):

        # For each airline_id this streams:
        # - airline_id
        # - number_of_flights
        # - sum_of_delays
        # - squared_sum_of_delays
        # These will be used in the reducer to compute the mean and the std.

        print('%s\t%s\t%s\t%s' % (old_airline_id, number_of_flights, sum_of_delays, squared_sum_of_delays))
        number_of_flights = 0
        sum_of_delays = 0.0
        squared_sum_of_delays = 0.0

    number_of_flights += 1
    sum_of_delays += arrival_delay
    squared_sum_of_delays += pow(arrival_delay, 2)
    old_airline_id = airline_id

# Stream the output for the last airline_id
if old_airline_id is not None:
    print('%s\t%s\t%s\t%s' % (old_airline_id, number_of_flights, sum_of_delays, squared_sum_of_delays))
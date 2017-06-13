#!/usr/bin/env python
import sys

"""
Disclaimer:
This implementation makes use of code produced for
the homework assignment 5.
"""

total_flights = 0
total_sum_of_delays = 0.0
total_sum_of_squared_delays = 0.0
old_airline_id = None

# Input from stdin (handled via Hadoop Streaming)
for line in sys.stdin:

    # Remove whitespace and split up lines
    line = line.strip()
    line = line.split('\t')
    if len(line) != 4:
        continue

    # Get the airline_id, the number of flights and the two sums:
    # the sum of delays and the sum of squared delays
    airline_id, number_of_flights, sum_of_delays, sum_of_squared_delays = line

    try:
        number_of_flights = float(number_of_flights)
        sum_of_delays = float(sum_of_delays)
        sum_of_squared_delays = float(sum_of_squared_delays)
    except ValueError:
        continue

    # This usses the assumption that the pairs (key, value)
    # are sorted after the mapping phase and therefore
    # the input for the reducer is sorted by key.

    if (old_airline_id is not None) and (old_airline_id != airline_id):
        mean_delays = total_sum_of_delays / total_flights

        # Uses the shortcut formula to calculate the standard deviation
        # as described in the report.
        std_dev = pow(((total_sum_of_squared_delays - (pow(total_sum_of_delays, 2) / total_flights)) / total_flights), 0.5)

        print('%s\t%s\t%s' % (old_airline_id, mean_delays, std_dev))
        total_flights = 0
        total_sum_of_delays = 0.0
        total_sum_of_squared_delays = 0.0
        old_airline_id = None

    total_flights += number_of_flights
    total_sum_of_delays += sum_of_delays
    total_sum_of_squared_delays += sum_of_squared_delays
    old_airline_id = airline_id

# Stream the output for the last airline_id
if old_airline_id is not None:
    mean_delays = total_sum_of_delays / total_flights
    mean_delays = total_sum_of_delays / total_flights
    std_dev = pow(((total_sum_of_squared_delays - (pow(total_sum_of_delays, 2) / total_flights)) / total_flights), 0.5)
    print('%s\t%s\t%s' % (old_airline_id, mean_delays, std_dev))

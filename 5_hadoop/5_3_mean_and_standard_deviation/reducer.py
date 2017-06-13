#!/usr/bin/env python
import sys

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

    # Get airline_id and the distance
    airline_id, number_of_flights, sum_of_delays, sum_of_squared_delays = line

    try:
        number_of_flights = float(number_of_flights)
        sum_of_delays = float(sum_of_delays)
        sum_of_squared_delays = float(sum_of_squared_delays)
    except ValueError:
        continue

    # This if-statement only works because Hadoop sorts
    # the output of the mapping phase by key (here, by
    # airline_id) before it is passed to the reducers. Each
    # reducer gets all the values for a given key. Each
    # reducer might get the values for MULTIPLE keys.
    if (old_airline_id is not None) and (old_airline_id != airline_id):
        mean_delays = total_sum_of_delays / total_flights
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

# We have to output the shortest distance for the last airline_id!
if old_airline_id is not None:
    mean_delays = total_sum_of_delays / total_flights
    mean_delays = total_sum_of_delays / total_flights
    std_dev = pow(((total_sum_of_squared_delays - (pow(total_sum_of_delays, 2) / total_flights)) / total_flights), 0.5)
    print('%s\t%s\t%s' % (old_airline_id, mean_delays, std_dev))

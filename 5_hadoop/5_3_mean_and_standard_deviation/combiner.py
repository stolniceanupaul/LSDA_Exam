#!/usr/bin/env python
import sys

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
    if (old_airline_id is not None) and (old_airline_id != airline_id):
        print('%s\t%s\t%s\t%s' % (old_airline_id, number_of_flights, sum_of_delays, squared_sum_of_delays))
        number_of_flights = 0
        sum_of_delays = 0.0
        squared_sum_of_delays = 0.0

    number_of_flights += 1
    sum_of_delays += arrival_delay
    squared_sum_of_delays += pow(arrival_delay, 2)
    old_airline_id = airline_id

# We have to output the shortest distance for the last airline_id!
if old_airline_id is not None:
    print('%s\t%s\t%s\t%s' % (old_airline_id, number_of_flights, sum_of_delays, squared_sum_of_delays))
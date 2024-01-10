import itertools
def generate_early_shift_distributions(total_officers=15, stations=3):
    """
    Generates all possible distributions of officers across three police stations
    for the early shift that sum up to a specific total.

    :param total_officers: Total number of officers to be distributed.
    :param stations: Number of stations.
    :return: List of tuples representing different officer distributions for the early shift.
    """
    distributions = []

    # Iterate through all possible combinations
    for distribution in itertools.product(range(total_officers + 1), repeat=stations):
        if sum(distribution) == total_officers:
            distributions.append(distribution)

    return distributions

import sys
from dataclasses import dataclass
import itertools
from pyproj import Proj, Transformer, transform
from math import floor, ceil, radians, cos, sin, asin, sqrt


@dataclass
class Location:
    x: float
    y: float


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


def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 3956  # Radius of Earth in miles. Use 6371 for kilometers
    return c * r


def bng_to_latlong(easting, northing):
    """
    Convert British National Grid (BNG) coordinates to Latitude and Longitude.

    :param easting: Easting value of BNG coordinate
    :param northing: Northing value of BNG coordinate
    :return: (latitude, longitude)
    """
    # Define the British National Grid (BNG) projection
    bng_proj = Proj(init='epsg:27700')

    # Define the WGS84 geographic coordinate system
    wgs84_proj = Proj(init='epsg:4326')

    # Perform the transformation
    longitude, latitude = transform(bng_proj, wgs84_proj, easting, northing)

    return latitude, longitude

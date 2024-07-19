from edinburgh_challenge.utility import Location
from dataclasses import dataclass

from enum import Enum


police_station_1 = Location(x=55.868709, y=-4.2579871)
police_station_2 = Location (x=55.849171,y= -4.2164508)
police_station_3 = Location(x=55.830168,y= -4.2468263)

class Shift(Enum):
    early = "Early"
    day = "Day"
    night = "Night"


@dataclass
class PoliceStations:
    one:  police_station_1
    two: police_station_2
    three: police_station_3

police_stations = PoliceStations(
        one=police_station_1,
        two=police_station_2,
        three=police_station_3
        )

police_stations_dict = {
    "Station_1":police_station_1,
    "Station_2":police_station_2,
    "Station_3":police_station_3,
}

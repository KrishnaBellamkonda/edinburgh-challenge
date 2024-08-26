from dataclasses import dataclass, asdict as dataclass_asdict
from typing import List, Dict, Any
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from edinburgh_challenge.constants import Shift
from edinburgh_challenge.processing import PoliceStation



class Model(ABC):
    @abstractmethod
    def make_allocation(self, incidents, officers, current_time):
        pass

@dataclass
class CustodyIncident:
    urn:str
    hour:int
    day: int
    division: str
    max_travel_time:float
    max_wait_time:float
    max_processing_time:float
    global_time: float  # Time at which the incident was reported (24*day + hour)
    resolution_time: float = None  # Global time when the incident is resolved (# travel + wait + processing time)
    type: str = "Custody"
    priority:str = "Prompt" # Assign Prompt priority


@dataclass
class Incident:
    urn: str
    latitude: float
    longitude: float
    day:int
    hour:int
    global_time: float  # Time at which the incident was reported (24*day + hour)
    deployment_time: float  # Time taken to resolve the incident after reaching the location
    priority: str
    distances: Dict[str, float]  # Distances from each station
    resolved: bool = False
    resolving_officer: str = None
    response_time: float = None  # Global time when the response arrives at the scene
    resolution_time: float = None  # Global time when the incident is resolved
    allocation_time:float = None
    type: str = "Incident"

    def asdict(self) -> Dict[str, Any]:
        # Using the dataclass built-in asdict to handle the conversion
        return dataclass_asdict(self)

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, Incident):
            return self.urn == other.urn
        return False


def todataclass(data: Dict[str, Any]) -> Incident:
    # Directly unpacking the dictionary to the Incident dataclass
    return [Incident(**item) for item in data]

@dataclass
class Officer:
    name: str
    station: str
    available: bool = True
    return_time: float = 0.0


# Shift distribution structure
# Example: {'Early': {'Station_1': 5, 'Station_2': 5, 'Station_3': 5}, ...}
ShiftDistribution = Dict[str, Dict[str, int]]

weekday_shift_distribution = {
    'Early': 20,
    'Day': 50,
    'Night':30
}

weekend_shift_distribution = {
    'Early': 20,
    'Day': 40,
    'Night': 40
}

class SimulationWithMaxUtilisation:
    SPEED_MPH = 30  # Speed in miles per hour

    def __init__(self, df: pd.DataFrame, station_coords: List[tuple], shift_distribution: ShiftDistribution, n_days: int = 7, verbose:int=0):
        self.df = df.copy()
        self.station_coords = station_coords
        self.shift_distribution = shift_distribution
        self.current_time = 0
        self.officers = { key: [] for key in shift_distribution["Early"].keys() } # Station_1, Station_2 and Station_3
        self.n_days = n_days
        self.verbose = verbose
        self.hours = [i for i in range(24*self.n_days + 1)]
        self.hour_index = 0
        self.return_times = [] # array keeps a track of return times of officers

        # This resolved incidents is made to keep a track of when an incident
        # is allocated to an officer. This is to conduct the evaluating model check
        self.resolved_incidents = [] # This list keeps a track and order of the resolved incidents as they are allocated


        # Making the df compatible with data structures
        df.columns = [x.lower() for x in df.columns]


    def print_dashboard(self, allocations):
        print(f"\n--- Day {self.current_time // 24 + 1}, Hour {self.current_time % 24} ---")
        pending_cases = [inc for inc in self.cumulative_incidents if not inc.resolved]
        print(f"Pending Cases: {len(pending_cases)}")
        print("Pending Incident URNs:", [inc.urn for inc in pending_cases])

        inv_allocations = {v: k for k, v in allocations.items()}

        for station, officers in self.officers.items():
            print(f"\n{station}:")
            for officer in officers:
                # Find the incident the officer is currently assigned to
                if officer.name in inv_allocations.keys():
                    status = inv_allocations[officer.name]
                else:
                    status = "Busy"
                print(f"  - {officer.name}: {status}")

    def calculate_distance(self, lat1, lon1, lat2, lon2):
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

    def generate_incidents_for_hour(self, global_time):
        hour_incidents = self.df[(self.df['time'] == global_time)]
        incidents = []
        for _, row in hour_incidents.iterrows():
            deployment_time = row.pop("deployment time (hrs)")
            row.pop("time")  # Not needed as we use global_time
            distances = {f'Station_{i+1}': self.calculate_distance(row['latitude'], row['longitude'], lat, lon)
                         for i, (lat, lon) in enumerate(self.station_coords)}
            incidents.append(Incident(**row,
                                      deployment_time=deployment_time,
                                      distances=distances,
                                      global_time=global_time))
        return incidents


    def update_officers_for_shift(self, shift):
        for station, num_officers in self.shift_distribution[shift].items():
            # Update the number of officers for each station
            self.officers[station] = [Officer(f"Officer_{station}_{shift}_{i}", station=station) for i in range(num_officers)]

    def process_allocations(self, allocations):
        #if self.verbose > 0:
            #print(f"{allocations=}")
        for urn, officer_id in allocations.items():
            incident = next((inc for inc in self.cumulative_incidents if inc.urn == urn), None)
            #print("incident:", incident)
            if officer_id is None:
                if incident:
                    incident.resolved = False
                continue

            officer = next((off for station_officers in self.officers.values() for off in station_officers if off.name == officer_id), None)
            if incident and officer:
                travel_time = incident.distances[officer.station] / self.SPEED_MPH
                officer.available = False  # Mark officer as busy
                officer.return_time = self.current_time + travel_time + incident.deployment_time

                self.return_times.append(officer.return_time)

                incident.resolved = True  # Mark incident as resolved
                incident.resolving_officer = officer.name  # Assign officer to incident
                incident.allocation_time = self.current_time
                incident.response_time = self.current_time + travel_time # Global time when the response reached
                incident.resolution_time = officer.return_time # Global Time when the incident was resolved
                self.resolved_incidents.append(incident)



    def update_officer_availability(self):
        for station, officers in self.officers.items():
            for officer in officers:
                if not officer.available and self.current_time >= officer.return_time:
                    officer.available = True


    def get_return_times_for_next_hour(self):
        global_time = self.current_time
        return_times_for_next_hour = [rt for rt in self.return_times if rt <= global_time + 1]
        return sorted(return_times_for_next_hour)


    def run(self, model):
        self.cumulative_incidents = []  # Global list to track all incidents

        while self.current_time < 24 * self.n_days:  # For a week-long simulation

            day = self.current_time // 24 + 1
            hour = self.current_time % 24

            # Update officer availability at the start of each timestep
            self.update_officer_availability()

            # Update officers for shift change
            if hour in [0, 8, 16]:
                shift = 'Early' if hour == 0 else 'Day' if hour == 8 else 'Night'
                self.update_officers_for_shift(shift)
                if self.verbose == -1:
                    no_of_officers = len([officer for station in self.officers.values() for officer in station])
                    print(f"{day=} {shift=} {no_of_officers=}")

            total_officers = len(self.officers["Station_1"]) + len(self.officers["Station_2"]) + len(self.officers["Station_3"])

            # Generate and add new incidents
            new_incidents = self.generate_incidents_for_hour(self.current_time)
            self.cumulative_incidents.extend(new_incidents)

            # Filter to get only pending incidents
            pending_incidents = [inc for inc in self.cumulative_incidents if not inc.resolved]

            allocations = model.make_allocation(pending_incidents, self.officers, self.current_time)

            # Process allocations and update the state
            self.process_allocations(allocations)

            # After making the allocations for the hour
            # get to each return time of the officer
            # and make new allocations
            return_times_within_hour = self.get_return_times_for_next_hour()

            for time in return_times_within_hour:
                self.current_time = time
                self.update_officer_availability()
                pending_incidents = [inc for inc in self.cumulative_incidents if not inc.resolved]
                allocations = model.make_allocation(pending_incidents, self.officers, self.current_time)
                # Process allocations and update the state
                self.process_allocations(allocations)
                self.return_times.remove(time)



            if self.verbose > 2:
                self.print_dashboard(allocations)


            # Pending cases after allocation
            pending_incidents = [inc for inc in self.cumulative_incidents if not inc.resolved]
            #print(f"{pending_incidents=}")

            #self.current_time += 1  # Move to the next hour
            self.hour_index += 1
            self.current_time = self.hours[self.hour_index]

            if self.verbose > 3:
                input("Press Enter to continue to the next hour...\n")

    # Checks and Analysis
    def analyze_simulation_results(self):

        simulation = self

        # Initialize counters and accumulators
        incident_counts = {'Immediate': 0, 'Prompt': 0, 'Standard': 0}
        resolved_counts = {'Immediate': 0, 'Prompt': 0, 'Standard': 0}
        within_threshold_counts = {'Immediate': 0, 'Prompt': 0, 'Standard': 0}
        response_times = {'Immediate': [], 'Prompt': [], 'Standard': []}
        resolution_times = {'Immediate': [], 'Prompt': [], 'Standard': []}
        deployment_times = {'Immediate': [], 'Prompt': [], 'Standard': []}
        thresholds = {'Immediate': 1, 'Prompt': 3, 'Standard': 6}
        total_officer_hours = {}
        unresolved_incidents = 0

        # Analyze each incident
        for incident in simulation.cumulative_incidents:
            priority = incident.priority
            incident_counts[priority] += 1

            if incident.resolved:
                resolved_counts[priority] += 1

                response_time = incident.response_time
                resolution_time = incident.resolution_time

                response_times[priority].append(response_time)
                resolution_times[priority].append(resolution_time)

                deployment_time = incident.deployment_time
                deployment_times[priority].append(deployment_time)

                if incident.resolving_officer not in total_officer_hours.keys():
                    total_officer_hours[incident.resolving_officer]  = 0

                time_spent_on_incident = incident.resolution_time - incident.global_time
                total_officer_hours[incident.resolving_officer] += time_spent_on_incident

                # Calculate the time from incident report to response arrival
                time_to_response = incident.response_time - incident.global_time


                # Check if the response was within the threshold
                if time_to_response <= thresholds[priority]:
                    within_threshold_counts[priority] += 1
            else:
                unresolved_incidents += 1

        # Calculate percentages and mean times
        completion_percentages = {p: (resolved_counts[p] / incident_counts[p]) * 100 if incident_counts[p] > 0 else 0 for p in incident_counts}
        mean_response_times = {p: np.mean(response_times[p]) if response_times[p] else 0 for p in response_times}
        mean_deployment_times = {p: np.mean(deployment_times[p]) if deployment_times[p] else 0 for p in deployment_times}
        threshold_compliance = {p: (within_threshold_counts[p] / incident_counts[p]) * 100 if resolved_counts[p] > 0 else 0 for p in incident_counts}

        # Calculate officer utilization
        #for station in simulation.officers.values():
        #    for officer in station:
        #        if not officer.available:
        #            total_officer_hours[officer.name] += (simulation.current_time - officer.return_time)

        #officer_utilization = sum(total_officer_hours.values()) / (len(total_officer_hours) * simulation.current_time) * 100
        mean_total_office_hours = sum(total_officer_hours.values())/len(total_officer_hours.values())
        unresolved_incident_percentage = (unresolved_incidents / len(simulation.cumulative_incidents)) * 100 if simulation.cumulative_incidents else 0

        return {
            "Completion Percentages": completion_percentages,
            "Mean Response Times": mean_response_times,
            "Mean Deployment Times": mean_deployment_times,
            "Threshold Compliance": threshold_compliance,
            #"Officer Utilization": officer_utilization,
            "Mean Officer Hours": mean_total_office_hours,
            "Unresolved Incident Percentage": unresolved_incident_percentage
        }

    def check_simulation(self):
        simulation = self
        # Initialize officer assignments based on all shifts
        officer_assignments = {}
        for shift, stations in self.shift_distribution.items():
            for station, num_officers in stations.items():
                for i in range(num_officers):
                    officer_name = f"Officer_{station}_{shift}_{i}"
                    officer_assignments[officer_name] = []

        incident_response = {'Immediate': {'total': 0, 'within_time': 0},
                             'Prompt': {'total': 0, 'within_time': 0},
                             'Standard': {'total': 0, 'within_time': 0}}

        time_travel_occurred = False

        for incident in simulation.resolved_incidents:
            if incident.resolved:
                # Check officer assignments and time traveling
                if incident.resolving_officer:
                    officer_assignments[incident.resolving_officer].append(incident.resolution_time)
                    if len(officer_assignments[incident.resolving_officer]) > 1:
                        if officer_assignments[incident.resolving_officer][-2] > incident.resolution_time:
                            time_travel_occurred = True

                # Count incidents and check response time
                incident_response[incident.priority]['total'] += 1
                target_time = {'Immediate': 1, 'Prompt': 3, 'Standard': 6}[incident.priority]
                if incident.response_time - incident.global_time <= target_time:
                    incident_response[incident.priority]['within_time'] += 1

        # Calculate percentages
        for priority in incident_response:
            total = incident_response[priority]['total']
            if total > 0:
                incident_response[priority]['percentage'] = (incident_response[priority]['within_time'] / total) * 100
            else:
                incident_response[priority]['percentage'] = 0

        return officer_assignments, incident_response, time_travel_occurred

class Simulator:
    SPEED_MPH = 30  # Speed in miles per hour

    def __init__(self, df: pd.DataFrame, police_stations: List[PoliceStation], 
                 shift_distribution: ShiftDistribution, 
                 shift_distribution_weekend: ShiftDistribution,
                 custody_suites_df: pd.DataFrame,
                 custody_inds_df:pd.DataFrame,
                 n_days: float = 7,
                 verbose:int=0):
        
        self.df = df
        self.custody_incidents_df = custody_suites_df
        self.custody_inds_df = custody_inds_df
        
        self.police_stations = police_stations
        self.station_coords = [(ps.location.x, ps.location.y) for ps in self.police_stations]
        self.shift_distribution = shift_distribution
        self.shift_distribution_weekend = shift_distribution_weekend
        self.current_time = 0
        self.officers = { key: [] for key in shift_distribution["Early"].keys() } # Station_1, Station_2 and Station_3
        self.verbose = verbose
        self.n_days = n_days
        self.hours = [i for i in range(24*self.n_days + 1)]
        self.hour_index = 0
        self.return_times = [] # array keeps a track of return times of officers

        # This resolved incidents is made to keep a track of when an incident
        # is allocated to an officer. This is to conduct the evaluating model check
        self.resolved_incidents = [] # This list keeps a track and order of the resolved incidents as they are allocated

        # Making the df compatible with data structures
        df.columns = [x.lower() for x in df.columns]

        self.officers_by_shift = {
            "Weekday":{
                "Day":{},
                "Night":{},
                "Early":{}
            }, 
            
            "Weekend":{
                "Day":{},
                "Night":{},
                "Early":{}  
            }

        }
        

    def print_dashboard(self, allocations):
        print(f"\n--- Day {self.current_time // 24 + 1}, Hour {self.current_time % 24} ---")
        pending_cases = [inc for inc in self.cumulative_incidents if not inc.resolved]
        print(f"Pending Cases: {len(pending_cases)}")
        print("Pending Incident URNs:", [inc.urn for inc in pending_cases])

        inv_allocations = {v: k for k, v in allocations.items()}

        for station, officers in self.officers.items():
            print(f"\n{station}:")
            for officer in officers:
                # Find the incident the officer is currently assigned to
                if officer.name in inv_allocations.keys():
                    status = inv_allocations[officer.name]
                else:
                    status = "Busy"
                print(f"  - {officer.name}: {status}")

    def calculate_distance(self, lat1, lon1, lat2, lon2):
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


    def generate_incidents_for_hour(self):
        for global_time, hour_incidents in self.df.groupby('time'):
            incidents = []

            for urn, row in hour_incidents.iterrows():
                deployment_time = row["deployment_time"]
#                station_cols = [f"station_{i+1}" for i in range(len(self.station_coords))]
                station_cols = [name for name in self.df.columns if "station_" in name]

                distances = {col: row[col] for col in station_cols}

                incidents.append(Incident(**row.drop(["deployment_time", "time", "sno"] + station_cols).to_dict(),
                                          urn=urn, 
                                          deployment_time=deployment_time,
                                          distances=distances,
                                          global_time=global_time))

            yield incidents

    def generate_custodies_for_hour(self):
        for _, division_cincs in self.custody_incidents_df.groupby(['Day', 'Hour']):
            cins = []
            for _, row in division_cincs.iterrows():
                num_incidents = int(row['Total_Custodies'])
                for p in range(num_incidents):
                    urn = f"CI_{row['Day']}_{row['Hour']}_{row['DivisionID']}_{p}"
                    max_travel_time = row['Max_Travel_Time'] 
                    max_wait_time = row['Max_Wait_Time']  
                    max_processing_time = row['Max_Processing_Time'] 
                    deployment_time = max_travel_time+max_wait_time+max_processing_time
                    incident = CustodyIncident(urn=urn, 
                                               hour=row['Hour'], 
                                               day=row['Day'], 
                                               division=row['DivisionID'], 
                                               max_travel_time=max_travel_time, 
                                               max_wait_time=max_wait_time,
                                               max_processing_time=max_processing_time,
                                               global_time=self.current_time)
                    cins.append(incident)
            yield cins
            #yield []   

            
    def data_to_incident(self, urn, row):
        deployment_time = row.pop("deployment_time")
        global_time = row.pop("time")  # Not needed as we use global_time
        station_cols = [f"station_{i+1}" for i in range(len(self.station_coords))]
        distances = row[station_cols].to_dict()
        row = row.drop(station_cols)
        return Incident(**row,
                          urn=urn, 
                          deployment_time=deployment_time,
                          distances=distances,
                          global_time=global_time)
            
    def create_officers_for_shift(self):
        for wd in self.officers_by_shift.keys():
            is_weekday = wd == "Weekday"
            sd = self.shift_distribution if is_weekday else self.shift_distribution_weekend
            for shift in ["Day", "Early", "Night"]:
                for station, num_officers in sd[shift].items():
                    # Update the number of officers for each station
                    day = "Weekday" if is_weekday else "Weekend"
                    self.officers_by_shift[day][shift][station] = [Officer(f"Officer_{station}_{day}_{shift}_{i}", station=station) for i in range(num_officers)]
        
    def add_officers_for_shift(self, shift, is_weekday):
        sd = self.shift_distribution if is_weekday else self.shift_distribution_weekend 
        for station, num_officers in sd[shift].items():
            # Update the number of officers for each station
            day = "Weekday" if is_weekday else "Weekend"
            self.officers[station] += self.officers_by_shift[day][shift][station]
            
    def remove_officers_for_shift(self, shift):
        for station, num_officers in self.shift_distribution[shift].items():
            # Update the number of officers for each station
            self.officers[station] = [off for off in self.officers[station] if shift not in off.name ]

            
    def process_allocations(self, allocations):
        # Prepare a set of urns to remove from pending incidents
        urns_to_remove = set()

        # Build a dictionary for fast officer lookup
        officer_dict = {off.name: off for station_officers in self.officers.values() for off in station_officers}

        # List of updates to perform on the DataFrame
        updates = []
        custody_updates = []

        for urn, officer_id in allocations.items():
           # if urn == "PS-20180105-3585":
            #    print(f'{allocations=}')
            # Access the row as a Series for fast column access
            is_custody_incident = "CI" in urn
            
            if not is_custody_incident:
                incident = self.df.loc[urn]

                if officer_id is None:
                    if not pd.isna(incident["resolved"]):  # If the incident is present and resolved is not NaN
                        self.df.at[urn, "resolved"] = False
                    continue

                officer = officer_dict.get(officer_id)
                if officer and officer.available:
                    travel_time = incident[officer.station] / self.SPEED_MPH
                    officer.available = False  # Mark officer as busy
                    officer.return_time = self.current_time + 2*travel_time + incident["deployment_time"]
                    self.return_times.append(officer.return_time)
                    if (officer.return_time < np.ceil(self.current_time)):
                        self.return_times_within_hour.append(officer.return_time)
                        sorted(self.return_times_within_hour)

                    # Collect the updates to perform in a single DataFrame operation
                    updates.append((urn, 
                                    True, 
                                    officer.name, 
                                    self.current_time, # allocation_time
                                    self.current_time + travel_time, # response_time
                                    officer.return_time # resolution_time
                                   )) 
                    urns_to_remove.add(urn)
            else:
                incident = self.custody_inds_df.loc[urn]
                officer = officer_dict.get(officer_id)
                if officer and officer.available:
                    travel_time = incident["max_travel_time"]
                    officer.available = False  # Mark officer as busy
                    officer.return_time = self.current_time + incident["max_processing_time"] + 2*travel_time + incident["max_wait_time"]
                    self.return_times.append(officer.return_time)
                    if (officer.return_time < np.ceil(self.current_time)):
                        self.return_times_within_hour.append(officer.return_time)
                        sorted(self.return_times_within_hour)
                    self.custody_inds_df.at[urn, "resolution_time"] = officer.return_time - travel_time
                    self.custody_inds_df.at[urn, "resolving_officer"] = officer.name
                    
                    urns_to_remove.add(urn)
                        
                    # Collect updates for custody_df
                

        # Apply the collected updates to the DataFrame
        for urn, resolved, resolving_officer, allocation_time, response_time, resolution_time in updates:
            self.df.at[urn, "resolved"] = True
            self.df.at[urn, "resolving_officer"] = resolving_officer
            self.df.at[urn, "allocation_time"] = allocation_time
            self.df.at[urn, "response_time"] = response_time
            self.df.at[urn, "resolution_time"] = resolution_time
            
        # Update pending_incidents based on urns_to_remove
        self.pending_incidents = [inc for inc in self.pending_incidents if inc.urn not in urns_to_remove]

        # Update officers' availability in officer_dict
        for officer_id in allocations.values():
            officer = officer_dict.get(officer_id)
            if officer:
                officer.available = False  # Mark officer as busy

        return urns_to_remove  # Return urns removed from pending_incidents


    def update_officer_availability(self):
        for station, officers in self.officers.items():
            for officer in officers:
                if self.current_time >= officer.return_time:
                    officer.available = True
                else:
                    officer.available = False 


    def get_return_times_for_next_hour(self):
        global_time = self.current_time
        return_times_for_next_hour = [rt for rt in np.unique(self.return_times) if rt <= global_time + 1]
        return sorted(return_times_for_next_hour)

    
    def schedule_breaks(self, shift, time):
        for station, officers in self.officers.items():
            selective_officers = [off for off in officers if shift in off.name]
            n_to_deactivate = len(selective_officers) // 2
            offs = selective_officers[:n_to_deactivate] if time == 0 else selective_officers[n_to_deactivate:]
            for off in offs:
                if not off.available:
                    off.return_time = off.return_time + 1 
                    self.return_times.append(off.return_time)
                    off.available = False
                else:
                    off.available = False

            self.return_times.append(self.current_time + 1)
                    
    def run(self, model):
        
        # Create all of the officers
        self.create_officers_for_shift()
        
        #self.cumulative_incidents = []  # Global list to track all incidents
        self.pending_incidents = []
        event_generator = self.generate_incidents_for_hour()
        hour_gen = self.generate_custodies_for_hour()
        
        while self.current_time < 24 * self.n_days:  # For a week-long simulation

            day = self.current_time // 24 + 1
            hour = self.current_time % 24
            
            is_weekday = day not in [5, 6] # Sat or Sun

            # Update officer availability at the start of each timestep
            self.update_officer_availability()
            
            if is_weekday:
                # Update officers for shift change
                if day*24+hour == 24:
                    self.add_officers_for_shift("Night", is_weekday)
                    self.current_shift = "Night"
                
                if hour in [7, 14, 22]:
                    add_shift = 'Early' if hour == 7 else 'Day' if hour == 14 else 'Night'
                    self.current_shift = add_shift
                    self.add_officers_for_shift(add_shift, is_weekday)

                if hour in [16, 0, 7]:
                    remove_shift = 'Early' if hour == 16 else 'Day' if hour == 0 else 'Night'
                    self.remove_officers_for_shift(remove_shift)

                    if self.verbose == -1:
                        no_of_officers = len([officer for station in self.officers.values() for officer in station])
                        print(f"{day=} {self.current_shift=} {no_of_officers=}")

                total_officers = sum([len(self.officers[f"station_{i+1}"]) for i in range(len(self.station_coords))])
            else:
                if hour in [7, 16, 21]:
                    add_shift = 'Early' if hour == 7 else 'Day' if hour == 14 else 'Night'
                    self.current_shift = add_shift
                    self.add_officers_for_shift(add_shift, is_weekday)

                if hour in [15, 2, 7]:
                    remove_shift = 'Early' if hour == 16 else 'Day' if hour == 0 else 'Night'
                    self.remove_officers_for_shift(remove_shift)

                    if self.verbose == -1:
                        no_of_officers = len([officer for station in self.officers.values() for officer in station])
                        print(f"{day=} {self.current_shift=} {no_of_officers=}")

                total_officers = sum([len(self.officers[f"station_{i+1}"]) for i in range(len(self.station_coords))])
            
            # Lunch break
            # 7 - 15 : Early
            # 16 - 2 : Day
            # 21 - 7 : Night
            lunch_breaks =  [7, 13] if self.current_shift == "Early" else [16, 20] if self.current_shift == "Day" else [21, 4]
            if hour in lunch_breaks:
                self.schedule_breaks(self.current_shift, lunch_breaks.index(hour))
            
            # Generate and add new incidents
            custody_incidents = next(hour_gen)
            new_incidents = next(event_generator)
            self.pending_incidents.extend(new_incidents)
            self.pending_incidents.extend(custody_incidents)
            
            allocations = model.make_allocation(self.pending_incidents, self.officers, self.current_time)
            
            
            # Process allocations and update the state
            self.process_allocations(allocations)

            # After making the allocations for the hour
            # get to each return time of the officer
            # and make new allocations
            self.return_times_within_hour = self.get_return_times_for_next_hour()

            while self.return_times_within_hour:
                # Ensure the list is sorted
                self.return_times_within_hour.sort()

                # Process the earliest return time within the hour
                self.current_time = self.return_times_within_hour[0]

                # Update officer availability based on the current time
                self.update_officer_availability()

                # Make allocations for pending incidents at the current time
                allocations = model.make_allocation(self.pending_incidents, self.officers, self.current_time)

                # Process allocations and update the state
                self.process_allocations(allocations)

                # Remove the processed time from both lists
                self.return_times.remove(self.current_time)
                self.return_times_within_hour.remove(self.current_time)

            #for time in self.return_times_within_hour:
            #    self.current_time = time
            #    self.update_officer_availability()
            #    #pending_incidents = [inc for inc in self.cumulative_incidents if not inc.resolved]
            #    allocations = model.make_allocation(self.pending_incidents, self.officers, self.current_time)
            #    # Process allocations and update the state
            #    self.process_allocations(allocations)
            #    self.return_times.remove(time)

            if self.verbose > 2:
                self.print_dashboard(allocations)


            # Pending cases after allocation
            self.hour_index += 1
            self.current_time = self.hours[self.hour_index]

            if self.verbose > 3:
                input("Press Enter to continue to the next hour...\n")

    # Checks and Analysis
    def analyze_simulation_results(self):
        simulation = self

        # Initialize dictionaries to collect data
        incident_counts = {'Immediate': 0, 'Prompt': 0, 'Standard': 0, "Other resolution": 0}
        resolved_counts = {'Immediate': 0, 'Prompt': 0, 'Standard': 0, "Other resolution": 0}
        within_threshold_counts = {'Immediate': 0, 'Prompt': 0, 'Standard': 0, "Other resolution": 0}
        response_times = {'Immediate': [], 'Prompt': [], 'Standard': [], "Other resolution": []}
        deployment_times = {'Immediate': [], 'Prompt': [], 'Standard': [], "Other resolution": []}
        #thresholds = {'Immediate': 1, 'Prompt': 3, 'Standard': 6, "Other resolution": 20}
        thresholds = {'Immediate': 1/12, 'Prompt': 2/3, 'Standard': 1000, "Other resolution": 1000}
        total_officer_hours = {}
        unresolved_incidents = 0

        # Group incidents by priority
        days_filter = (simulation.df["time"] < 24 * self.n_days)
        grouped = simulation.df[days_filter].groupby('priority')

        # Process each group
        for priority, incidents in grouped:
            incident_counts[priority] = len(incidents)

            # Filter resolved incidents
            resolved_incidents = incidents[incidents["resolved"] == True]
            resolved_counts[priority] = len(resolved_incidents)

            response_times[priority] = resolved_incidents["response_time"] - resolved_incidents["allocation_time"]
            deployment_times[priority] = resolved_incidents["deployment_time"]

            for officer in resolved_incidents["resolving_officer"].unique():
                officer_incidents = resolved_incidents[resolved_incidents["resolving_officer"] == officer]
                total_time = (officer_incidents["resolution_time"] - officer_incidents["allocation_time"]).sum()
                if officer not in total_officer_hours:
                    total_officer_hours[officer] = 0
                total_officer_hours[officer] += total_time

            # Calculate the time from incident report to response arrival
            time_to_response = (resolved_incidents["response_time"] - resolved_incidents["allocation_time"])

            # Check if the response was within the threshold
            within_threshold_counts[priority] = (time_to_response <= thresholds[priority]).sum()

            unresolved_incidents += len(incidents[incidents["resolved"] == False])

        # Calculate percentages and mean times
        completion_percentages = {p: (resolved_counts[p] / incident_counts[p]) * 100 if incident_counts[p] > 0 else 0 for p in incident_counts}
        mean_response_times = {p: np.mean(response_times[p]) if len(response_times[p]) > 0 else 0 for p in response_times}
        mean_deployment_times = {p: np.mean(deployment_times[p]) if len(deployment_times[p]) > 0 else 0 for p in deployment_times}
        threshold_compliance = {p: (within_threshold_counts[p] / incident_counts[p]) * 100 if incident_counts[p] > 0 else 0 for p in incident_counts}

        # Calculate mean officer hours
        mean_total_officer_hours = sum(total_officer_hours.values()) / len(total_officer_hours) if total_officer_hours else 0

        unresolved_incident_percentage = (unresolved_incidents / len(simulation.df)) * 100 if len(simulation.df) > 0 else 0

        return {
            "Completion Percentages": completion_percentages,
            "Mean Response Times": mean_response_times,
            "Mean Deployment Times": mean_deployment_times,
            "Threshold Compliance": threshold_compliance,
            "Mean Officer Hours": mean_total_officer_hours,
            "Unresolved Incident Percentage": unresolved_incident_percentage
        }

    def check_simulation(self):
        simulation = self
        # Initialize officer assignments based on all shifts
        officer_assignments = {}
        for shift, stations in self.shift_distribution.items():
            for station, num_officers in stations.items():
                for i in range(num_officers):
                    officer_name = f"Officer_{station}_Weekday_{shift}_{i}"
                    officer_assignments[officer_name] = []
                    
        for shift, stations in self.shift_distribution_weekend.items():
            for station, num_officers in stations.items():
                for i in range(num_officers):
                    officer_name = f"Officer_{station}_Weekend_{shift}_{i}"
                    officer_assignments[officer_name] = []

        incident_response = {'Immediate': {'total': 0, 'within_time': 0},
                             'Prompt': {'total': 0, 'within_time': 0},
                             'Standard': {'total': 0, 'within_time': 0}, 
                             'Other resolution': {'total': 0, 'within_time': 0}}

        time_travel_occurred = False

        resolved_incidents = self.df["resolved"] == True
        for urn, incident in self.df[resolved_incidents].sort_values(["allocation_time", "response_time"]).iterrows():
            # Check officer assignments and time traveling
            if incident["resolving_officer"]:
                officer_assignments[incident["resolving_officer"]].append(incident["resolution_time"])
                if len(officer_assignments[incident["resolving_officer"]]) > 1:
                    if officer_assignments[incident["resolving_officer"]][-2] > incident["resolution_time"]:
                        time_travel_occurred = True

            # Count incidents and check response time
            incident_response[incident.priority]['total'] += 1
            #target_time = {'Immediate': 1, 'Prompt': 3, 'Standard': 6, "Other resolution":20}[incident.priority]
            target_time =  {'Immediate': 1/12, 'Prompt': 2/3, 'Standard': 1000, "Other resolution": 1000}[incident.priority]

            if incident["response_time"] - incident["allocation_time"] <= target_time:
                incident_response[incident["priority"]]['within_time'] += 1

        # Calculate percentages
        for priority in incident_response:
            total = incident_response[priority]['total']
            if total > 0:
                incident_response[priority]['percentage'] = (incident_response[priority]['within_time'] / total) * 100
            else:
                incident_response[priority]['percentage'] = 0

        return officer_assignments, incident_response, time_travel_occurred

# class Simulator:
#     """
    
#     The Simulator class performs the simulation based on an input Model.
#     A Model is the brain that does the allocation of police officers based
#     on the task queue and current state of the simulation (police officers free, 
#     police stations in which officers are located, etc. )
    

#     """
#     SPEED_MPH = 30  # Speed in miles per hour

#     def __init__(self, df: pd.DataFrame, station_coords: List[tuple], shift_distribution: ShiftDistribution, verbose:int=0):
#         self.df = df.copy()
#         self.station_coords = station_coords
#         self.shift_distribution = shift_distribution
#         self.current_time = 0
#         self.officers = { key: [] for key in shift_distribution["Early"].keys() } # Station_1, Station_2 and Station_3
#         self.verbose = verbose
#         self.hours = [i for i in range(24*7 + 1)]
#         self.hour_index = 0
#         self.return_times = [] # array keeps a track of return times of officers

#         # This resolved incidents is made to keep a track of when an incident
#         # is allocated to an officer. This is to conduct the evaluating model check
#         self.resolved_incidents = [] # This list keeps a track and order of the resolved incidents as they are allocated


#         # Making the df compatible with data structures
#         df.columns = [x.lower() for x in df.columns]


#     def print_dashboard(self, allocations: Dict[str, Officer]):
#         """
#         print_dashboard print the current state of the simulation.
#         """

#         print(f"\n--- Day {self.current_time // 24 + 1}, Hour {self.current_time % 24} ---")
#         pending_cases = [inc for inc in self.cumulative_incidents if not inc.resolved]
#         print(f"Pending Cases: {len(pending_cases)}")
#         print("Pending Incident URNs:", [inc.urn for inc in pending_cases])

#         inv_allocations = {v: k for k, v in allocations.items()}

#         for station, officers in self.officers.items():
#             print(f"\n{station}:")
#             for officer in officers:
#                 # Find the incident the officer is currently assigned to
#                 if officer.name in inv_allocations.keys():
#                     status = inv_allocations[officer.name]
#                 else:
#                     status = "Busy"
#                 print(f"  - {officer.name}: {status}")

#     def calculate_distance(self, lat1:float, lon1:float, lat2:float, lon2:float):
#         """
#         Calculate the great circle distance between two points
#         on the earth (specified in decimal degrees)
#         """
#         # Convert decimal degrees to radians
#         lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

#         # Haversine formula
#         dlat = lat2 - lat1
#         dlon = lon2 - lon1
#         a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
#         c = 2 * asin(sqrt(a))
#         r = 3956  # Radius of Earth in miles. Use 6371 for kilometers
#         return c * r

#     def generate_incidents_for_hour(self, global_time:float):
#         """
#         This function filters the incidents that arise in the
#         next hour of the global time while the simulation is 
#         running. 
#         """
#         hour_incidents = self.df[(self.df['time'] == global_time)]
#         incidents = []
#         for _, row in hour_incidents.iterrows():
#             deployment_time = row.pop("deployment time (hrs)")
#             row.pop("time")  # Not needed as we use global_time
#             distances = {f'Station_{i+1}': self.calculate_distance(row['latitude'], row['longitude'], lat, lon)
#                          for i, (lat, lon) in enumerate(self.station_coords)}
#             incidents.append(Incident(**row,
#                                       deployment_time=deployment_time,
#                                       distances=distances,
#                                       global_time=global_time))
#         return incidents

#     def update_officers_for_shift(self, shift:Shift):
#         """
#         This function assigns officers for each police station for the shift. 
#         """
#         for station, num_officers in self.shift_distribution[shift].items():
#             # Update the number of officers for each station
#             self.officers[station] = [Officer(f"Officer_{station}_{shift}_{i}", station=station) for i in range(num_officers)]

#     def process_allocations(self, allocations: Dict[str, Officer]):
#         """
#         This function uses the allocations provided to it by the
#         Model and implements the allocation as and when they come.
#         """
#         for urn, officer_id in allocations.items():
#             incident = next((inc for inc in self.cumulative_incidents if inc.urn == urn), None)
#             #print("incident:", incident)
#             if officer_id is None:
#                 if incident:
#                     incident.resolved = False
#                 continue

#             officer = next((off for station_officers in self.officers.values() for off in station_officers if off.name == officer_id), None)
#             if incident and officer:
#                 travel_time = incident.distances[officer.station] / self.SPEED_MPH
#                 officer.available = False  # Mark officer as busy
#                 # Travel time for going and coming back is added 
#                 officer.return_time = self.current_time +incident.deployment_time + 2*travel_time 
#                 self.return_times.append(officer.return_time)

#                 incident.resolved = True  # Mark incident as resolved
#                 incident.resolving_officer = officer.name  # Assign officer to incident
#                 incident.allocation_time = self.current_time
#                 incident.response_time = self.current_time + travel_time # Global time when the response reached
#                 incident.resolution_time = officer.return_time # Global Time when the incident was resolved
#                 self.resolved_incidents.append(incident)

#     def update_officer_availability(self):
#         """
#         Updates the status of officer to available or busy. 
#         """
#         for station, officers in self.officers.items():
#             for officer in officers:
#                 if not officer.available and self.current_time >= officer.return_time:
#                     officer.available = True

#     def update_lunch_availability(self):
#         return 

#     def get_return_times_for_next_hour(self):
#         """
#         Obtains the return time - , i.e, the time taken by an officer
#         to return from completing a task. 
#         """
#         global_time = self.current_time
#         return_times_for_next_hour = [rt for rt in self.return_times if rt <= global_time + 1]
#         return sorted(return_times_for_next_hour)


#     def run(self, model:Model):
#         """
#         This function runs the simulation. 
#         """

#         self.cumulative_incidents = []  # Global list to track all incidents

#         while self.current_time < 24 * 7:  # For a week-long simulation

#             day = self.current_time // 24 + 1
#             hour = self.current_time % 24

#             # Assume day 4 - Friday, 5 - Saturday and 6 - Sunday
#             is_weekday = day in [4, 5, 6]

#             # Update officer availability at the start of each timestep
#             self.update_officer_availability()

#             # Update officers for shift change
#             # On weekdays
#             if is_weekday:
#                 print(f"{is_weekday=} {day=}")
                
#             if hour in [0, 8, 16]:
#                 shift = 'Early' if hour == 0 else 'Day' if hour == 8 else 'Night'
#                 self.update_officers_for_shift(shift)
#                 if self.verbose == -1:
#                     no_of_officers = len([officer for station in self.officers.values() for officer in station])
#                     print(f"{day=} {shift=} {no_of_officers=}")

#             total_officers = len(self.officers["Station_1"]) + len(self.officers["Station_2"]) + len(self.officers["Station_3"])

#             # Adds a lunch break for the officers
#             self.update_lunch_availability()

#             # Generate and add new incidents
#             new_incidents = self.generate_incidents_for_hour(self.current_time)
#             self.cumulative_incidents.extend(new_incidents)

#             # Filter to get only pending incidents
#             pending_incidents = [inc for inc in self.cumulative_incidents if not inc.resolved]

#             allocations = model.make_allocation(pending_incidents, self.officers, self.current_time)

#             # Process allocations and update the state
#             self.process_allocations(allocations)

#             # After making the allocations for the hour
#             # get to each return time of the officer
#             # and make new allocations
#             return_times_within_hour = self.get_return_times_for_next_hour()

#             for time in return_times_within_hour:
#                 self.current_time = time
#                 self.update_officer_availability()
#                 pending_incidents = [inc for inc in self.cumulative_incidents if not inc.resolved]
#                 allocations = model.make_allocation(pending_incidents, self.officers, self.current_time)
                
#                 # Process allocations and update the state
#                 self.process_allocations(allocations)
#                 self.return_times.remove(time)

#             if self.verbose > 2:
#                 self.print_dashboard(allocations)


#             # Pending cases after allocation
#             pending_incidents = [inc for inc in self.cumulative_incidents if not inc.resolved]

#             #self.current_time += 1  # Move to the next hour
#             self.hour_index += 1
#             self.current_time = self.hours[self.hour_index]

#             if self.verbose > 3:
#                 input("Press Enter to continue to the next hour...\n")

#     # Checks and Analysis
#     def analyze_simulation_results(self):
#         """
#         A function that returns relevant infometrics about the simulation.
#         """

#         simulation = self

#         # Initialize counters and accumulators
#         incident_counts = {'Immediate': 0, 'Prompt': 0, 'Standard': 0}
#         resolved_counts = {'Immediate': 0, 'Prompt': 0, 'Standard': 0}
#         within_threshold_counts = {'Immediate': 0, 'Prompt': 0, 'Standard': 0}
#         response_times = {'Immediate': [], 'Prompt': [], 'Standard': []}
#         resolution_times = {'Immediate': [], 'Prompt': [], 'Standard': []}
#         deployment_times = {'Immediate': [], 'Prompt': [], 'Standard': []}
#         thresholds = {'Immediate': 1, 'Prompt': 3, 'Standard': 6}
#         total_officer_hours = {}
#         unresolved_incidents = 0

#         # Analyze each incident
#         for incident in simulation.cumulative_incidents:
#             priority = incident.priority
#             incident_counts[priority] += 1

#             if incident.resolved:
#                 resolved_counts[priority] += 1

#                 response_time = incident.response_time
#                 resolution_time = incident.resolution_time

#                 response_times[priority].append(response_time)
#                 resolution_times[priority].append(resolution_time)

#                 deployment_time = incident.deployment_time
#                 deployment_times[priority].append(deployment_time)

#                 if incident.resolving_officer not in total_officer_hours.keys():
#                     total_officer_hours[incident.resolving_officer]  = 0

#                 time_spent_on_incident = incident.resolution_time - incident.global_time
#                 total_officer_hours[incident.resolving_officer] += time_spent_on_incident

#                 # Calculate the time from incident report to response arrival
#                 time_to_response = incident.response_time - incident.global_time


#                 # Check if the response was within the threshold
#                 if time_to_response <= thresholds[priority]:
#                     within_threshold_counts[priority] += 1
#             else:
#                 unresolved_incidents += 1

#         # Calculate percentages and mean times
#         completion_percentages = {p: (resolved_counts[p] / incident_counts[p]) * 100 if incident_counts[p] > 0 else 0 for p in incident_counts}
#         mean_response_times = {p: np.mean(response_times[p]) if response_times[p] else 0 for p in response_times}
#         mean_deployment_times = {p: np.mean(deployment_times[p]) if deployment_times[p] else 0 for p in deployment_times}
#         threshold_compliance = {p: (within_threshold_counts[p] / incident_counts[p]) * 100 if resolved_counts[p] > 0 else 0 for p in incident_counts}

#         # Calculate officer utilization
#         #for station in simulation.officers.values():
#         #    for officer in station:
#         #        if not officer.available:
#         #            total_officer_hours[officer.name] += (simulation.current_time - officer.return_time)

#         #officer_utilization = sum(total_officer_hours.values()) / (len(total_officer_hours) * simulation.current_time) * 100
#         mean_total_office_hours = sum(total_officer_hours.values())/len(total_officer_hours.values())
#         unresolved_incident_percentage = (unresolved_incidents / len(simulation.cumulative_incidents)) * 100 if simulation.cumulative_incidents else 0

#         return {
#             "Completion Percentages": completion_percentages,
#             "Mean Response Times": mean_response_times,
#             "Mean Deployment Times": mean_deployment_times,
#             "Threshold Compliance": threshold_compliance,
#             #"Officer Utilization": officer_utilization,
#             "Mean Officer Hours": mean_total_office_hours,
#             "Unresolved Incident Percentage": unresolved_incident_percentage
#         }

#     def check_simulation(self):
#         """
#         This function checks if the all the requirements / checks implemented in the
#         simulation are being met. 
#         """
#         simulation = self
#         # Initialize officer assignments based on all shifts
#         officer_assignments = {}
#         for shift, stations in self.shift_distribution.items():
#             for station, num_officers in stations.items():
#                 for i in range(num_officers):
#                     officer_name = f"Officer_{station}_{shift}_{i}"
#                     officer_assignments[officer_name] = []

#         incident_response = {'Immediate': {'total': 0, 'within_time': 0},
#                              'Prompt': {'total': 0, 'within_time': 0},
#                              'Standard': {'total': 0, 'within_time': 0}}

#         time_travel_occurred = False

#         for incident in simulation.resolved_incidents:
#             if incident.resolved:
#                 # Check officer assignments and time traveling
#                 if incident.resolving_officer:
#                     officer_assignments[incident.resolving_officer].append(incident.resolution_time)
#                     if len(officer_assignments[incident.resolving_officer]) > 1:
#                         if officer_assignments[incident.resolving_officer][-2] > incident.resolution_time:
#                             time_travel_occurred = True

#                 # Count incidents and check response time
#                 incident_response[incident.priority]['total'] += 1
#                 target_time = {'Immediate': 1, 'Prompt': 3, 'Standard': 6}[incident.priority]
#                 if incident.response_time - incident.global_time <= target_time:
#                     incident_response[incident.priority]['within_time'] += 1

#         # Calculate percentages
#         for priority in incident_response:
#             total = incident_response[priority]['total']
#             if total > 0:
#                 incident_response[priority]['percentage'] = (incident_response[priority]['within_time'] / total) * 100
#             else:
#                 incident_response[priority]['percentage'] = 0

#         return officer_assignments, incident_response, time_travel_occurred


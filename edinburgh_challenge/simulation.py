import numpy as np
from math import radians, cos, sin, asin, sqrt
import pandas as pd

from dataclasses import dataclass
from typing import List, Dict
import numpy as np
from abc import ABC, abstractmethod

class Model(ABC):
    @abstractmethod
    def make_allocation(self, incidents, officers, current_time):
        pass

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

@dataclass
class Officer:
    name: str
    station: str
    available: bool = True
    return_time: float = 0.0

from typing import Dict, List

# Shift distribution structure
# Example: {'Early': {'Station_1': 5, 'Station_2': 5, 'Station_3': 5}, ...}
ShiftDistribution = Dict[str, Dict[str, int]]

from math import radians, cos, sin, asin, sqrt

class Simulation:
    SPEED_MPH = 30  # Speed in miles per hour

    def __init__(self, df: pd.DataFrame, station_coords: List[tuple], shift_distribution: ShiftDistribution, verbose:int=0):
        self.df = df.copy()
        self.station_coords = station_coords
        self.shift_distribution = shift_distribution
        self.current_time = 0
        self.officers = { key: [] for key in shift_distribution["Early"].keys() } # Station_1, Station_2 and Station_3
        self.verbose = verbose

        # This resolved incidents is made to keep a track of when an incident
        # is allocated to an officer. This is to conduct the evaluating model check
        self.resolved_incidents = [] # This list keeps a track and order of the resolved incidents as they are allocated
        self.working_hours = self.make_officer_working_hours()

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
        for urn, officer_id in allocations.items():
            incident = next((inc for inc in self.cumulative_incidents if inc.urn == urn), None)
            if officer_id is None:
                if incident:
                    incident.resolved = False
                continue

            officer = next((off for station_officers in self.officers.values() for off in station_officers if off.name == officer_id), None)
            if incident and officer:
                travel_time = incident.distances[officer.station] / self.SPEED_MPH
                officer.available = False  # Mark officer as busy
                officer.return_time = self.current_time + travel_time + incident.deployment_time
                self.working_hours[officer.name] += officer.return_time - self.current_time

                incident.resolved = True  # Mark incident as resolved
                incident.resolving_officer = officer.name  # Assign officer to incident
                incident.response_time = self.current_time + travel_time # Global time when the response reached
                incident.resolution_time = officer.return_time # Global Time when the incident was resolved

                self.resolved_incidents.append(incident)


    def update_officer_availability(self):
        for station, officers in self.officers.items():
            for officer in officers:
                if not officer.available and self.current_time >= officer.return_time:
                    officer.available = True


    def run(self, model):
        self.cumulative_incidents = []  # Global list to track all incidents

        while self.current_time < 24 * 7:  # For a week-long simulation

            day = self.current_time // 24 + 1
            hour = self.current_time % 24

            # Update officer availability attotal_officer_hours the start of each timestep
            self.update_officer_availability()

            # Update officers for shift change
            if hour in [0, 8, 16]:
                shift = 'Early' if hour == 0 else 'Day' if hour == 8 else 'Night'
                self.update_officers_for_shift(shift)

            total_officers = len(self.officers["Station_1"]) + len(self.officers["Station_2"]) + len(self.officers["Station_3"])

            # Generate and add new incidents
            new_incidents = self.generate_incidents_for_hour(self.current_time)
            self.cumulative_incidents.extend(new_incidents)

            # Filter to get only pending incidents
            pending_incidents = [inc for inc in self.cumulative_incidents if not inc.resolved]
            allocations = model.make_allocation(pending_incidents, self.officers, self.current_time)

            # Process allocations and update the state
            self.process_allocations(allocations)

            if self.verbose > 2:
                self.print_dashboard(allocations)

            # Pending cases after allocation
            pending_incidents = [inc for inc in self.cumulative_incidents if not inc.resolved]

            self.current_time += 1  # Move to the next hour
            if self.verbose > 3:
                input("Press Enter to continue to the next hour...\n")


    def make_officer_working_hours(self):
        officer_assignments = {}
        for shift, stations in self.shift_distribution.items():
            for station, num_officers in stations.items():
                for i in range(num_officers):
                    officer_name = f"Officer_{station}_{shift}_{i}"
                    officer_assignments[officer_name] = 0.0
        return officer_assignments

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
        total_officer_hours = {officer.name: 0 for station in simulation.officers.values() for officer in station}
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
        threshold_compliance = {p: (within_threshold_counts[p] / resolved_counts[p]) * 100 if resolved_counts[p] > 0 else 0 for p in incident_counts}

        # Calculate officer utilization
        officer_utilization = sum(self.working_hours.values()) / (len(self.working_hours.values()))

        unresolved_incident_percentage = (unresolved_incidents / len(simulation.cumulative_incidents)) * 100 if simulation.cumulative_incidents else 0

        return {
            "Completion Percentages": completion_percentages,
            "Mean Response Times": mean_response_times,
            "Mean Deployment Times": mean_deployment_times,
            "Threshold Compliance": threshold_compliance,
            "Mean Officer Time": officer_utilization,
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

class SimulationWithMaxUtilisation:
    SPEED_MPH = 30  # Speed in miles per hour

    def __init__(self, df: pd.DataFrame, station_coords: List[tuple], shift_distribution: ShiftDistribution, verbose:int=0):
        self.df = df.copy()
        self.station_coords = station_coords
        self.shift_distribution = shift_distribution
        self.current_time = 0
        self.officers = { key: [] for key in shift_distribution["Early"].keys() } # Station_1, Station_2 and Station_3
        self.verbose = verbose
        self.hours = [i for i in range(24*7 + 1)]
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

        while self.current_time < 24 * 7:  # For a week-long simulation

            day = self.current_time // 24 + 1
            hour = self.current_time % 24

            # Update officer availability at the start of each timestep
            self.update_officer_availability()

            # Update officers for shift change
            if hour in [0, 8, 16]:
                shift = 'Early' if hour == 0 else 'Day' if hour == 8 else 'Night'
                self.update_officers_for_shift(shift)

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

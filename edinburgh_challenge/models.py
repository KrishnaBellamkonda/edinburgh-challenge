from abc import ABC, abstractmethod
from collections import defaultdict
from math import floor, ceil, radians, cos, sin, asin, sqrt
import heapq
import numpy as np

class Model(ABC):
    @abstractmethod
    def make_allocation(self, incidents, officers, current_time):
        pass



class NaiveModel():
    def make_allocation(self, incidents, officers, current_time):
        # Sort incidents by priority
        incidents.sort(key=lambda inc: inc.priority)

        allocations = {}
        officers_allocated = []
        for inc in incidents:
            allocated = False
            # Sort stations by distance to the incident
            sorted_stations = sorted(inc.distances, key=inc.distances.get)

            for station in sorted_stations:
                # Check for available officer in the station
                available_officers = [off for off in officers[station] if (off.available and off not in officers_allocated) ]
                if available_officers:
                    # Allocate the first available officer
                    chosen_officer = available_officers[0]
                    allocations[inc.urn] = chosen_officer.name
                    officers_allocated.append(chosen_officer)
                    allocated = True
                    break

            if not allocated:
                # No officers available for this incident
                allocations[inc.urn] = None

        return allocations

class EnhancedModel(Model):
    def make_allocation(self, incidents, officers, current_time):
        # Adjusting the priority mechanism to balance between priority, waiting time, and travel time
        incidents.sort(key=lambda inc: (inc.priority, current_time - inc.global_time, min(inc.distances.values())))

        allocations = {}
        allocated_officers = set()  # Set to keep track of officers already allocated

        for inc in incidents:
            # Find the nearest station with available officers
            nearest_stations = sorted(inc.distances, key=inc.distances.get)

            for station in nearest_stations:
                # Filter out officers who are already allocated
                available_officers = [off for off in officers[station] if off.available and off.name not in allocated_officers]

                if available_officers:
                    # Allocate the first available officer
                    chosen_officer = available_officers[0]
                    allocations[inc.urn] = chosen_officer.name
                    allocated_officers.add(chosen_officer.name)  # Mark officer as allocated
                    chosen_officer.available = False  # Mark officer as busy
                    # Assuming return_time is calculated elsewhere
                    break
            else:
                # No officers available for this incident
                allocations[inc.urn] = None

        return allocations

class SimplifiedModelNotBest(NaiveModel):

    SPEED_MPH = 30

    def __init__(self, shift_distribution, police_stations_dict):
        super().__init__()
        self.shift_distribution = shift_distribution
        self.incident_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.resolution_times = {"Immediate":[], "Standard":[], "Prompt":[]}
        self.processed_incidents = set()
        self.police_stations_dict = police_stations_dict

    def update_incident_count(self, incidents):
        """
        Update the count of incidents for each hour of each day.
        Only update for new incidents.
        """
        for incident in incidents:
            if incident.urn not in self.processed_incidents:
                day = incident.day
                hour = incident.hour
                priority = incident.priority
                self.incident_counts[day][hour][priority] += 1
                self.processed_incidents.add(incident.urn)

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


    def get_incident_resolution_time(self, incident, officer, current_time):
        officer_police_station_location = self.police_stations_dict[officer.station]
        distance_to_incident = self.calculate_distance(officer_police_station_location.x,
                                                  officer_police_station_location.y,
                                                  incident.latitude,
                                                  incident.longitude
                                                 )
        travel_time = distance_to_incident / self.SPEED_MPH
        deployment_time = incident.deployment_time
        resolution_time = current_time + travel_time + deployment_time
        return resolution_time

    def get_incident_priority_count(self, incidents):
        counts = defaultdict(int)
        for inc in incidents:
            if not inc.resolved:
                counts[inc.priority] += 1
        return counts

    def get_no_available_officers_next_hour(self, officers, current_time):
        next_hour = ceil(current_time)
        no_available_officers = 0
        for station in officers.values():
            for officer in station:
                if officer.available:
                    no_available_officers += 1
                else:
                    if officer.return_time <= next_hour:
                        no_available_officers += 1
        return no_available_officers


    def get_mean_resolution_time(self):
        return {
            "Immediate":np.mean(self.resolution_times["Immediate"]),
            "Prompt":np.mean(self.resolution_times["Prompt"]),
            "Standard": np.mean(self.resolution_times["Standard"])
        }

    def make_allocation(self, incidents, officers, current_time):
        self.update_incident_count(incidents)

        day = current_time // 24 + 1
        hour = floor(current_time % 24)
        incidents.sort(key=lambda inc: inc.priority)

        if day == 1:
            allocations = {}
            officers_allocated = []
            for inc in incidents:
                allocated = False
                # Sort stations by distance to the incident
                sorted_stations = sorted(inc.distances, key=inc.distances.get)

                for station in sorted_stations:
                    # Check for available officer in the station
                    available_officers = [off for off in officers[station] if (off.available and off not in officers_allocated) ]

                    if available_officers:
                        # Allocate the first available officer
                        chosen_officer = available_officers[0]
                        allocations[inc.urn] = chosen_officer.name
                        officers_allocated.append(chosen_officer)
                        allocated = True

                        # If allocated, calculate the
                        # total resolution time
                        # and save it in a dictionary
                        resolution_time = self.get_incident_resolution_time(inc, chosen_officer, current_time)
                        self.resolution_times[inc.priority].append(resolution_time)
                        break

                if not allocated:
                    # No officers available for this incident
                    allocations[inc.urn] = None
        else:

            # Get an estimate of the number of previous immediate, promt
            # and standard cases
            next_immediate_cases = self.incident_counts[day-1][hour]["Immediate"]
            next_prompt_cases = self.incident_counts[day-1][hour]["Prompt"]
            next_standard_cases = self.incident_counts[day-1][hour]["Standard"]
            next_necessary_cases = next_immediate_cases + next_prompt_cases

            # Current number of cases
            incident_counts = self.get_incident_priority_count(incidents)
            now_immediate_cases = incident_counts["Immediate"]
            now_prompt_cases = incident_counts["Prompt"]
            now_standard_cases = incident_counts["Standard"]
            now_necessary_cases = now_immediate_cases + now_standard_cases

            no_available_officers_next_hour = self.get_no_available_officers_next_hour(officers, current_time)

            print(no_available_officers_next_hour)
            print(f"{now_necessary_cases=}")
            print(f"{next_necessary_cases=}")

            # If a standard case is being assigned, and we expect a more important prompt case to
            # be issued in the next hour, we wait.
            allocations = {}
            officers_allocated = []
            for inc in incidents:
                if inc.priority == "Standard":
                    pass
                else:
                    allocated = False
                    # Sort stations by distance to the incident
                    sorted_stations = sorted(inc.distances, key=inc.distances.get)

                    for station in sorted_stations:
                        # Check for available officer in the station
                        available_officers = [off for off in officers[station] if (off.available and off not in officers_allocated) ]

                        if available_officers:
                            # Allocate the first available officer
                            chosen_officer = available_officers[0]
                            allocations[inc.urn] = chosen_officer.name
                            officers_allocated.append(chosen_officer)
                            allocated = True

                            # If allocated, calculate the
                            # total resolution time
                            # and save it in a dictionary
                            resolution_time = self.get_incident_resolution_time(inc, chosen_officer, current_time)
                            self.resolution_times[inc.priority].append(resolution_time)
                            break

        return allocations


class SimplifiedModel(NaiveModel):

    SPEED_MPH = 30

    def __init__(self, shift_distribution, police_stations_dict):
        super().__init__()
        self.shift_distribution = shift_distribution
        self.incident_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.resolution_times = {"Immediate":[], "Standard":[], "Prompt":[]}
        self.processed_incidents = set()
        self.police_stations_dict = police_stations_dict

    def update_incident_count(self, incidents):
        """
        Update the count of incidents for each hour of each day.
        Only update for new incidents.
        """
        for incident in incidents:
            if incident.urn not in self.processed_incidents:
                day = incident.day
                hour = incident.hour
                priority = incident.priority
                self.incident_counts[day][hour][priority] += 1
                self.processed_incidents.add(incident.urn)

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


    def get_incident_resolution_time(self, incident, officer, current_time):
        officer_police_station_location = self.police_stations_dict[officer.station]
        distance_to_incident = self.calculate_distance(officer_police_station_location.x,
                                                  officer_police_station_location.y,
                                                  incident.latitude,
                                                  incident.longitude
                                                 )
        travel_time = distance_to_incident / self.SPEED_MPH
        deployment_time = incident.deployment_time
        resolution_time = current_time + travel_time + deployment_time
        return resolution_time

    def get_incident_priority_count(self, incidents):
        counts = defaultdict(int)
        for inc in incidents:
            if not inc.resolved:
                counts[inc.priority] += 1
        return counts

    def get_no_available_officers_next_hour(self, officers, current_time):
        next_hour = ceil(current_time)
        no_available_officers = 0
        for station in officers.values():
            for officer in station:
                if officer.available:
                    no_available_officers += 1
                else:
                    if officer.return_time <= next_hour:
                        no_available_officers += 1
        return no_available_officers


    def get_mean_resolution_time(self):
        return {
            "Immediate":np.mean(self.resolution_times["Immediate"]),
            "Prompt":np.mean(self.resolution_times["Prompt"]),
            "Standard": np.mean(self.resolution_times["Standard"])
        }


class GreedyModel(NaiveModel):

    SPEED_MPH = 30

    def __init__(self, shift_distribution, police_stations_dict):
        super().__init__()
        self.shift_distribution = shift_distribution
        self.incident_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.resolution_times = {"Immediate":[], "Standard":[], "Prompt":[]}
        self.processed_incidents = set()
        self.police_stations_dict = police_stations_dict

    def update_incident_count(self, incidents):
        """
        Update the count of incidents for each hour of each day.
        Only update for new incidents.
        """
        for incident in incidents:
            if incident.urn not in self.processed_incidents:
                day = incident.day
                hour = incident.hour
                priority = incident.priority
                self.incident_counts[day][hour][priority] += 1
                self.processed_incidents.add(incident.urn)

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


    def get_incident_resolution_time(self, incident, officer, current_time):
        officer_police_station_location = self.police_stations_dict[officer.station]
        distance_to_incident = self.calculate_distance(officer_police_station_location.x,
                                                  officer_police_station_location.y,
                                                  incident.latitude,
                                                  incident.longitude
                                                 )
        travel_time = distance_to_incident / self.SPEED_MPH
        deployment_time = incident.deployment_time
        resolution_time = current_time + travel_time + deployment_time
        return resolution_time

    def get_incident_priority_count(self, incidents):
        counts = defaultdict(int)
        for inc in incidents:
            if not inc.resolved:
                counts[inc.priority] += 1
        return counts

    def get_no_available_officers_next_hour(self, officers, current_time):
        next_hour = ceil(current_time)
        no_available_officers = 0
        for station in officers.values():
            for officer in station:
                if officer.available:
                    no_available_officers += 1
                else:
                    if officer.return_time <= next_hour:
                        no_available_officers += 1
        return no_available_officers


    def get_mean_resolution_time(self):
        return {
            "Immediate":np.mean(self.resolution_times["Immediate"]),
            "Prompt":np.mean(self.resolution_times["Prompt"]),
            "Standard": np.mean(self.resolution_times["Standard"])
        }

    def predict_peak_time_adjustments(self, weights_dict, current_time):
        # Initialize weights

        # Adjust weights based on peak hours and days
        if current_time % 24 in [8, 9, 10]:
            # Adjust the weight for Immediate incidents during peak hours
            weights_dict['Immediate'] *= 1.2  # Adjust the weight as needed

        if current_time // 24 + 1 in [1, 2]:
            # Adjust the weight for Prompt incidents on peak days
            weights_dict['Prompt'] *= 1.2

        if current_time % 24 in [15, 16]:
            # Adjust the weight for Standard incidents during peak hours
            weights_dict['Standard'] *= 1.2

        return self.normalise_weights(weights_dict)

    def normalise_weights(self, weights_dict):
        total_weight = sum(weights_dict.values())
        normalized_weights_dict = {incident_type: weight / total_weight for incident_type, weight in weights_dict.items()}
        return normalized_weights_dict

    def make_allocation(self, incidents, officers, current_time):
        self.update_incident_count(incidents)

        # Define thresholds and priority weights for each priority
        thresholds = {'Immediate': 1, 'Prompt': 3, 'Standard': 6}

        time_remaining_factor = 1.0  # Adjust the weight for time remaining
        priority_weights = {'Immediate': 8.7, 'Prompt': 5, 'Standard': 1}  # Adjusted weights
        priority_weights = self.normalise_weights(priority_weights)

        # Adjust priority weights based on predicted peak times
        priority_weights = self.predict_peak_time_adjustments(priority_weights, current_time)

        # Function to calculate the score based on distance and priority
        def calculate_score(travel_time, time_since_reported, priority):
            time_remaining = thresholds[priority] - time_since_reported - travel_time
            urgency = time_remaining_factor/time_remaining
            #urgency = (time_since_reported + travel_time) / thresholds[priority]
            return urgency * priority_weights[priority]

        # Get all available officers
        available_officers = [off for station_officers in officers.values() for off in station_officers if off.available]

        allocations = {}

        for officer in available_officers:
            officer_station = officer.station
            officer_station_location = self.police_stations_dict[officer_station]
            incident_queue = []

            for inc in incidents:
                if inc.urn not in allocations:  # Only consider unallocated incidents
                    travel_time = self.calculate_distance(
                        officer_station_location.x, officer_station_location.y,
                        inc.latitude, inc.longitude) / self.SPEED_MPH
                    time_since_reported = current_time - inc.global_time
                    score = calculate_score(travel_time, time_since_reported, inc.priority)
                    heapq.heappush(incident_queue, (-score, inc.urn, inc))  # Using negative score for max heap

            # Allocate this officer to the most urgent incident
            if incident_queue:
                _, _, most_urgent_incident = heapq.heappop(incident_queue)
                allocations[most_urgent_incident.urn] = officer.name
                travel_time = self.calculate_distance(
                    officer_station_location.x, officer_station_location.y,
                    most_urgent_incident.latitude, most_urgent_incident.longitude) / self.SPEED_MPH
                resolution_time = current_time + travel_time + most_urgent_incident.deployment_time
                self.resolution_times[most_urgent_incident.priority].append(resolution_time)

        # Mark unallocated incidents
        for inc in incidents:
            if inc.urn not in allocations:
                allocations[inc.urn] = None

        return allocations



# Note: The rest of the SimplifiedModel class remains unchanged.

# Explanation:
# The calculate_score function considers both the travel time and the time since the incident was reported,
# and combines them with the adjusted priority weights.
# A higher score indicates a higher priority for allocation.
# This should improve the allocation of officers by balancing the urgency of incidents with their distance from available officers.

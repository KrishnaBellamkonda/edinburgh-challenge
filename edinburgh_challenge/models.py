from abc import ABC, abstractmethod


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

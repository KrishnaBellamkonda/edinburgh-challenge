import numpy as np
import pandas as pd
import datetime
from dataclasses import dataclass

from edinburgh_challenge.utility import bng_to_latlong
from edinburgh_challenge.constants import Location

@dataclass
class PoliceStation:
    location: Location
    division: str
    name: str
    n_officers: int
    simulation_name: str = ""

# Utility
def haversine(lat1, lon1, lat2, lon2):
    """
    Vectorized Haversine formula to calculate the great circle distance
    between two points on the earth (specified in decimal degrees).
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 3956  # Radius of Earth in miles. Use 6371 for kilometers
    return c * r



#
def calculate_metric(results_dict):
    # Immediate cases
    immediate_completion_pct = results_dict["Completion Percentages"]["Immediate"]
    immediate_threshold_pct = results_dict["Threshold Compliance"]["Immediate"]
    immediate_deployment_time = results_dict["Mean Deployment Times"]["Immediate"]

    prompt_completion_pct = results_dict["Completion Percentages"]["Prompt"]
    prompt_threshold_pct = results_dict["Threshold Compliance"]["Prompt"]
    prompt_deployment_time = results_dict["Mean Deployment Times"]["Prompt"]

    standard_completion_pct = results_dict["Completion Percentages"]["Standard"]
    standard_threshold_pct = results_dict["Threshold Compliance"]["Standard"]
    standard_deployment_time = results_dict["Mean Deployment Times"]["Standard"]



    immediate_portion = (immediate_threshold_pct/100)
    prompt_portion = (prompt_threshold_pct/100)
    standard_portion = (standard_threshold_pct/100)

    # Immediate
    metric = (3*immediate_portion + 1.5*prompt_portion + 1*standard_portion)/(5.5)

    return metric

def calculate_simulation_performance(results_dict, verbose = False):
    # Information from the results analysis
    immediate_completion_pct = results_dict["Completion Percentages"]["Immediate"]
    immediate_threshold_pct = results_dict["Threshold Compliance"]["Immediate"]

    prompt_completion_pct = results_dict["Completion Percentages"]["Prompt"]
    prompt_threshold_pct = results_dict["Threshold Compliance"]["Prompt"]

    standard_completion_pct = results_dict["Completion Percentages"]["Standard"]
    standard_threshold_pct = results_dict["Threshold Compliance"]["Standard"]

    mean_officer_hours  = results_dict["Mean Officer Hours"]

    # Rescaling these values
    immediate_completion_pct /= 100
    prompt_completion_pct /= 100
    standard_completion_pct /= 100

    immediate_threshold_pct /= 100
    prompt_threshold_pct /= 100
    standard_threshold_pct /= 100

    immediate_incompletion_pct = 1 - immediate_completion_pct
    prompt_incompletion_pct = 1- prompt_completion_pct
    standard_incompletion_pct = 1 - standard_completion_pct

    # Calculating the score

    # First factor - Incident resolved within threshold (Scale - 0 to 1)
    incident_within_threshold = (2*immediate_threshold_pct + 1.5*prompt_threshold_pct + 1*standard_threshold_pct)/(4.5)

    # Second factor - Officer utilisation
    # 8 hours per shift, 7 days in the simulation (Scale - 0 to 1)
    officer_utilisation = (mean_officer_hours)/(8*7 +1)

    # Third factor - Unresolved Incidents (Scale - 0 to 1)
    unresolved_incidents = ((6*immediate_incompletion_pct)+ 2*(prompt_incompletion_pct) + 1*(standard_incompletion_pct))/9

    if verbose:
        print(f"{incident_within_threshold=}")
        print(f"{officer_utilisation=}")
        print(f"{unresolved_incidents=}")

    # Total scale, (0 to 1)
    performance_metric = 0.8*incident_within_threshold + 0.2*officer_utilisation - unresolved_incidents*0.3
    return performance_metric


# Loading Datasets
def load_incidents_df(incidents_source = "data/Incidents2018.csv"):
    incidents_df = pd.read_csv(incidents_source)
    incidents_df["DATETIME_CREATED"] = pd.to_datetime(incidents_df["DATETIME_CREATED"])
    incidents_df.sort_values("DATETIME_CREATED", inplace=True)
    return incidents_df

def load_deployment_df(avg_deployment_source = "data/WEST_AVERAGE_DEPLOYMENT_byDIVISIONandDATE_MINUTES.csv"):
    return pd.read_csv(avg_deployment_source)

def load_estates_df(estates_dataset_source = "data/Estate_Informationv3.xlsx"):
    estates_df = pd.read_excel(estates_dataset_source)
    return estates_df

def load_max_custody_df(max_custody_source = "data/Max Custody Times.csv"):
    max_custody_df = pd.read_csv(max_custody_source)
    max_custody_df["Date1"] = pd.to_datetime(max_custody_df["Date1"])
    
    # Columns to impute
    cols_to_impute = ['Max_Travel_Time', 'Max_Wait_Time', 'Max_Processing_Time']

    # Group by 'divisionid' and 'division_name', then apply mean imputation within each group
    for col in cols_to_impute:
        max_custody_df[col] = max_custody_df.groupby(['Custody_Suite_Name_Full'])[col].transform(lambda x: x.fillna(x.mean()))

        
    return max_custody_df

# Data Cleaning
def fill_deployment_from_avg_values(incidents_df, avg_deploment_df):
    # Turn the avg deployment DataFrame to a dictionary
    avg_deployment_dict = avg_deploment_df.set_index(["date_", "DivisionID"])["AVG_Deployment_Time_Mins"].to_dict()

    # Processing on Incidents Dataset
    incidents_df.loc[:, "Date"] = incidents_df["DATETIME_CREATED"].dt.strftime("%Y-%m-%d")
    missing_deployment = incidents_df["Deployment_Time_Mins"].isna()

    # Identify missing dates
    incidents_dates = incidents_df["Date"].unique()
    avg_deployment_dates = avg_deploment_df["date_"].unique()
    missing_dates = [d for d in incidents_dates if d not in avg_deployment_dates]

    # Calculate the mean deployment time for incidents with non-null values for the missing dates
    mean_deployment_times = incidents_df[~incidents_df["Deployment_Time_Mins"].isna()].groupby(["Date", "DivisionID"])["Deployment_Time_Mins"].mean()

    for date in missing_dates:
        for division_id in incidents_df["DivisionID"].unique():
            mean_deployment_time = mean_deployment_times.get((date, division_id), None)
            if mean_deployment_time is not None:
                avg_deployment_dict[(date, division_id)] = mean_deployment_time

    def deployment_look_up(row):
        return avg_deployment_dict.get((row["Date"], row["DivisionID"]), None)

    # Apply the lookup only to rows with missing deployment time
    incidents_df.loc[missing_deployment, "Deployment_Time_Mins"] = incidents_df[missing_deployment].apply(deployment_look_up, axis=1)

    return incidents_df


def calculate_events_per_division(incidents_df, avg_deploment_df):
    # Processing on Incidents Dataset
    incidents_df["Date"] = incidents_df["DATETIME_CREATED"].dt.strftime("%Y-%m-%d")

    # Identify missing dates
    incidents_dates = incidents_df["Date"].unique()
    avg_deployment_dates = avg_deploment_df["date_"].unique()
    missing_dates = [d for d in incidents_dates if d not in avg_deployment_dates]

    # Group by Date and DivisionID to count the number of events
    event_counts = incidents_df.groupby(["Date", "DivisionID"]).size().reset_index(name='Event_Count')

    # Filter the counts for the missing dates
    missing_event_counts = event_counts[event_counts["Date"].isin(missing_dates)]

    return missing_event_counts


# Preprocessing
def preprocess_incidents(incidents_df, avg_deployment_df, ps_coords,  n_values=1_000_000, start_date = False):

    # Cleanup incidents dataset
    incidents_df = fill_deployment_from_avg_values(incidents_df, avg_deployment_df)

    # Renaming columns
    incidents_df = incidents_df.rename(columns={
        "Deployment_Time_Mins":"Deployment Time (hrs)",
        "GIS_LATITUDE":"Latitude",
        "GIS_LONGITUDE":"Longitude",
        "Call_Priority":"Priority",
        "ISR_NO":"URN"
    })

    # Making a Global Time Metric
    if not start_date:
        start_date = incidents_df["DATETIME_CREATED"].min().date()
    else:
        format = "%d-%m-%Y"
        start_date = datetime.datetime.strptime(start_date, format)
    day_of_week = start_date.weekday() + 1 # Finally, Monday = 1

    start_date_dt = datetime.datetime.combine(start_date, datetime.datetime.min.time())
    timedelta = (incidents_df["DATETIME_CREATED"] - start_date_dt)

    incidents_df["Day"] = (timedelta.dt.days) + day_of_week # Start day is Monday
    
    # Select values
    non_zero_filter = (incidents_df["Day"] > 0)
    incidents_df = incidents_df[non_zero_filter]
    incidents_df = incidents_df.iloc[:n_values]
    
    incidents_df["Hour"] = timedelta.dt.seconds // 3600

    incidents_df["Time"] = (incidents_df["Day"]-1)*24 + incidents_df["Hour"]

    incidents_df["Priority"] = incidents_df["Priority"].str.capitalize()

    incidents_df = incidents_df[["URN", "Latitude", "Longitude", "Day", "Hour", "Time", "Priority", "Deployment Time (hrs)"]]
    incidents_df.columns = [x.lower() for x in incidents_df.columns]

    # Adding distance columns
    for i, (station_lat, station_lon) in enumerate(ps_coords):
        incidents_df[f'Station_{i+1}'] = haversine(incidents_df['latitude'].values, incidents_df['longitude'].values, station_lat, station_lon)

    # 
    incidents_df.columns = [x.lower() for x in incidents_df.columns]
    incidents_df["sno"] = range(1, len(incidents_df) + 1)
    incidents_df.rename(columns={"deployment time (hrs)":"deployment_time"}, inplace=True)
    incidents_df["deployment_time"] /= 60
    incidents_df["resolved"] = False
    incidents_df["resolving_officer"] = None
    incidents_df["allocation_time"] = None
    incidents_df["response_time"] = None
    incidents_df["resolution_time"] = None
    incidents_df.set_index("urn", inplace=True, drop=True)

    return incidents_df

def process_custody_df(max_custody_df, start_date = False):
    # From Date1 to Global Days
    max_custody_df.sort_values(["Date1"])
    if not start_date:
        start_date = max_custody_df["Date1"].min().date()
    else:
        format = "%d-%m-%Y"
        start_date = datetime.datetime.strptime(start_date, format)

    day_of_week = start_date.weekday() + 1 # Finally, Monday = 1
    start_date_dt = datetime.datetime.combine(start_date, datetime.datetime.min.time())

    max_custody_df["Day"] = (max_custody_df["Date1"] - start_date_dt).dt.days + day_of_week
    
    non_zero_filter = (max_custody_df["Day"] > 0)
    max_custody_df = max_custody_df[non_zero_filter]

    max_custody_df[["Max_Travel_Time", "Max_Wait_Time", "Max_Processing_Time"]] = max_custody_df[["Max_Travel_Time", "Max_Wait_Time", "Max_Processing_Time"]] / 60
    
    custody_grouped_df = max_custody_df.groupby(['Day', 'DivisionID'])['Total_Custodies', "Max_Travel_Time", "Max_Wait_Time", "Max_Processing_Time"].sum().reset_index()
    
    shift_proportions = [0.2, 0.4, 0.4]
    hours_per_shift = 8

    def distribute_n_cus(df):
        hourly_data = []
        for _, row in df.iterrows():
            n_cus = row['Total_Custodies']
            day = row['Day']
            division = row['DivisionID']
            travel_time = row['Max_Travel_Time']
            wait_time = row['Max_Wait_Time']
            processing_time = row['Max_Processing_Time']

            for shift_idx, shift_prop in enumerate(shift_proportions):
                shift_n_cus = n_cus * shift_prop
                hourly_n_cus = shift_n_cus / hours_per_shift

                for hour in range(shift_idx * hours_per_shift, (shift_idx + 1) * hours_per_shift):
                    hourly_data.append({
                        'Day': day, 
                        'Hour': hour + 1, 
                        'DivisionID': division, 
                        'Total_Custodies': hourly_n_cus, 
                        'Max_Travel_Time':travel_time,
                        'Max_Wait_Time':wait_time,
                        'Max_Processing_Time':processing_time,                    
                    })

        l = pd.DataFrame(hourly_data)
        l["Total_Custodies"] = np.round( l["Total_Custodies"]).astype(int)
        return l
    
    hourly_df = distribute_n_cus(custody_grouped_df)
    
    # List to hold expanded rows
    expanded_rows = []

    # Iterate over each row in hourly_df
    for _, row in hourly_df.iterrows():
        day = row['Day']
        hour = row['Hour']
        division = row['DivisionID']
        n_cus = int(row['Total_Custodies'])  # Convert n_cus to integer
        max_travel_time = row['Max_Travel_Time']
        max_processing_time = row['Max_Processing_Time']
        max_wait_time = row['Max_Wait_Time']
        
        
        # Create n_cus incidents with the same day, hour, and division
        for p in range(n_cus):
            urn = f"CI_{day}_{hour}_{division}_{p}"
            # Append each incident as a dictionary to the list
            expanded_rows.append({
                'urn': urn,
                'hour': hour,
                'day': day,
                'division': division,
                'max_travel_time': max_travel_time,
                'max_wait_time': max_wait_time,
                'max_processing_time': max_processing_time,
                'deployment_time': max_travel_time + max_wait_time + max_processing_time
            })
    
    n = pd.DataFrame(expanded_rows)
    n["Resolution_Time"] = None
    
    return hourly_df, n.set_index("urn")


def generate_simulation_specs(estates_df):
    police_station_filter = estates_df["Property Classification"] == "Police Station"
    local_policing_filter = estates_df["Primary Classification (Use)"] == "Local Policing"
    deployment_filter = estates_df["Deployment station? Y/N"] == "Yes"
    estates_df = estates_df[police_station_filter & local_policing_filter & deployment_filter]
    response_officers_count = estates_df["DPU_ResponseOfficers_Count"]

    def convert_dataframe(df):
        """
        Convert a DataFrame with Easting and Northing columns to Latitude and Longitude.

        :param df: DataFrame with columns 'X' (Easting) and 'Y' (Northing)
        :return: DataFrame with additional columns 'Latitude' and 'Longitude'
        """
        latitudes = []
        longitudes = []

        for index, row in df.iterrows():
            lat, lon = bng_to_latlong(row['X (Easting)'], row['Y (Northing)'])
            latitudes.append(lat)
            longitudes.append(lon)

        df['Latitude'] = latitudes
        df['Longitude'] = longitudes

        return df

    df_converted = convert_dataframe(estates_df)
    ps_data = df_converted[["Site Name", "Latitude", "Longitude", "DPU_ResponseOfficers_Count", "Division"]]
    ps_data = ps_data.rename(columns={"Site Name":"Name", "DPU_ResponseOfficers_Count":"n_officers"})
    police_stations = [ PoliceStation(location=Location(x=row["Latitude"], y=row["Longitude"]), division=row["Division"], name=row["Name"], simulation_name=f"station_{n+1}", n_officers=row["n_officers"]) for n, (_, row) in enumerate(ps_data.iterrows())]
    police_stations_dict = {f"station_{n+1}": Location(x=row["Latitude"], y=row["Longitude"]) for n, (_,row) in enumerate(ps_data.iterrows())}

    shift_distribution = {
    "Early":{f"station_{n+1}":int(n_officers*0.2) for n, n_officers  in enumerate(ps_data["n_officers"])},
    "Day":{f"station_{n+1}":int(n_officers*0.5) for n,n_officers  in enumerate(ps_data["n_officers"])},
    "Night":{f"station_{n+1}":int(n_officers*0.3) for n,n_officers  in enumerate(ps_data["n_officers"])},
    }

    shift_distribution_weekend = {
    "Early":{f"station_{n+1}":int(n_officers*0.2) for n, n_officers  in enumerate(ps_data["n_officers"])},
    "Day":{f"station_{n+1}":int(n_officers*0.4) for n,n_officers  in enumerate(ps_data["n_officers"])},
    "Night":{f"station_{n+1}":int(n_officers*0.4) for n,n_officers  in enumerate(ps_data["n_officers"])},
    }

    ps_coords = [ (p.x, p.y) for p in
                police_stations_dict.values()]

    return ps_coords, shift_distribution, shift_distribution_weekend, police_stations_dict, police_stations

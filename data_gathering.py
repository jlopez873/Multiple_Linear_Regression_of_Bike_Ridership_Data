import time
import os
from glob2 import glob
import numpy as np
import pandas as pd

import seaborn as sns
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set(style="white", rc=custom_params)
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['figure.figsize'] = [16,5]

import warnings
warnings.filterwarnings('ignore')

def calc_progress(processed_files, total_files, start_time):
    """
    Calculate and print the progress information.

    Args:
        processed_files (int): Number of files processed so far.
        total_files (int): Total number of files being processed.
        start_time (float): Start time of the file processing.

    """
    elapsed_time = time.time() - start_time
    avg_time_per_file = elapsed_time / processed_files
    remaining_files = total_files - processed_files
    estimated_remaining_time = avg_time_per_file * remaining_files
    
    # Convert elapsed time to hours, minutes, and seconds
    elapsed_hours, elapsed_rem = divmod(elapsed_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_rem, 60)
    elapsed_hours, elapsed_minutes, elapsed_seconds = int(elapsed_hours), int(elapsed_minutes), int(elapsed_seconds)

    # Convert average time per file to hours, minutes, and seconds
    avg_hours, avg_rem = divmod(avg_time_per_file, 3600)
    avg_minutes, avg_seconds = divmod(avg_rem, 60)
    avg_hours, avg_minutes, avg_seconds = int(avg_hours), int(avg_minutes), int(avg_seconds)

    # Convert estimated remaining time to hours, minutes, and seconds
    est_hours, est_rem = divmod(estimated_remaining_time, 3600)
    est_minutes, est_seconds = divmod(est_rem, 60)
    est_hours, est_minutes, est_seconds = int(est_hours), int(est_minutes), int(est_seconds)

    # Print the elapsed time, average time per file, and estimated remaining time
    print(f"Elapsed Time:             {elapsed_hours:02d}:{elapsed_minutes:02d}:{elapsed_seconds:02d}")
    print(f"Avg. Time per File:       {avg_hours:02d}:{avg_minutes:02d}:{avg_seconds:02d}")
    print(f"Estimated Remaining Time: {est_hours:02d}:{est_minutes:02d}:{est_seconds:02d}\n")
    
def process_data(files):
    """
    Process data from a list of files and return a consolidated DataFrame.

    Args:
        files (list): List of file paths to be processed.

    Returns:
        pandas.DataFrame: A DataFrame containing consolidated data.
    """
    # Create an empty DataFrame to compile data
    tripdata = pd.DataFrame()

    # Initialize the start time
    start_time = time.time()

    # Iterate over each file
    for i, file in enumerate(files):
        print(f"Processing file {i+1}/{len(files)}")
        df = pd.read_parquet(file)

        # Calculate the ETA
        calc_progress(i+1, len(files), start_time)

        # Standardize names and drop segmented columns
        if 'starttime' in df.columns:
            # Rename columns
            df.rename(columns={
                'start station id': 'start_station_id',
                'start station name': 'start_station_name',
                'start station latitude': 'start_lat',
                'start station longitude': 'start_lng',
                'end station id': 'end_station_id',
                'end station name': 'end_station_name',
                'end station latitude': 'end_lat',
                'end station longitude': 'end_lng'
            }, inplace=True)

            # Drop columns
            df.drop(['stoptime', 'tripduration', 'bikeid', 'birth year', 'gender'], axis=1, inplace=True)

            # Map usertype column values
            df['usertype'] = df['usertype'].map({'Subscriber': 'member', 'Customer': 'casual'})

        if 'started_at' in df.columns:
            # Rename columns
            df.rename(columns={
                'started_at': 'starttime',
                'member_casual': 'usertype'
            }, inplace=True)

            # Drop columns
            df.drop(['ride_id', 'rideable_type', 'ended_at'], axis=1, inplace=True)

        # Map station_id columns and add borough data
        df['start_station_id'] = df['start_station_name'].map(map_station_id)
        df['end_station_id'] = df['end_station_name'].map(map_station_id)
        df['start_borough'] = df['start_station_id'].map(map_borough)
        df['end_borough'] = df['end_station_id'].map(map_borough)

        # Drop station name and spatial columns
        df.drop(['start_station_id', 'end_station_id', 'start_station_name', 'end_station_name', 'start_lat', 'start_lng',
                 'end_lat', 'end_lng'], axis=1, inplace=True)

        # Convert to datetime
        df['starttime'] = pd.to_datetime(df['starttime'])

        # Set a datetime index
        df.set_index('starttime', inplace=True)

        # Concatenate the clean data to the tripdata
        tripdata = pd.concat([tripdata, df])
        
    return tripdata
    
def consolidate_data(tripdata):
    """
    Consolidate data in the tripdata DataFrame and perform data cleaning.

    Args:
        tripdata (pandas.DataFrame): The DataFrame containing trip data.

    Returns:
        pandas.DataFrame: A cleaned and consolidated DataFrame.
    """
    # Get the number of original observations
    og_rows = len(tripdata)
    print(f"Total Rows: {og_rows:,.0f}")
    
    # **Exclude observations originating or ending in the Bronx, New Jersey, or Unknown**
    tripdata = tripdata[(tripdata['start_borough'] != 'Bronx') & (tripdata['start_borough'] != 'New Jersey') & 
            (tripdata['start_borough'] != 'Unknown') & (tripdata['end_borough'] != 'Bronx') & 
            (tripdata['end_borough'] != 'New Jersey') & (tripdata['end_borough'] != 'Unknown')]
    
    # Remove the Bronx, Unknown, and New Jersey categories
    for col in ['start', 'end']:
        tripdata[f'{col}_borough'].cat.remove_categories(['Bronx', 'Unknown', 'New Jersey'], inplace=True)
        
    # Get the number of observations after removing New Jersey stations
    no_nj = len(tripdata)

    # Calculate the number of observations removed
    nj_dropped = og_rows - no_nj

    print(f"Rows Removed: {nj_dropped:,.0f}")
    print(f"Rows Remaining: {no_nj:,.0f}")
    
    print(f"Percent Loss: {(nj_dropped/og_rows)*100:.2f}%")
    
    tripdata.isna().sum()
    
    display(
        pd.DataFrame.from_records(
            [(col, tripdata[col].nunique(), tripdata[col].dtype, 
              round((tripdata[col].memory_usage(deep=True)/1024)/1024, 2)) for col in tripdata.columns], 
            columns=['Column Name', 'Unique', 'Data Type','Memory Usage']
        )
    )
    
    return tripdata
    
def main():
    """
    Main function to execute data processing and consolidation.
    """
    # Get a list of files in the directory
    files = sorted(glob(os.path.join('../raw_data/bikeshare', '*.parquet')))
    
    # Gather station data
    stations = pd.read_parquet('../clean_data/bike_stations.parquet')
    
    # Create a dictionary from station name and station id
    map_station_id = stations[['station_name', 'station_id']].set_index('station_name')['station_id'].to_dict()
    
    # Create a dictionary from station id and borough
    map_borough = stations[['station_id', 'borough']].set_index('station_id')['borough'].to_dict()
    
    # Convert data types
    for col in ['start_borough', 'end_borough']:
        tripdata[col] =tripdata[col].fillna('Unknown')
        tripdata[col] =tripdata[col].astype('category')
        
    tripdata['usertype'] = tripdata['usertype'].astype('category')
    
    # Consolidate the tripdata and measure loss
    tripdata = consolidate_data(tripdata)
    
    # Save the consolidated data
    tripdata.to_parquet('../clean_data/clean_tripdata.parquet')
    
if __name__ == '__main__':
    main()
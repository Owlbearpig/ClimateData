from pathlib import Path
import os
from datetime import datetime
import numpy as np
import pandas as pd
from functions import find_files
import matplotlib.pyplot as plt
from sklearn.cluster import k_means

PROJECT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))


class TDSDataPoint:
    # thz data for a single measurement, data+timestamp from file.
    def __init__(self, filepath):
        self.filepath = filepath
        self.date = self.getdate()
        self.data = self.parse_data()
        self.relative_time = None

    def __repr__(self):
        return repr(str(self.date))

    def getdate(self):
        s = str(self.filepath.stem)
        s_split = s.split('_')

        return datetime.strptime(s_split[0]+s_split[1], '%Y%m%d%H%M%S')

    def parse_data(self):
        df = pd.read_csv(self.filepath, sep='	', skiprows=3)
        data = np.array(df)  # same arrays as taipan. t = data[:, 0], y = data[:, 1]
        # can add preprocessing function here. Offset correction, bandpass filter, ...

        return data


class ClimateLogEntry:
    # parses climate data, stores as one object
    def __init__(self, file_line):
        self.file_line = file_line
        self.date, self.temp, self.rh, self.ah, self.idx = self.parse_line()

    def __repr__(self):
        return repr(f'Date: {str(self.date)}, Temp: {self.temp} deg, RH: {self.rh}%, AH: {self.ah}%, index: {self.idx}')

    def parse_line(self):
        # splits line into useful parts
        l = str(self.file_line)
        l = l.replace('         ', '')
        l_split = l.split(',')

        date = datetime.strptime(l_split[0], '%Y%m%d%H%M%S.0')
        temperature = float(l_split[1])
        relative_humidity = float(l_split[2])
        absolute_humidity = float(l_split[3])
        index = int(l_split[4])

        return date, temperature, relative_humidity, absolute_humidity, index


class DataPoint:
    # combined thz and climate data
    def __init__(self, tds_data_point, climate_log_entry):
        self.tds_data = tds_data_point
        self.climate_data = climate_log_entry
        self.measurement_type = ''
        self.nearest_ref_dp = None

    def __repr__(self):
        return repr(str(self.tds_data.date))


def parse_tds_data_files(data_dir):
    # find all thz tds data files and read content into TDSDataPoints (One file, one TDSDataPoint).
    tds_data_files = find_files(data_dir, file_extension='.dat')

    return [TDSDataPoint(file) for file in tds_data_files]


def parse_humidity_files(data_dir):
    # find all rpie humidity log files and parse readlines into individual ClimateLogEntries.
    humidity_files = find_files(data_dir, file_extension='.txt', search_str='Humidity')

    climate_log_entries = []
    for humidity_file in humidity_files:
        with open(humidity_file) as file:
            lines = file.readlines()
            for line in lines:
                climate_log_entries.append(ClimateLogEntry(line))

    return climate_log_entries


def match_files(tds_data_points, climate_entries):
    # loop through each tds data point and find the nearest climate log entry comparing dates -> combine into DataPoint
    data_points = []

    for tds_data_point in tds_data_points:

        smallest_time_diff, nearest_climate_entry = np.inf, None
        for climate_entry in climate_entries:
            abs_time_diff = abs((tds_data_point.date - climate_entry.date).total_seconds())
            if abs_time_diff < smallest_time_diff:
                smallest_time_diff, nearest_climate_entry = abs_time_diff, climate_entry

        data_points.append(DataPoint(tds_data_point, nearest_climate_entry))

    return data_points


def determine_measurement_type(data_points):
    # https://realpython.com/k-means-clustering-python/, check measurement_type_plot ;) (3 clusters)
    means = np.array([])
    for data_point in data_points:
        tds_data = data_point.tds_data.data
        abs_mean = np.mean(np.abs(tds_data[:, 1]))

        means = np.append(means, abs_mean)

    threshold = 0.00100 # Set to none when adding new tds data files to check if threshold works...
    if not threshold:
        clusters = k_means(means.reshape(-1, 1), n_clusters=3)
        cluster_centers = clusters[0].flatten()

        plt.scatter(range(len(means)), means)
        for cluster_center in cluster_centers:
            plt.axhline(y=cluster_center, color='r', linestyle='-')
        plt.show()

    # loop through points again and decide of above or below threshold (or invalid, some files are 0...)
    for data_point in data_points:
        tds_data = data_point.tds_data.data
        abs_mean = np.mean(np.abs(tds_data[:, 1]))

        if np.isclose(abs_mean, 0):
            data_point.measurement_type = 'Invalid'
        elif abs_mean < threshold:
            data_point.measurement_type = 'Sample'
        elif abs_mean > threshold:
            data_point.measurement_type = 'Reference'


def add_relative_time(data_points):
    sorted_data_points = sorted(data_points, key=lambda dp: dp.tds_data.date)
    for dp in sorted_data_points:
        dp.tds_data.relative_time = dp.tds_data.date - sorted_data_points[0].tds_data.date

    return sorted_data_points


def add_ref_to_samples(data_points):
    ref_list = [dp for dp in data_points if dp.measurement_type == 'Reference']
    for dp in data_points:
        if dp.measurement_type == 'Reference':
            dp.nearest_ref_dp = dp
        else:
            ref_time_diffs = np.array([(dp.tds_data.date-ref_dp.tds_data.date).total_seconds() for ref_dp in ref_list])
            dp.nearest_ref_dp = ref_list[np.argmin(np.abs(ref_time_diffs))]


def main():
    data_dir = PROJECT_DIR / 'Data'

    tds_data_points = parse_tds_data_files(data_dir)

    climate_entries = parse_humidity_files(data_dir)

    matched_data_points = match_files(tds_data_points, climate_entries)

    determine_measurement_type(matched_data_points)

    add_ref_to_samples(matched_data_points)

    return add_relative_time(matched_data_points)


if __name__ == '__main__':
    """
    Missing offset correction and frequency filter, but shouldn't be too hard to add (look in TDSDataPoint).
    
    data_dir: top level data(rpi climate, Menlo data) directory
    
    What this(main()) should do:
    1. Find all thz data files and parse each file (parse_tds_data_files)
    
    2. Find all climate data files and parse each line of each file 
    into a ClimateLogEntry object thingy (parse_humidity_files)
    
    3. Go through each thz data file and find the nearest climate log entry (wrt. time) 
    and combine each into a single DataPoint (match_files)
    
    4. Add an attribute to each DataPoint which says 
    if it's a Reference, Sample or Invalid(later ones look weird : ) ) measurement (determine_measurement_type)
    
    5. Go through all the DataPoints again and find the nearest reference DataPoint
    then add it as an attribute (add_ref_to_samples)
    
    6. Eh, last step is probably not really needed, it just sorts the points and adds an attribute 
    showing the time difference to the very first measurement (add_relative_time)
    
    """
    sorted_data_points = main()

    for dp in sorted_data_points:
        print(dp, dp.measurement_type)
        print(dp.climate_data, '\n')
        thz_data = dp.tds_data.data
        """
        plt.plot(thz_data[:, 0], thz_data[:, 1])
        plt.show()
        """

    temp = [dp.climate_data.temp for dp in sorted_data_points]

    plt.plot(temp)
    plt.show()

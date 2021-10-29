from pathlib import Path
import os
from datetime import datetime
import numpy as np
import pandas as pd
from functions import find_files
import matplotlib.pyplot as plt

PROJECT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))


class TDSDataPoint:
    def __init__(self, filepath):
        self.filepath = filepath
        self.date = self.getdate()
        self.data = self.parse_data()

    def __repr__(self):
        return repr(str(self.date))

    def getdate(self):
        s = str(self.filepath.stem)
        s_split = s.split('_')

        return datetime.strptime(s_split[0]+s_split[1], '%Y%m%d%H%M%S')

    def parse_data(self):
        df = pd.read_csv(self.filepath, sep='	', skiprows=3)

        return np.array(df)


class ClimateLogEntry:
    def __init__(self, file_line):
        self.date, self.temp, self.rh, self.ah, self.idx = self.parse_line(file_line)

    def __repr__(self):
        return repr(f'Date: {str(self.date)}, Temp: {self.temp} deg, RH: {self.rh}%, AH: {self.ah}%, index: {self.idx}')

    def parse_line(self, file_line):
        l = str(file_line)
        l = l.replace('         ', '')
        l_split = l.split(',')

        date = datetime.strptime(l_split[0], '%Y%m%d%H%M%S.0')
        temperature = float(l_split[1])
        relative_humidity = float(l_split[2])
        absolute_humidity = float(l_split[3])
        index = int(l_split[4])

        return date, temperature, relative_humidity, absolute_humidity, index


class DataPoint:
    def __init__(self, tds_data_point, climate_log_entry):
        self.tds_data = tds_data_point
        self.climate_data = climate_log_entry


def parse_tds_data_files(data_dir):
    tds_data_files = find_files(data_dir, file_extension='.dat')

    return [TDSDataPoint(file) for file in tds_data_files]

def parse_humidity_files(data_dir):
    humidity_files = find_files(data_dir, file_extension='.txt', search_str='Humidity')

    climate_log_entries = []
    for humidity_file in humidity_files:
        with open(humidity_file) as file:
            lines = file.readlines()
            for line in lines:
                climate_log_entries.append(ClimateLogEntry(line))

    return climate_log_entries


def match_files(tds_data_points, climate_entries):
    data_points = []

    for tds_data_point in tds_data_points:

        smallest_time_diff, nearest_climate_entry = np.inf, None
        for climate_entry in climate_entries:
            abs_time_diff = abs((tds_data_point.date - climate_entry.date).seconds)
            if abs_time_diff < smallest_time_diff:
                smallest_time_diff = abs_time_diff
                nearest_climate_entry = climate_entry

        data_points.append(DataPoint(tds_data_point, nearest_climate_entry))

    return data_points

if __name__ == '__main__':
    data_dir = PROJECT_DIR / 'Data'

    tds_data_points = parse_tds_data_files(data_dir)

    climate_entries = parse_humidity_files(data_dir)

    matched_data_points = match_files(tds_data_points, climate_entries)

    for data_point in matched_data_points:
        print(data_point.tds_data.date, data_point.climate_data.date)

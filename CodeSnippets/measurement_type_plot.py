import matplotlib.pyplot as plt
import numpy as np
from main import PROJECT_DIR, parse_tds_data_files, parse_humidity_files, match_files

data_dir = PROJECT_DIR / 'Data'

tds_data_points = parse_tds_data_files(data_dir)

climate_entries = parse_humidity_files(data_dir)

matched_data_points = match_files(tds_data_points, climate_entries)

if __name__ == '__main__':
    means = []
    for data_point in matched_data_points:
        mean = np.mean(np.abs(data_point.tds_data.data[:, 1]))

        if (mean > 0.0011190) and (mean < 0.0011200):
            print(data_point.tds_data.filepath)
            plt.plot(data_point.tds_data.data[:, 0], data_point.tds_data.data[:, 1])
            plt.show()
        means.append(mean)

    plt.scatter(range(len(means)), means)
    plt.show()
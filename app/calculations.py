import numpy as np
from names import *
import random


def data_calculation(temperature_series, cloud_cover_series, humidity_series, precipitation_series,
                     U, Alpha, lat, soil_albedo, leaf_index_a, day_number, NV, VZ, roots, hunit):
    return [random.uniform(0.5, 9.5) for _ in range(16)]


def handle_data_for_HUNIT(data_dict, hunit_len: int):
    BazTmin = 6
    BazTmax = 25
    temp_dict = data_dict
    for k, v in temp_dict.items():
        v = v[OM_Temperature]
        grouped_lists = [v[i:i + 8] for i in range(0, len(v), 8)]
        temp_coef = []
        min_list = []
        max_list = []
        TEMP = []
        for group in grouped_lists:
            t_max = max(group)
            t_min = min(group)
            min_list.append(t_min)
            max_list.append(t_max)
            for x in group:
                temp_coef.append((x - t_min) / (t_max - t_min))
        grouped_temp_coef = [temp_coef[i:i + 8] for i in range(0, len(temp_coef), 8)]
        for temp_coef_group, tmin, tmax in zip(grouped_temp_coef, min_list, max_list):
            for element in temp_coef_group:
                x = tmin + (tmax - tmin) * element
                TEMP.append(BazTmax if x > BazTmax else x)
        HUNIT = [np.mean(TEMP[i:i + 8]) for i in range(0, len(TEMP), 8)]

        def HUNIT_for_map(H, tmin, tmax):
            if tmin > BazTmin and tmax < BazTmax:
                return H
            elif tmax < BazTmin:
                return 0
            else:
                return H

        HUNIT = np.cumsum(list(map(HUNIT_for_map, HUNIT, min_list, max_list)))
        temp_dict.update({k: list(HUNIT[-hunit_len:])})
    return temp_dict


if __name__ == '__main__':
    pass

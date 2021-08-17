import os
import numpy as np
import pandas as pd
import datetime

from haversine import haversine_vector, Unit


def cal_bearing_degree(start_x, start_y, end_x, end_y, compass=True):
    """
    Function to return direction as a compass bearing in degrees
    :param start_x: origin longitude
    :param start_y: origin latitutde
    :param end_x: destination longitude
    :param end_y: destination latitude
    :param compass: returns direction in compass degrees (0-360) else bearing (-180 to 180)
    :return:
    """
    diff_x = (end_x - start_x)
    x = np.cos(np.radians(end_y)) * np.sin(np.radians(diff_x))
    y = np.cos(np.radians(start_y)) * np.sin(np.radians(end_y)) - np.sin(np.radians(start_y)) * np.cos(
        np.radians(end_y)) * np.cos(np.radians(diff_x))
    brng = np.degrees(np.arctan2(x, y))
    c_brng = (brng + 360) % 360
    return (
        c_brng
        if compass
        else brng
    )


def build_ts_data(data_sub_folder="data"):
    # Start building the consolidated dataframe for all featurized trajectories
    appended_data = []
    print("Preprocessing raw trajectories for time series...")

    # loop through each .txt, featurize the trajectory, and append to list
    for i, filename in enumerate(os.listdir(data_sub_folder)):
        traject = np.loadtxt(f"{data_sub_folder}/{filename}", delimiter=";", skiprows=1)

        traj_df = pd.DataFrame(traject, columns=["long", "lat", "x_velocity",
                                                 "y_velocity", "seconds_since_start"])

        # create t+1 lat, long cols for vector operations
        traj_df["long_next"] = traj_df["long"].shift(-1)
        traj_df["lat_next"] = traj_df["lat"].shift(-1)
        traj_df["lat_long"] = list(zip(traj_df["lat"], traj_df["long"]))
        traj_df["lat_long_next"] = list(zip(traj_df["lat_next"], traj_df["long_next"]))

        # grab the distance between time steps
        traj_df["hdist_nm"] = haversine_vector(traj_df["lat_long_next"].to_list(),
                                               traj_df["lat_long"].to_list(),
                                               unit=Unit.NAUTICAL_MILES)

        # grab the direction (compass bearing in degrees) between time steps
        traj_df["bearing_compass_degrees"] = cal_bearing_degree(traj_df["long"], traj_df["lat"],
                                                                traj_df["long_next"], traj_df["lat_next"],
                                                                compass=True)
        # shift values down 1
        traj_df["hdist_nm"] = traj_df["hdist_nm"].shift(1)
        traj_df["bearing_compass_degrees"] = traj_df["bearing_compass_degrees"].shift(1)

        # calculating speed and acceleration
        traj_df["seconds_delta"] = traj_df["seconds_since_start"].diff(1)
        traj_df["speed_nm_per_sec"] = traj_df["hdist_nm"] / traj_df["seconds_delta"]
        traj_df["speed_nm_per_hour"] = traj_df["speed_nm_per_sec"] * 3600
        traj_df["acceleration_nm_per_sec"] = (traj_df["speed_nm_per_sec"] - traj_df["speed_nm_per_sec"].shift(1)) / \
                                             traj_df[
                                                 "seconds_delta"]

        # slap the trip id and time step markers on
        traj_df["series_id"] = np.repeat(i, len(traj_df.index))
        traj_df["time_step"] = np.arange(start=0, stop=len(traj_df.index))
        traj_df["date"] = datetime.date(2000, 1, 1)  # insert an arbitrary date for time aware to work
        traj_df["date_time"] = pd.to_datetime(traj_df["date"]) \
                               + pd.to_timedelta(traj_df['seconds_since_start'], unit='s')

        # discard the dummy anchor date
        traj_df.drop('date', axis=1, inplace=True)

        # add the processed trajectory dataframe to list
        appended_data.append(traj_df)

    print("Done preprocessing trajectories for time series")
    master_df = pd.concat(appended_data)

    # re-arrange columns
    master_df = master_df[["series_id", "date_time", "x_velocity", "y_velocity",
                           "hdist_nm", "bearing_compass_degrees", "speed_nm_per_sec",
                           "acceleration_nm_per_sec"]]
    return master_df


if __name__ == "__main__":
    ts_filename = "ais_ts_ad.csv"
    preprocessed_df = build_ts_data()
    preprocessed_df.to_csv(ts_filename, index=False)

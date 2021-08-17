import os
import numpy as np
import pandas as pd


def main():
    print("Preprocessing Ushant AIS data for clustering (non-TS)")

    # a final pre-processed dataframe
    final_df = pd.DataFrame(columns=('trip_id', 'o_long', 'o_lat', 'd_long', 'd_lat',
                                     'avg_x_velocty', 'avg_y_velocity', 'total_duration_seconds'))

    path = "data"  # folder containing trajectories

    # loop through each .txt and grab the relevant data

    for i, filename in enumerate(os.listdir(path)):
        traject = np.loadtxt(f"{path}/{filename}", delimiter=";", skiprows=1)

        traj_df = pd.DataFrame(traject, columns=["long", "lat", "x_velocity",
                                                 "y_velocity", "seconds_since_start"])

        final_df.loc[i] = [i,
                           traj_df["long"][0],
                           traj_df["lat"][0],
                           traj_df["long"].iloc[-1],
                           traj_df["lat"].iloc[-1],
                           traj_df["x_velocity"].mean(),
                           traj_df["y_velocity"].mean(),
                           traj_df["seconds_since_start"].max()]

    # dump to csv and pkl
    final_df.to_csv("ais_clustering.csv", index=False)
    # final_df.to_pickle("ushant_trajectories.pkl")

    print("Done preprocessing Ushant AIS clustering data")


if __name__ == "__main__":
    main()

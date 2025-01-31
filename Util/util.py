import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pandas as pd


def display_with_plotly(battery_data, frequencies, test_number):
    """
    Uses the plotly library to graph Nyquist diagrams
    :param battery_data: Dataframe containing the entire dataset
    :param frequencies:  List of frequencies used for EIS measurements
    :param test_number: Number of the test to be displayed
    :return:
    """
    # Reorganize dataset structure
    ordered_column = []
    battery_data = battery_data[battery_data['cycle'] == test_number]
    column_list = battery_data.columns.tolist()[0:-3]
    n_frequencies = frequencies.shape[0]

    for i in range(0, n_frequencies*2, 2):
        ordered_column.append(column_list[i])

    for i in range(0, n_frequencies*2, 2):
        ordered_column.append(column_list[i+1])

    ordered_column.append('SoC')
    ordered_column.append('Battery')
    ordered_column.append('cycle')
    battery_data = battery_data[ordered_column]

    # Set hover information
    hover_df = pd.DataFrame()
    hover_df["Freq"] = frequencies

    batt_names = battery_data["Battery"].unique()
    hover_template = ('Real: %{customdata[0]} Ω<br>' +
                      'Imaginary: %{customdata[1]} Ω<br>' +
                      'Frequency: %{customdata[2]} Hz<br>' +
                      '<extra><b>%{customdata[3]}</b></extra>')

    # Nyquist plot for each battery
    for b_name in batt_names:
        fig_ly = go.Figure()
        fig_ly.update_layout(
            xaxis_title="Re (Ω)", yaxis_title="-Im (Ω)")
        df_batt1 = battery_data[battery_data['Battery'] == b_name]
        label_idx = [df_batt1.shape[1] - 2, df_batt1.shape[1] - 3]  # [batt_name, SoC]
        for i in range(0, df_batt1.shape[0]):
            label_string = f"{df_batt1.iloc[i, label_idx[0]]}_SoC_{df_batt1.iloc[i, label_idx[1]]}_Test_{test_number}"
            hover_df["b_name"] = label_string
            custom_data = np.stack((
                                  df_batt1.iloc[i, 0:n_frequencies], -df_batt1.iloc[i, n_frequencies:2 * n_frequencies],
                                  hover_df["Freq"],
                                  hover_df["b_name"]), axis=-1)
            fig_ly.add_trace(
                go.Scatter(x=df_batt1.iloc[i, 0:n_frequencies], y=-df_batt1.iloc[i, n_frequencies:2 * n_frequencies],
                           name=label_string,
                           mode="lines+markers", customdata=custom_data))

        fig_ly.update_traces(hovertemplate=hover_template)
        fig_ly.show()


def load_data(data_path, data_freq_path, sep=";"):
    """
    Return the dataset in an unchanged format -> [(ZRe,ZIm) x N_frequencies, SoC, Battery, Test_number]
    :param data_path: Path to the file containing the entire dataset in a single file
    :param data_freq_path: Path to the file containing the frequency values used for EIS measurements
    :param sep: Separator used in the csv file, the default is ";"
    :return:   Dataframe containing the entire dataset ,
               List of frequencies used for EIS measurements,
               List of time periods associated to each frequency
    """
    df = pd.read_csv(data_path, sep=sep)
    frequencies = pd.read_csv(data_freq_path, sep=sep)
    periods = 1 / np.array(frequencies['frequencies'])
    return df, frequencies, periods

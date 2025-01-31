import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import mplcursors


def extract_eis_features(data_file, frequencies_file, battery=None, soc=None, cycle=None, choice_plot_graph=False):
    """
    Extract EIS features from impedance data and plot the Nyquist curve with key features.

    Parameters:
    - data_file: CSV file with impedance data
    - frequencies_file: CSV file with corresponding frequencies
    - battery: Optional battery identifier
    - soc: Optional state of charge
    - cycle: Optional cycle number

    Returns:
    Dictionary of extracted EIS features
    """
    # Load data
    impedance_data = pd.read_csv(data_file)
    frequencies = pd.read_csv(frequencies_file)['frequencies'].values

    # Filter data if specific parameters are provided
    if battery is not None:
        impedance_data = impedance_data[impedance_data['Battery'] == battery]
    if soc is not None:
        impedance_data = impedance_data[impedance_data['SoC'] == soc]
    if cycle is not None:
        impedance_data = impedance_data[impedance_data['cycle'] == cycle]

    # Extract real and imaginary columns
    real_columns = [col for col in impedance_data.columns if '-Re' in col]
    imag_columns = [col for col in impedance_data.columns if '-Im' in col]

    # Flatten real and imaginary data
    Zreal = impedance_data[real_columns].values.flatten()
    Zimg = impedance_data[imag_columns].values.flatten()

    def advanced_x_intercept(xnew, ynew):
        """
        method to find x-axis intercept using curve fitting and root finding.

        Parameters:
        - xnew: Interpolated real impedance values
        - ynew: Interpolated imaginary impedance values

        Returns:
        Estimated x-axis intercept
        """
        # use more points for fitting
        num_points = min(len(xnew)//4, 10)
        x_fit = xnew[:num_points]
        y_fit = ynew[:num_points]

        #higher degree polynomial for better curve representation
        degrees = [3, 4, 5]
        best_intercept = xnew[0]
        best_r2 = -np.inf

        for degree in degrees:
            # Fit polynomial
            coeffs = np.polyfit(x_fit, y_fit, degree)
            poly = np.poly1d(coeffs)

            # Find roots of the polynomial
            roots = poly.roots
            real_roots = roots[np.isreal(roots)].real

            # Find valid roots within data range
            valid_roots = real_roots[
                # (real_roots >= xnew.min()) &
                # (real_roots <= xnew.max())
                real_roots <= xnew.min()
            ]

            # Calculate R-squared for model quality
            y_pred = poly(x_fit)
            r2 = 1 - (np.sum((y_fit - y_pred)**2) /
                      np.sum((y_fit - np.mean(y_fit))**2))

            # Update best intercept if R-squared is better
            if r2 > best_r2 and len(valid_roots) > 0:
                best_intercept = valid_roots[0]
                best_r2 = r2

        return best_intercept

    # Interpolation and peak detection
    def interpolate_data(Zreal, Zimg, num_points=500):
        # Sort points
        sort_idx = np.argsort(Zreal)
        Zreal_sorted = Zreal[sort_idx]
        Zimg_sorted = Zimg[sort_idx]

        # Remove duplicates
        unique_mask = np.concatenate(([True], np.diff(Zreal_sorted) > 1e-10))
        Zreal_sorted = Zreal_sorted[unique_mask]
        Zimg_sorted = Zimg_sorted[unique_mask]

        # Interpolation
        kind = 'cubic' if len(Zreal_sorted) >= 4 else 'linear'
        interp_fn = interp1d(Zreal_sorted, Zimg_sorted,
                             kind=kind,
                             bounds_error=False,
                             fill_value="extrapolate")

        Zreal_interp = np.linspace(
            Zreal_sorted.min(), Zreal_sorted.max(), num_points)
        Zimg_interp = interp_fn(Zreal_interp)

        # Light smoothing
        if len(Zreal_sorted) >= 5:
            window_length = min(11, len(Zimg_interp)//4*2+1)
            if window_length > 2:
                Zimg_interp = savgol_filter(
                    Zimg_interp, window_length, polyorder=2)

        return Zreal_interp, Zimg_interp

    def plot_graph(Zreal, Zimg, x_tailhead, y_tailhead, x_ymax, y_ymax, x_intercept, x_extrapolate, y_extrapolate, x_tail, coef):
        # Plotting Nyquist curve with features
        plt.figure(figsize=(10, 6))
        plt.plot(Zreal, Zimg, 'o', label="Raw Data", alpha=0.5)
        plt.plot(xnew, ynew, '-', label="Interpolated Curve", linewidth=2)
        plt.scatter(x_tailhead, y_tailhead, color='red',
                    label="tailhead", zorder=5)
        plt.scatter(x_ymax, y_ymax, color='yellow',
                    label="ymax(Z'')", zorder=6)
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        plt.axvline(x_intercept, color='blue', linestyle='--',
                    label="Intercept")  # intercept

        # Highlight the extrapolated line
        plt.plot(x_extrapolate, y_extrapolate, color='pink',
                 linestyle='--', label="Extrapolated Line")

        # Key features
        plt.scatter(x_intercept, 0, color='purple',
                    label="X-Axis Intercept", zorder=5)

        plt.plot(x_tail, np.polyval(coef, x_tail), 'g--',
                 label="Tail Fit (Slope)", linewidth=2)
        plt.legend()
        plt.xlabel("Z' (Real Impedance)")
        plt.ylabel("Z'' (Imaginary Impedance)")
        plt.title("Nyquist Plot with Extracted Features")
        plt.grid()

        cursor = mplcursors.cursor(plt.gca(), hover=True)

        @cursor.connect("add")
        def on_add(sel):
            # Find the closest point in the original data
            idx = np.argmin(
                np.sqrt((Zreal - sel.target[0])**2 + (Zimg - sel.target[1])**2))

            if idx < len(frequencies):
                sel.annotation.set_text(
                    f"Freq: {frequencies[idx]:.5f} Hz\n"
                    f"Z': {Zreal[idx]:.5f}\n"
                    f"Z'': {Zimg[idx]:.5f}"
                )
            else:
                sel.annotation.set_text("Point out of bounds")

        # Invert y-axis
        plt.gca().invert_yaxis()

        plt.show()

    # Interpolate data
    xnew, ynew = interpolate_data(Zreal, Zimg)

    # Feature extraction
    # finding min curvature using curvature analysis
    dZ = np.gradient(ynew, xnew)
    d2Z = np.gradient(dZ)
    tailhead_idx = np.argmin(d2Z)

    # Key features
    x_tailhead = xnew[tailhead_idx]
    y_tailhead = ynew[tailhead_idx]

    # Tail and slope (simplified)
    tail_start_idx = tailhead_idx + len(xnew)//4
    x_tail = xnew[tail_start_idx:]
    y_tail = ynew[tail_start_idx:]

    try:
        # Linear fit for tail
        coef = np.polyfit(x_tail, y_tail, 1)
        slope = coef[0]
        tailhead = x_tail[0]
    except:
        slope = 0
        tailhead = x_tailhead

    # Extrapolate to find the x-axis intercept using a polynomial fit
    # Use a higher-degree polynomial (degree 3 or higher based on curve shape)
    num_points = 3  # Use the first 5 points (or adjust based on data)
    x_left = xnew[:num_points]
    y_left = ynew[:num_points]

    # Perform a polynomial fit (degree 3 or higher, depending on the curve's nature)
    degree = 3  # Adjust the degree as needed
    coeffs = np.polyfit(x_left, y_left, degree)  # Fit a polynomial of degree 3
    # Create a polynomial function from the coefficients
    poly = np.poly1d(coeffs)

    x_intercept = advanced_x_intercept(xnew, ynew)

    print(f"Extrapolated x-axis intercept: {x_intercept}")

    # Generate the extended curve for plotting
    x_extrapolate = np.linspace(min(x_left), x_intercept, 100)
    y_extrapolate = poly(x_extrapolate)

    # Diameter estimation (rough approximation)
    diameter = 2 * (x_tailhead - x_intercept)

    # Find ymax(Z'') as the peak of the semicircle
    # Find the indices where xnew lies between intercept and x_tailhead
    mask = (xnew >= x_intercept) & (xnew <= x_tailhead)

    # Extract the corresponding xnew and ynew values within the range
    x_in_range = xnew[mask]
    y_in_range = ynew[mask]

    x_ymax = 0
    y_ymax = 0

    # Find the index of the minimum y value in the range (maximum Z'')
    if len(y_in_range) > 0:
        # Index of minimum y in the range
        ymax_idx_in_range = np.argmin(y_in_range)
        x_ymax = x_in_range[ymax_idx_in_range]    # x-coordinate of ymax(Z'')
        y_ymax = y_in_range[ymax_idx_in_range]    # y-coordinate of ymax(Z'')
        print(f"Maximum Z'' (ymin) in range: y = {y_ymax}, x = {x_ymax}")
    else:
        print("No points found in the specified range")

    # Shape parameter (approximation)
    shape = (x_tailhead - x_intercept) / max(x_tailhead, 1e-10)

    # Shoulder detection (dummy)
    shoulder = 0  # Placeholder

    if choice_plot_graph:
        plot_graph(Zreal, Zimg, x_tailhead, y_tailhead, x_ymax, y_ymax,
                   x_intercept, x_extrapolate, y_extrapolate, x_tail, coef)

    return {
        'x_tailhead': x_tailhead,
        'y_tailhead': y_tailhead,
        'intercept': x_intercept,
        'x_ymax': x_ymax,
        'y_ymax': y_ymax,
        # 'tailhead': tailhead,
        'slope': slope,
        'diameter': diameter,
        'shape': shape,
        'shoulder': shoulder,
        'battery': battery,
        'soc': soc,
        'cycle': cycle
    }


# Example usage
if __name__ == "__main__":
    features = extract_eis_features(
        './Data/Data_IFR14500_Batt_1-11_cycle_1_2.csv',
        './Data/Frequencies.csv',
        battery='B01',
        soc=100,
        cycle=1,
        choice_plot_graph=True
    )

    print(f'features: {features}')

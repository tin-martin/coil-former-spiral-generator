# selector.py
import matplotlib.pyplot as plt
import numpy as np
import csv

class PointSelector:
    """
    A class to interactively select points on a matplotlib graph and save them to CSV.

    Attributes:
        fig (matplotlib.figure.Figure): The figure object for the plot.
        ax (matplotlib.axes.Axes): The axes object for the plot.
        selected_points (list): A list to store the (x, y) coordinates of selected points.
        point_markers (list): A list to store the plot markers for selected points.
    """

    def __init__(self):
        """
        Initializes the PointSelector with a plot and event handlers.
        """
        config = {}
        filename = 'polygon_spiral_config.csv'
        try:
            with open(filename, 'r', newline='') as f:
                reader = csv.reader(f)
                next(reader) # Skip header row
                for row in reader:
                    if len(row) == 2:
                        try:
                            config[row[0]] = float(row[1])
                        except ValueError:
                            print(f"Warning: Could not convert value '{row[1]}' for parameter '{row[0]}' to float. Skipping.")
                    else:
                        print(f"Warning: Skipping malformed row in config CSV: {row}")
        except FileNotFoundError:
            print(f"Error: Configuration file '{filename}' not found. Using default hardcoded values.")
            config = {
                'initial_plot_xlim_min': -50.0,
                'initial_plot_xlim_max': 50.0,
                'initial_plot_ylim_min': -50.0,
                'initial_plot_ylim_max': 50.0,
            }
        except Exception as e:
            print(f"An unexpected error occurred while loading config from CSV: {e}. Using default hardcoded values.")
            config = {
                'initial_plot_xlim_min': -50.0,
                'initial_plot_xlim_max': 50.0,
                'initial_plot_ylim_min': -50.0,
                'initial_plot_ylim_max': 50.0,
            }

        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Click to select points. Press 'q' to quit and save.")
        self.ax.set_xlabel("X-axis")
        self.ax.set_ylabel("Y-axis")
        self.ax.grid(True)
        self.ax.set_xlim(config.get('initial_plot_xlim_min'), config.get('initial_plot_xlim_max'))
        self.ax.set_ylim(config.get('initial_plot_ylim_min'), config.get('initial_plot_ylim_max'))


        self.selected_points = []
        self.point_markers = []

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        print("Instructions for PointSelector:")
        print("1. Click on the graph to select points.")
        print("2. Each click will add a new point.")

    def on_click(self, event):
        """
        Event handler for mouse clicks on the plot.
        Adds the clicked point to the list and updates the plot.
        """
        if event.inaxes != self.ax:
            return

        x, y = event.xdata, event.ydata
        if x is not None and y is not None:
            self.selected_points.append(np.array([x, y]))
            marker, = self.ax.plot(x, y, 'rx', markersize=10)
            self.point_markers.append(marker)
            self.fig.canvas.draw()
            print(f"Selected point: ({x:.2f}, {y:.2f})")

    def on_key_press(self, event):
        """
        Event handler for key presses.
        Closes the plot and saves the selected points to a CSV file when 'q' is pressed.
        """
        if event.key == 'q':
            plt.close(self.fig)

    def show(self):
        """
        Displays the matplotlib plot and starts the event loop.
        """
        plt.show()

def load_points_from_csv(filename="irregular_polygon_points.csv"):
    """
    Loads points from a CSV file and returns them as a list of numpy arrays.

    Args:
        filename (str): The path to the CSV file.

    Returns:
        list: A list of numpy arrays, where each array is a [x, y] point.
              Returns an empty list if the file is not found or an error occurs.
    """
    loaded_points = []
    try:
        with open(filename, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader) # Skip header row (assuming 'x', 'y')
            for row in reader:
                if len(row) == 2:
                    try:
                        x = float(row[0])
                        y = float(row[1])
                        loaded_points.append(np.array([x, y]))
                    except ValueError:
                        print(f"Skipping invalid data in CSV row: {row}")
                else:
                    print(f"Skipping malformed CSV row: {row}")
        return loaded_points
    except FileNotFoundError:
        print(f"Error: CSV file '{filename}' not found.")
        print("Please run the 'selector.py' script to interactively define points and save them to this CSV.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while loading points from CSV: {e}")
        return []

if __name__ == "__main__":
    selector = PointSelector()
    selector.show()
# circular.py (Standalone version, does not inherit from polygon.py)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
import math
import os # For saving files
import ezdxf # DXF export library
import csv # For reading CSV configuration
from matplotlib.text import Text # Import Text for displaying dimensions

class CircularSpiralApp:
    """
    Application for generating and visualizing circular spirals.
    This is a standalone application, not inheriting from PolygonSpiralApp.
    """
    def __init__(self):
        """
        Initializes the CircularSpiralApp. Sets up UI elements, loads config,
        and manages the plot.
        """
        # Load configuration from CSV
        self._load_config_from_csv("polygon_spiral_config.csv")

        # Define circular-specific parameters (from config or defaults)
        # Using values loaded from config where applicable, otherwise defaults
        self.initial_inner_diameter = getattr(self, 'initial_inner_diameter', 1.0)
        self.min_inner_diameter = getattr(self, 'min_inner_diameter', 0.1)
        self.max_inner_diameter = getattr(self, 'max_inner_diameter', 100.0)

        self.initial_fixed_distance_d = getattr(self, 'initial_fixed_distance_d', 0.5)
        self.min_initial_d = getattr(self, 'min_initial_d', 0.1)
        self.max_fixed_d = getattr(self, 'max_fixed_d', 20.0) # Using max_fixed_d for consistency

        self.initial_num_turns = getattr(self, 'initial_num_turns', 5.0)
        self.min_num_turns = getattr(self, 'min_num_turns', 0.0)
        self.max_num_turns = getattr(self, 'max_num_turns', 50.0)

        self.initial_plot_xlim_min = getattr(self, 'initial_plot_xlim_min', -50.0)
        self.initial_plot_ylim_min = getattr(self, 'initial_plot_ylim_min', -50.0)
        self.initial_plot_xlim_max = getattr(self, 'initial_plot_xlim_max', 50.0)
        self.initial_plot_ylim_max = getattr(self, 'initial_plot_ylim_max', 50.0)

        self.initial_max_x_length = getattr(self, 'initial_max_x_length', 100.0)
        self.initial_max_y_length = getattr(self, 'initial_max_y_length', 100.0)
        self.dxf_insunits = getattr(self, 'dxf_insunits', 4)

        # Hardcode spiral direction for circular
        self.inner_to_outer = "0" # "0" for inner to outer, "1" for outer to inner

        # Set up the Matplotlib figure and axes
        self.fig, self.ax = plt.subplots(figsize=(10, 7.5))
        plt.subplots_adjust(left=0.1, bottom=0.45) # Adjusted bottom margin for controls

        # Initialize plot elements
        self.current_spiral_points = []
        self.spiral_line, = self.ax.plot([], [], linestyle='-', color='green', linewidth=2)

        # Initialize text for bounding box dimensions
        self.bbox_text = self.ax.text(0.98, 0.98, '',
                                      horizontalalignment='right',
                                      verticalalignment='top',
                                      transform=self.ax.transAxes,
                                      fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # Set plot labels, grid, and initial limits
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlim(self.initial_plot_xlim_min, self.initial_plot_xlim_max)
        self.ax.set_ylim(self.initial_plot_ylim_min, self.initial_plot_ylim_max)

        # --- UI Element Definitions for Circular Spiral ---

        # Common vertical position for the first row of textboxes
        row1_bottom = 0.34
        # Common vertical position for the second row of textboxes
        row2_bottom = 0.30

        # Define text box axes for Max Y-Length and Max X-Length
        # Max Y-Len: Left side of the first row
        self.ax_max_y_length = plt.axes([0.1, row1_bottom, 0.15, 0.03], facecolor='lightgoldenrodyellow')
        self.max_y_length_textbox = TextBox(self.ax_max_y_length, "Max Y-Len", initial=str(self.initial_max_y_length))
        self.max_y_length_textbox.on_submit(self._submit_max_y_length)
        self._current_max_y_length = self.initial_max_y_length

        # Max X-Len: Shifted to the right on the first row
        self.ax_max_x_length = plt.axes([0.1 + 0.225, row1_bottom, 0.15, 0.03], facecolor='lightgoldenrodyellow')
        self.max_x_length_textbox = TextBox(self.ax_max_x_length, "Max X-Len", initial=str(self.initial_max_x_length))
        self.max_x_length_textbox.on_submit(self._submit_max_x_length)
        self._current_max_x_length = self.initial_max_x_length

        # Define text box axis for Trace Gap (D-Gap): Left side of the second row
        self.ax_initial_d = plt.axes([0.1, row2_bottom, 0.15, 0.03], facecolor='lightgoldenrodyellow')
        self.d_gap_textbox = TextBox(self.ax_initial_d, "Trace Gap", initial=str(self.initial_fixed_distance_d))
        self.d_gap_textbox.on_submit(self._submit_d_gap)
        self._current_initial_d = self.initial_fixed_distance_d

        # Define text box axis for Inner Diameter: Right side of the second row
        self.ax_inner_diameter = plt.axes([0.1 + 0.225, row2_bottom, 0.15, 0.03], facecolor='lightgoldenrodyellow')
        self.inner_diameter_textbox = TextBox(self.ax_inner_diameter, "Inner Dia.", initial=str(self.initial_inner_diameter))
        self.inner_diameter_textbox.on_submit(self._submit_inner_diameter)
        self._current_inner_diameter = self.initial_inner_diameter


        # Define slider axis for Turns (position remains the same, adjusted earlier to bottom=0.45)
        self.ax_num_turns = plt.axes([0.1, 0.22, 0.8, 0.03], facecolor='lightgoldenrodyellow')
        self.s_num_turns = Slider(self.ax_num_turns, 'Turns', self.min_num_turns, self.max_num_turns, valinit=self.initial_num_turns, valstep=0.1)

        # Define button axes
        self.ax_reset_button = plt.axes([0.35, 0.05, 0.1, 0.04])
        self.ax_export_button = plt.axes([0.55, 0.05, 0.1, 0.04])

        # Create buttons
        self.reset_button = Button(self.ax_reset_button, 'Reset', color='lightblue', hovercolor='0.975')
        self.export_button = Button(self.ax_export_button, 'Export DXF', color='lightgreen', hovercolor='0.975')

        # Attach update method to slider changes and textbox submissions
        self.s_num_turns.on_changed(self.update)

        # Attach methods to button clicks
        self.reset_button.on_clicked(self.reset)
        self.export_button.on_clicked(self.export_dxf)

        # Set the title for the circular spiral plot
        if self.inner_to_outer == "0":
            self.ax.set_title("Inner-to-Outer Circular Spiral")
        else:
            self.ax.set_title("Outer-to-Inner Circular Spiral")

        # Initial update to draw the spiral with default values
        self.update(None)

    def _load_config_from_csv(self, filename):
        """
        Loads configuration parameters from a CSV file into instance attributes.
        """
        config = {}
        try:
            with open(filename, 'r', newline='') as f:
                reader = csv.reader(f)
                header_skipped = False
                for i, row in enumerate(reader):
                    if not header_skipped: # Skip header row
                        header_skipped = True
                        continue

                    if len(row) == 2:
                        key = row[0].strip()
                        value_str = row[1].strip()
                        try:
                            # Try to convert to float, otherwise keep as string
                            config[key] = float(value_str)
                        except ValueError:
                            config[key] = value_str
                    elif row:
                        print(f"Warning: Skipping malformed row in config CSV (line {i+1}): {row}. Expected 2 columns.")
        except FileNotFoundError:
            print(f"Error: Configuration file '{filename}' not found. Using default values for circular app.")
        except Exception as e:
            print(f"Error reading configuration CSV: {e}. Using default values for circular app.")

        # Assign loaded (or default) values to instance attributes (relevant for circular spiral)
        self.initial_inner_diameter = config.get('initial_inner_diameter', 1.0)
        self.min_inner_diameter = config.get('min_inner_diameter', 0.1)
        self.max_inner_diameter = config.get('max_inner_diameter', 100.0)

        self.initial_fixed_distance_d = config.get('initial_fixed_distance_d', 0.5)
        self.min_initial_d = config.get('min_initial_d', 0.1)
        self.max_fixed_d = config.get('max_fixed_d', 20.0)

        self.initial_num_turns = config.get('initial_num_turns', 5.0)
        self.min_num_turns = config.get('min_num_turns', 0.0)
        self.max_num_turns = config.get('max_num_turns', 50.0)

        self.initial_plot_xlim_min = config.get('initial_plot_xlim_min', -50.0)
        self.initial_plot_ylim_min = config.get('initial_plot_ylim_min', -50.0)
        self.initial_plot_xlim_max = config.get('initial_plot_xlim_max', 50.0)
        self.initial_plot_ylim_max = config.get('initial_plot_ylim_max', 50.0)

        self.initial_max_x_length = config.get('initial_max_x_length', 100.0)
        self.initial_max_y_length = config.get('initial_max_y_length', 100.0)

        self.dxf_insunits = int(config.get('dxf_insunits', 4))


    def _submit_inner_diameter(self, text):
        """
        Callback for Inner Diameter text box submission.
        Parses the input, updates the internal value, and triggers plot update.
        """
        print(f"DEBUG: Attempting to submit inner diameter: '{text}'")
        try:
            val = float(text)
            print(f"DEBUG: Parsed value: {val}, Current stored: {self._current_inner_diameter}")

            original_val = val # Store original for comparison

            # Apply min/max constraints
            if val < self.min_inner_diameter:
                val = self.min_inner_diameter
                print(f"DEBUG: Value {original_val} below min {self.min_inner_diameter}, clamped to {val:.2f}")
            elif val > self.max_inner_diameter:
                val = self.max_inner_diameter
                print(f"DEBUG: Value {original_val} above max {self.max_inner_diameter}, clamped to {val:.2f}")

            # Update textbox value if it was clamped, otherwise set to parsed value
            self.inner_diameter_textbox.set_val(f"{val:.2f}")

            if self._current_inner_diameter != val: # Only update if value actually changed
                self._current_inner_diameter = val
                print(f"DEBUG: Inner diameter internally updated to: {self._current_inner_diameter:.2f}")
                self.update(None)
            else:
                print(f"DEBUG: Inner diameter did not change (value already {val:.2f} or clamped to same value).")

        except ValueError:
            print(f"ERROR: Invalid input for Inner Diameter. Please enter a number. Input was: '{text}'")
            self.inner_diameter_textbox.set_val(f"{self._current_inner_diameter:.2f}")
        self.fig.canvas.draw_idle()

    def _submit_d_gap(self, text):
        """
        Callback for Trace Gap text box submission.
        Parses the input, updates the internal value, and triggers plot update.
        """
        try:
            val = float(text)
            if val < self.min_initial_d:
                val = self.min_initial_d
                self.d_gap_textbox.set_val(f"{val:.2f}")
            elif val > self.max_fixed_d:
                val = self.max_fixed_d
                self.d_gap_textbox.set_val(f"{val:.2f}")
            self._current_initial_d = val
            self.update(None)
        except ValueError:
            print("Invalid input for Trace Gap. Please enter a number.")
            self.d_gap_textbox.set_val(f"{self._current_initial_d:.2f}")
        self.fig.canvas.draw_idle()

    def _submit_max_x_length(self, text):
        """
        Callback for Max X-Length text box submission.
        Parses the input, updates the internal value, and triggers plot update.
        """
        try:
            val = float(text)
            if val < 0.0:
                val = 0.0
                self.max_x_length_textbox.set_val(f"{val:.2f}")
            self._current_max_x_length = val
            self.update(None)
        except ValueError:
            print("Invalid input for Max X-Length. Please enter a number.")
            self.max_x_length_textbox.set_val(f"{self._current_max_x_length:.2f}")
        self.fig.canvas.draw_idle()

    def _submit_max_y_length(self, text):
        """
        Callback for Max Y-Length text box submission.
        Parses the input, updates the internal value, and triggers plot update.
        """
        try:
            val = float(text)
            if val < 0.0:
                val = 0.0
                self.max_y_length_textbox.set_val(f"{val:.2f}")
            self._current_max_y_length = val
            self.update(None)
        except ValueError:
            print("Invalid input for Max Y-Length. Please enter a number.")
            self.max_y_length_textbox.set_val(f"{self._current_max_y_length:.2f}")
        self.fig.canvas.draw_idle()

    def generate_circular_spiral(self, inner_diameter: float, trace_gap: float, num_turns: float, num_points_per_turn: int = 360):
        """
        Generates points for an Archimedean circular spiral.

        Args:
            inner_diameter (float): The diameter of the innermost part of the spiral.
            trace_gap (float): The radial distance between successive turns (pitch).
            num_turns (float): The total number of turns for the spiral.
            num_points_per_turn (int): Number of points to generate per full turn for smoothness.

        Returns:
            list: A list of numpy arrays, where each array is an [x, y] point
                  representing the spiral path.
        """
        points = []
        start_radius = inner_diameter / 2.0
        b = trace_gap / (2 * math.pi)

        total_theta = num_turns * 2 * math.pi
        num_points = int(num_turns * num_points_per_turn)

        if num_points == 0:
            return []

        if self.inner_to_outer == "0": # Inner to Outer
            theta_values = np.linspace(0, total_theta, num_points)
            for theta in theta_values:
                r = start_radius + b * theta
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                points.append(np.array([x, y]))
        else: # Outer to Inner
            outer_radius = start_radius + b * total_theta
            theta_values = np.linspace(0, total_theta, num_points)
            for theta in reversed(theta_values):
                r = outer_radius - b * (total_theta - theta)
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                points.append(np.array([x, y]))
        return points

    def update(self, val):
        """
        Updates the circular spiral plot based on current slider values and textbox inputs.
        Recalculates points, updates plot limits, and redraws.
        """
        current_inner_diameter = self._current_inner_diameter
        current_trace_gap = self._current_initial_d
        current_num_turns = self.s_num_turns.val

        try:
            generated_points = self.generate_circular_spiral(
                current_inner_diameter,
                current_trace_gap,
                current_num_turns
            )
            self.current_spiral_points = generated_points

        except ValueError as e:
            print(f"Error generating circular spiral: {e}")
            self.current_spiral_points = []

        if self.current_spiral_points:
            x_coords = [p[0] for p in self.current_spiral_points]
            y_coords = [p[1] for p in self.current_spiral_points]
            self.spiral_line.set_data(x_coords, y_coords)

            if x_coords and y_coords:
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                x_length = x_max - x_min
                y_length = y_max - y_min

                self.bbox_text.set_text(f"X-Length: {x_length:.2f}\nY-Length: {y_length:.2f}")

                if x_length > self._current_max_x_length or y_length > self._current_max_y_length:
                    self.spiral_line.set_color('red')
                else:
                    self.spiral_line.set_color('green')

                padding = max(x_length, y_length) * 0.1
                self.ax.set_xlim(x_min - padding, x_max + padding)
                self.ax.set_ylim(y_min - padding, y_max + padding)
            else:
                self.bbox_text.set_text("")
                self.spiral_line.set_data([], [])
                self.spiral_line.set_color('green')
                self.ax.set_xlim(self.initial_plot_xlim_min, self.initial_plot_xlim_max)
                self.ax.set_ylim(self.initial_plot_ylim_min, self.initial_plot_ylim_max)
        else:
            self.spiral_line.set_data([], [])
            self.bbox_text.set_text("")
            self.spiral_line.set_color('green')
            self.ax.set_xlim(self.initial_plot_xlim_min, self.initial_plot_xlim_max)
            self.ax.set_ylim(self.initial_plot_ylim_min, self.initial_plot_ylim_max)

        self.fig.canvas.draw_idle()

    def reset(self, event):
        """
        Resets all slider and text box values to defaults.
        """
        self.inner_diameter_textbox.set_val(f"{self.initial_inner_diameter:.2f}")
        self._current_inner_diameter = self.initial_inner_diameter

        self.d_gap_textbox.set_val(f"{self.initial_fixed_distance_d:.2f}")
        self._current_initial_d = self.initial_fixed_distance_d

        self.max_x_length_textbox.set_val(f"{self.initial_max_x_length:.2f}")
        self._current_max_x_length = self.initial_max_x_length

        self.max_y_length_textbox.set_val(f"{self.initial_max_y_length:.2f}")
        self._current_max_y_length = self.initial_max_y_length

        self.s_num_turns.set_val(self.initial_num_turns)

        self.update(None)

    def export_dxf(self, event):
        """
        Exports the generated spiral points to a DXF file.
        """
        if not self.current_spiral_points:
            print("No spiral points to export.")
            return

        doc = ezdxf.new('R2010')
        doc.header['$INSUNITS'] = self.dxf_insunits

        msp = doc.modelspace()

        if self.current_spiral_points:
            dxf_points = [(p[0], p[1], 0) for p in self.current_spiral_points]
            msp.add_lwpolyline(dxf_points, dxfattribs={'color': 2, 'layer': 'SPIRAL_CENTERLINE'}) # Color 2 is yellow in AutoCAD
            print("Spiral centerline exported to DXF.")
        else:
            print("No spiral centerline points to export.")

        filename = "circular_spiral_centerline.dxf" # Changed filename to be specific
        try:
            doc.saveas(filename)
            print(f"Circular spiral centerline data exported to {os.path.abspath(filename)}")
        except Exception as e:
            print(f"Error saving DXF file: {e}")

# --- Main Execution Block (for direct testing) ---
if __name__ == "__main__":
    app = CircularSpiralApp()
    plt.show()
# polygon.py (Modified from polygon_spiral_app_base.py)
# Defines the base class for polygon spiral applications.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
import os # For saving files
import ezdxf # DXF export library
import math # For mathematical operations like sqrt
import csv # For reading CSV configuration
from matplotlib.text import Text # Import Text for displaying dimensions

class PolygonSpiralApp:
    """
    Base application for generating and visualizing polygon spirals.
    Handles common UI elements, plot setup, and spiral generation logic.
    """
    def __init__(self):
        """
        Initializes the PolygonSpiralApp. Gets user input for spiral type,
        loads common parameters from CSV, and sets up the Matplotlib figure and UI.
        """
        # Load configuration from CSV
        self._load_config_from_csv("polygon_spiral_config.csv")

        # Get user input for spiral direction (inner-to-outer or outer-to-inner)
        """self.inner_to_outer = input("If inner to outer, type 0, otherwise type 1: ").strip().upper()
        while self.inner_to_outer not in ["0", "1"]:
            print("\nInvalid input! Please type 0 or 1.")
            self.inner_to_outer = input("If inner to outer, type 0. If not, type 1: ").strip().upper()
        """
        self.inner_to_outer = "0"
        # Get user input for gap type (square root non-linear or linear)

        # Set up the Matplotlib figure and axes
        self.fig, self.ax = plt.subplots(figsize=(10, 7.5))
        # Adjusted bottom margin to make more room for all sliders and buttons
        plt.subplots_adjust(left=0.1, bottom=0.45)


        # Initialize plot elements (line, scatter, labels)
        self.current_spiral_points = [] # Stores generated points for DXF export
        # Initialize with default valid color (blue)
        self.spiral_line, = self.ax.plot([], [], linestyle='-', color='green', linewidth=2)

        # Initialize text for bounding box dimensions
        # Position the text in the top right corner of the axes
        self.bbox_text = self.ax.text(0.98, 0.98, '',
                                      horizontalalignment='right',
                                      verticalalignment='top',
                                      transform=self.ax.transAxes,
                                      fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))


        # Set plot labels, grid, and initial limits from config
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlim(self.initial_plot_xlim_min, self.initial_plot_xlim_max)
        self.ax.set_ylim(self.initial_plot_ylim_min, self.initial_plot_ylim_max)



        # Define text box axis for D-Gap (adjusted Y position)
        self.ax_initial_d = plt.axes([0.1, 0.25, 0.1, 0.03], facecolor='lightgoldenrodyellow') #martin
        # Create text box for D-Gap
        self.d_gap_textbox = TextBox(self.ax_initial_d, "Trace Gap", initial=str(self.initial_fixed_distance_d))
        self.d_gap_textbox.on_submit(self._submit_d_gap)
        self._current_initial_d = self.initial_fixed_distance_d # Store current value

        # Define text box axes for Max X-Length and Max Y-Length
        self.ax_max_x_length = plt.axes([0.1+0.2, 0.3, 0.1, 0.03], facecolor='lightgoldenrodyellow')
        self.ax_max_y_length = plt.axes([0.1, 0.3, 0.1, 0.03], facecolor='lightgoldenrodyellow')

        # Create text boxes for Max X-Length and Max Y-Length
        self.max_x_length_textbox = TextBox(self.ax_max_x_length, "Max X-Len", initial=str(self.initial_max_x_length))
        self.max_y_length_textbox = TextBox(self.ax_max_y_length, "Max Y-Len", initial=str(self.initial_max_y_length))
        self.max_x_length_textbox.on_submit(self._submit_max_x_length)
        self.max_y_length_textbox.on_submit(self._submit_max_y_length)
        self._current_max_x_length = self.initial_max_x_length
        self._current_max_y_length = self.initial_max_y_length

        # Define slider axes for common spiral parameters (adjusted Y positions)
        self.ax_num_turns = plt.axes([0.1, 0.18, 0.8, 0.03], facecolor='lightgoldenrodyellow')

        # Create sliders for common spiral parameters
        self.s_num_turns = Slider(self.ax_num_turns, 'Turns', self.min_num_turns, self.max_num_turns, valinit=self.initial_num_turns, valstep=0.1)

        # Define button axes (adjusted Y positions)
        self.ax_reset_button = plt.axes([0.35, 0.05, 0.1, 0.04])
        self.ax_export_button = plt.axes([0.55, 0.05, 0.1, 0.04])

        # Create buttons
        self.reset_button = Button(self.ax_reset_button, 'Reset', color='lightblue', hovercolor='0.975')
        self.export_button = Button(self.ax_export_button, 'Export DXF', color='lightgreen', hovercolor='0.975')

        # Attach update method to common slider changes
        self.s_num_turns.on_changed(self.update)

        # Attach methods to button clicks
        self.reset_button.on_clicked(self.reset)
        self.export_button.on_clicked(self.export_dxf)

    def _load_config_from_csv(self, filename):
        """
        Loads configuration parameters from a CSV file into instance attributes.
        """
        config = {}
       
        with open(filename, 'r', newline='') as f:
            reader = csv.reader(f)
            # Skip header row (assuming format: parameter,value)
            # Ensure the CSV is properly formatted with at least two columns
            header_skipped = False
            for i, row in enumerate(reader):
                if not header_skipped:
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
       
        # Assign loaded (or default) values to instance attributes
        
        self.initial_inner_diameter = config.get('initial_inner_diameter', 5.0)

        self.initial_fixed_distance_d = config.get('initial_fixed_distance_d', 10.0)
        self.initial_num_turns = config.get('initial_num_turns', 5.0)

        # Ensure correct defaults if not found in config, for regular polygon specific
        self.initial_num_sides = config.get('initial_num_sides', 6.0)
        self.max_num_sides = config.get('max_num_sides', 6.0)
        self.max_initial_len = config.get('max_initial_len', 200.0)
        self.initial_segment_len = config.get('initial_segment_len', 20.0)

        self.max_fixed_d = config.get('max_fixed_d', 20.0)

        self.max_num_turns = config.get('max_num_turns', 50.0)

        self.initial_plot_xlim_min = config.get('initial_plot_xlim_min', -50.0)
        self.initial_plot_ylim_min = config.get('initial_plot_ylim_min', -50.0)
        self.initial_plot_xlim_max = config.get('initial_plot_xlim_max', 50.0)
        self.initial_plot_ylim_max = config.get('initial_plot_ylim_max', 50.0)
        self.minimum_d_threshold = config.get('minimum_d_threshold', 0.1)

        # New slider min value assignments
        self.min_initial_d = config.get('min_initial_d', 0.1)
        self.min_num_turns = config.get('min_num_turns', 0.0)
        self.min_num_sides = config.get('min_num_sides', 3.0)
        self.min_initial_len = config.get('min_initial_len', 0.1)

        # New DXF export assignments, ensuring integer types for units
        self.dxf_insunits = int(config.get('dxf_insunits', 4.0))

        # New max X/Y length assignments
        self.initial_max_x_length = config.get('initial_max_x_length', 100.0)
        self.initial_max_y_length = config.get('initial_max_y_length', 100.0)


    # --- Abstract/Common Spiral Generation Logic ---
    def generate_polygon_spiral(self, num_points: int, initial_d: float, outermost_polygon_vertices: list[tuple[float, float]], num_sides: int):
        """
        Generates points for a polygon spiral.
        Calculates points based on initial polygon vertices, distance 'd',
        decrease rate, and spiral direction.
        """

        num_sides = len(outermost_polygon_vertices)
        if num_sides < 3:
            raise ValueError("An irregular polygon must have at least 3 vertices.")
        if num_points < num_sides:
            return [np.array(v) for v in outermost_polygon_vertices]

        points = []
        for vertex in outermost_polygon_vertices:
            points.append(np.array(vertex))

        for n in range(num_sides, num_points):
            num_turns_completed = (n - num_sides) / num_sides

            # Apply linear or non-linear gap decrease


            current_d = initial_d

            if current_d <= 1e-9: # Stop if distance becomes too small
                break

            Q_n_minus_1_initial_idx = (n - 1) % num_sides
            Q_n_initial_idx = n % num_sides

            segment_vec_initial_polygon = np.array(outermost_polygon_vertices[Q_n_initial_idx]) - np.array(outermost_polygon_vertices[Q_n_minus_1_initial_idx])
            dir_current = segment_vec_initial_polygon / np.linalg.norm(segment_vec_initial_polygon)

            dir_reference_initial_polygon_idx = n % num_sides
            next_point_initial_polygon_idx = (n + 1) % num_sides

            dir_reference_raw = np.array(outermost_polygon_vertices[next_point_initial_polygon_idx]) - np.array(outermost_polygon_vertices[dir_reference_initial_polygon_idx])
            dir_reference = dir_reference_raw / np.linalg.norm(dir_reference_raw)

            dir_reference_perp = np.array([-dir_reference[1], dir_reference[0]])

            Q_n_minus_1 = points[n-1]
            Q_n_minus_sides = points[n - num_sides]

            delta_Q_prev = Q_n_minus_1 - Q_n_minus_sides

            denominator = np.dot(dir_current, dir_reference_perp)

            if np.isclose(denominator, 0):
                print(f"Warning: Denominator close to zero at n={n}. Skipping point.")
                break

            s1 = (current_d - np.dot(delta_Q_prev, dir_reference_perp)) / denominator
            s2 = (-current_d - np.dot(delta_Q_prev, dir_reference_perp)) / denominator

            positive_s_options = []
            if s1 > 1e-9:
                positive_s_options.append(s1)
            if s2 > 1e-9:
                positive_s_options.append(s2)

            if not positive_s_options:
                print(f"Warning: No positive 's' options at n={n}. Spiral converging.")
                break

            # Choose 's' based on spiral direction
            if self.inner_to_outer == "0":
                s = max(positive_s_options)
            else: # Outer to Inner
                s = min(positive_s_options)

            if np.isclose(s, 0):
                print(f"Warning: 's' is close to zero at n={n}. Spiral converging.")
                break

            Q_n = Q_n_minus_1 + s * dir_current
            points.append(Q_n)

        return points

    def _submit_d_gap(self, text):
        """
        Callback for D-Gap text box submission.
        Parses the input, updates the internal value, and triggers plot update.
        """
        try:
            val = float(text)
            # Apply min/max constraints
            if val < self.min_initial_d:
                val = self.min_initial_d
                self.d_gap_textbox.set_val(f"{val:.2f}") # Update textbox if value was clamped
            elif val > self.max_fixed_d:
                val = self.max_fixed_d
                self.d_gap_textbox.set_val(f"{val:.2f}") # Update textbox if value was clamped
            self._current_initial_d = val
            self.update(None) # Call update after value is set
        except ValueError:
            print("Invalid input for D-Gap. Please enter a number.")
            # Revert to last valid value
            self.d_gap_textbox.set_val(f"{self._current_initial_d:.2f}")
        self.fig.canvas.draw_idle()

    def _submit_max_x_length(self, text):
        """
        Callback for Max X-Length text box submission.
        Parses the input, updates the internal value, and triggers plot update.
        """
        try:
            val = float(text)
            if val < 0.0: # Ensure non-negative
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
            if val < 0.0: # Ensure non-negative
                val = 0.0
                self.max_y_length_textbox.set_val(f"{val:.2f}")
            self._current_max_y_length = val
            self.update(None)
        except ValueError:
            print("Invalid input for Max Y-Length. Please enter a number.")
            self.max_y_length_textbox.set_val(f"{self._current_max_y_length:.2f}")
        self.fig.canvas.draw_idle()


    # --- Update Function for Plotting ---
    def update(self, val):
        """
        Updates the spiral plot based on current slider values and textbox inputs.
        Recalculates points, updates plot limits, and redraws.
        """
        # Get D-Gap from internal stored value
        current_initial_d = self._current_initial_d

        current_num_turns = self.s_num_turns.val

        num_sides_from_polygon = self._get_current_num_sides()
        initial_polygon_vertices = self._get_initial_polygon_vertices(num_sides_from_polygon)

        total_points_to_generate = 1 + int(current_num_turns * num_sides_from_polygon)

        if total_points_to_generate < num_sides_from_polygon:
            total_points_to_generate = num_sides_from_polygon

        try:
            # Generate spiral points (centerline)
            generated_points = self.generate_polygon_spiral(
                total_points_to_generate,
                current_initial_d,
                initial_polygon_vertices,
                num_sides_from_polygon
            )
            self.current_spiral_points = generated_points

        except ValueError as e:
            print(f"Error generating spiral: {e}")
            self.current_spiral_points = []

        if self.current_spiral_points:
            # Update the spiral centerline plot
            x_coords = [p[0] for p in self.current_spiral_points]
            y_coords = [p[1] for p in self.current_spiral_points]
            self.spiral_line.set_data(x_coords, y_coords)

            # Calculate and display bounding box dimensions
            if x_coords and y_coords:
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                x_length = x_max - x_min
                y_length = y_max - y_min

                self.bbox_text.set_text(f"X-Length: {x_length:.2f}\nY-Length: {y_length:.2f}")

                # Change spiral line color based on max lengths
                if x_length > self._current_max_x_length or y_length > self._current_max_y_length:
                    self.spiral_line.set_color('red')
                else:
                    self.spiral_line.set_color('green') # Default valid color

                # Adjust plot limits based on current_spiral_points
                self.ax.set_xlim(x_min - 10, x_max + 10)
                self.ax.set_ylim(y_min - 10, y_max + 10)
            else:
                self.bbox_text.set_text("") # Clear text if no points
                self.spiral_line.set_data([], []) # Clear spiral line
                self.spiral_line.set_color('green') # Reset color to default if no points
                # Use initial plot limits from config if no points
                self.ax.set_xlim(self.initial_plot_xlim_min, self.initial_plot_xlim_max)
                self.ax.set_ylim(self.initial_plot_ylim_min, self.initial_plot_ylim_max)
        else:
            # Clear the plot if no points are generated
            self.spiral_line.set_data([], [])
            self.bbox_text.set_text("") # Clear text if no points
            self.spiral_line.set_color('green') # Reset color to default if no points
            # Reset plot limits to initial config if no spiral points
            self.ax.set_xlim(self.initial_plot_xlim_min, self.initial_plot_xlim_max)
            self.ax.set_ylim(self.initial_plot_ylim_min, self.initial_plot_ylim_max)

        self.fig.canvas.draw_idle()

    # --- Helper methods for update() (to be implemented by children) ---
    def _get_current_num_sides(self):
        """Returns the current number of sides based on slider or initial polygon."""
        raise NotImplementedError("Subclasses must implement _get_current_num_sides.")

    def _get_initial_polygon_vertices(self, num_sides):
        """Returns the initial polygon vertices for generation."""
        raise NotImplementedError("Subclasses must implement _get_initial_polygon_vertices.")

    # --- Reset Function ---
    def reset(self, event):
        """
        Resets common slider values and textbox values to defaults. Children will implement
        to reset their specific sliders.
        """
        self.d_gap_textbox.set_val(f"{self.initial_fixed_distance_d:.2f}")
        self._current_initial_d = self.initial_fixed_distance_d

        self.max_x_length_textbox.set_val(f"{self.initial_max_x_length:.2f}")
        self._current_max_x_length = self.initial_max_x_length

        self.max_y_length_textbox.set_val(f"{self.initial_max_y_length:.2f}")
        self._current_max_y_length = self.initial_max_y_length

        self.s_num_turns.set_val(self.initial_num_turns)
        # update(None) will be called by the child's reset after its specific sliders are set.
        self.update(None) # Call update to reflect reset values immediately


    # --- DXF Export Function ---
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

        # Export spiral centerline
        if self.current_spiral_points:
            dxf_points = [(p[0], p[1], 0) for p in self.current_spiral_points]
            msp.add_lwpolyline(dxf_points, dxfattribs={'color': 2, 'layer': 'SPIRAL_CENTERLINE'}) # Color 2 is yellow in AutoCAD
            print("Spiral centerline exported to DXF.")
        else:
            print("No spiral centerline points to export.")


        filename = "polygon_spiral_centerline.dxf"
        try:
            doc.saveas(filename)
            print(f"Spiral centerline data exported to {os.path.abspath(filename)}")
        except Exception as e:
            print(f"Error saving DXF file: {e}")

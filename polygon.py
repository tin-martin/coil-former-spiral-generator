# polygon_spiral_app_base.py
# Defines the base class for polygon spiral applications.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import os # For saving files
import ezdxf # DXF export library
import math # For mathematical operations like sqrt
import csv # For reading CSV configuration

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
        self.inner_to_outer = input("If inner to outer, type 0, otherwise type 1: ").strip().upper()
        while self.inner_to_outer not in ["0", "1"]:
            print("\nInvalid input! Please type 0 or 1.")
            self.inner_to_outer = input("If inner to outer, type 0. If not, type 1: ").strip().upper()

        # Get user input for gap type (square root non-linear or linear)
        self.square_root = input("If square root (non-linear gap), type 0, if linear type 1: ").strip().upper()
        while self.square_root not in ["0", "1"]:
            print("\nInvalid input! Please type 0 or 1.")
            self.square_root = input("If square root, type 0, if linear type 1: ").strip().upper()

        # Set up the Matplotlib figure and axes
        self.fig, self.ax = plt.subplots(figsize=(10, 7.5))
        # Adjusted bottom margin to make more room for all sliders and buttons
        plt.subplots_adjust(left=0.1, bottom=0.45)


        # Initialize plot elements (line, scatter, labels)
        self.current_spiral_points = [] # Stores generated points for DXF export

        # Initialize guideline plot elements
        self.guideline1_line, = self.ax.plot([], [], linestyle='-', color='red', linewidth=1.5, label='Guideline 1')
        self.guideline2_line, = self.ax.plot([], [], linestyle='-', color='green', linewidth=1.5, label='Guideline 2')

        # Set plot labels, grid, and initial limits from config
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlim(self.initial_plot_xlim_min, self.initial_plot_xlim_max)
        self.ax.set_ylim(self.initial_plot_ylim_min, self.initial_plot_ylim_max)
        self.ax.legend()


        # Define slider axes for common spiral parameters (adjusted Y positions)
        self.ax_initial_d = plt.axes([0.1, 0.26, 0.8, 0.03], facecolor='lightgoldenrodyellow')
        self.ax_decrease_rate_factor = plt.axes([0.1, 0.22, 0.8, 0.03], facecolor='lightgoldenrodyellow')
        self.ax_num_turns = plt.axes([0.1, 0.18, 0.8, 0.03], facecolor='lightgoldenrodyellow')
        self.ax_trace_width = plt.axes([0.1, 0.14, 0.8, 0.03], facecolor='lightgoldenrodyellow')
        self.ax_trace_width_decrease_rate = plt.axes([0.1, 0.10, 0.8, 0.03], facecolor='lightgoldenrodyellow')


        # Create sliders for common spiral parameters
        self.s_initial_d = Slider(self.ax_initial_d, "D-Gap", self.min_initial_d, self.max_fixed_d, valinit=self.initial_fixed_distance_d, valstep=0.1)
        self.s_decrease_rate_factor = Slider(self.ax_decrease_rate_factor, 'D-Rate', -self.max_decrease_rate_factor, self.max_decrease_rate_factor, valinit=self.initial_decrease_rate_factor, valstep=0.01)
        self.s_num_turns = Slider(self.ax_num_turns, 'Turns', self.min_num_turns, self.max_num_turns, valinit=self.initial_num_turns, valstep=0.1)
        self.s_trace_width = Slider(self.ax_trace_width, 'Width', self.minimum_trace_width, self.max_trace_width, valinit=self.initial_trace_width, valstep=0.1)
        self.s_trace_width_decrease_rate = Slider(self.ax_trace_width_decrease_rate, 'Width-Rate', self.min_trace_width_decrease_rate, self.max_trace_width_decrease_rate, valinit=self.initial_trace_width_decrease_rate, valstep=0.01)

        if self.inner_to_outer == "1":
            if(self.trace_width_decrease_direction == "0"):
                self.trace_width_decrease_direction = "1"
            else:
                self.trace_width_decrease_direction = "0"

        # Define button axes (adjusted Y positions)
        self.ax_reset_button = plt.axes([0.35, 0.05, 0.1, 0.04])
        self.ax_export_button = plt.axes([0.55, 0.05, 0.1, 0.04])

        # Create buttons
        self.reset_button = Button(self.ax_reset_button, 'Reset', color='lightblue', hovercolor='0.975')
        self.export_button = Button(self.ax_export_button, 'Export DXF', color='lightgreen', hovercolor='0.975')

        # Attach update method to common slider changes
        self.s_initial_d.on_changed(self.update)
        self.s_decrease_rate_factor.on_changed(self.update)
        self.s_num_turns.on_changed(self.update)
        self.s_trace_width.on_changed(self.update)
        self.s_trace_width_decrease_rate.on_changed(self.update)

        # Attach methods to button clicks
        self.reset_button.on_clicked(self.reset)
        self.export_button.on_clicked(self.export_dxf)

    def _load_config_from_csv(self, filename):
        """
        Loads configuration parameters from a CSV file into instance attributes.
        """
        config = {}
        try:
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
        except FileNotFoundError:
            print(f"Error: Configuration file '{filename}' not found. Using default hardcoded values.")
            # Fallback to hardcoded defaults if file not found
            config = {
                'initial_fixed_distance_d': 10.0,
                'initial_decrease_rate_factor': 0.0,
                'initial_num_turns': 5.0,
                'max_fixed_d': 20.0,
                'max_decrease_rate_factor': 10.0,
                'max_num_turns': 50.0,
                'initial_plot_xlim_min': -50.0,
                'initial_plot_xlim_max': 50.0,
                'initial_plot_ylim_min': -50.0,
                'initial_plot_ylim_max': 50.0,
                'minimum_d_threshold': 0.1,
                'initial_trace_width': 1.0,
                'max_trace_width': 10.0,
                'initial_trace_width_decrease_rate': 0.1,
                'max_trace_width_decrease_rate': 1.0,
                'minimum_trace_width': 0.1,

                # New parameters for sliders' min values
                'min_initial_d': 0.1,
                'min_num_turns': 0.0,
                'min_trace_width_decrease_rate': 0.0,
                'min_num_sides': 3.0,
                'min_initial_len': 0.1,

                # New parameters for DXF export
                'dxf_insunits': 4.0,
                'dxf_guideline_color': 8.0,
                'dxf_guideline_layer1': 'TRACE_GUIDELINE_1',
                'dxf_guideline_layer2': 'TRACE_GUIDELINE_2',
            }
        except Exception as e:
            print(f"An unexpected error occurred while loading config from CSV: {e}. Using default hardcoded values.")
            # Fallback to hardcoded defaults for other errors
            config = {
                'initial_fixed_distance_d': 10.0,
                'initial_decrease_rate_factor': 0.0,
                'initial_num_turns': 5.0,
                'max_fixed_d': 20.0,
                'max_decrease_rate_factor': 10.0,
                'max_num_turns': 50.0,
                'initial_plot_xlim_min': -50.0,
                'initial_plot_xlim_max': 50.0,
                'initial_plot_ylim_min': -50.0,
                'initial_plot_ylim_max': 50.0,
                'initial_trace_width': 1.0,
                'max_trace_width': 10.0,
                'initial_trace_width_decrease_rate': 0.1,
                'max_trace_width_decrease_rate': 1.0,
                'minimum_trace_width': 0.1,

                # New parameters for sliders' min values
                'min_initial_d': 0.1,
                'min_num_turns': 0.0,
                'min_trace_width_decrease_rate': 0.0,
                'min_num_sides': 3.0,
                'min_initial_len': 0.1,

                # New parameters for DXF export
                'dxf_insunits': 4.0,
                'dxf_guideline_color': 8.0,
                'dxf_guideline_layer1': 'TRACE_GUIDELINE_1',
                'dxf_guideline_layer2': 'TRACE_GUIDELINE_2',
            }

        self.trace_width_decrease_direction = str(int(config.get('trace_width_decrease_direction', 1.0)))

        # Assign loaded (or default) values to instance attributes
        self.initial_fixed_distance_d = config.get('initial_fixed_distance_d', 10.0)
        self.initial_decrease_rate_factor = config.get('initial_decrease_rate_factor', 0.0)
        self.initial_num_turns = config.get('initial_num_turns', 5.0)

        # Ensure correct defaults if not found in config, for regular polygon specific
        self.initial_num_sides = config.get('initial_num_sides', 6.0)
        self.max_num_sides = config.get('max_num_sides', 6.0)
        self.max_initial_len = config.get('max_initial_len', 200.0)
        self.initial_segment_len = config.get('initial_segment_len', 20.0)

        self.max_fixed_d = config.get('max_fixed_d', 20.0)
        self.max_decrease_rate_factor = config.get('max_decrease_rate_factor', 10.0)
        self.max_num_turns = config.get('max_num_turns', 50.0)

        self.initial_plot_xlim_min = config.get('initial_plot_xlim_min', -50.0)
        self.initial_plot_xlim_max = config.get('initial_plot_xlim_max', 50.0)
        self.initial_plot_ylim_min = config.get('initial_plot_ylim_min', -50.0)
        self.initial_plot_ylim_max = config.get('initial_plot_ylim_max', 50.0)
        self.minimum_d_threshold = config.get('minimum_d_threshold', 0.1)

        self.initial_trace_width = config.get('initial_trace_width', 1.0)
        self.max_trace_width = config.get('max_trace_width', 10.0)
        self.initial_trace_width_decrease_rate = config.get('initial_trace_width_decrease_rate', 0.1)
        self.max_trace_width_decrease_rate = config.get('max_trace_width_decrease_rate', 1.0)
        self.minimum_trace_width = config.get('minimum_trace_width', 0.1)

        # New slider min value assignments
        self.min_initial_d = config.get('min_initial_d', 0.1)
        self.min_num_turns = config.get('min_num_turns', 0.0)
        self.min_trace_width_decrease_rate = config.get('min_trace_width_decrease_rate', 0.0)
        self.min_num_sides = config.get('min_num_sides', 3.0)
        self.min_initial_len = config.get('min_initial_len', 0.1)

        # New DXF export assignments, ensuring integer types for colors and units
        self.dxf_insunits = int(config.get('dxf_insunits', 4.0))
        self.dxf_guideline_color = int(config.get('dxf_guideline_color', 8.0))
        self.dxf_guideline_layer1 = config.get('dxf_guideline_layer1', 'TRACE_GUIDELINE_1')
        self.dxf_guideline_layer2 = config.get('dxf_guideline_layer2', 'TRACE_GUIDELINE_2')


    # --- Abstract/Common Spiral Generation Logic ---
    def decrease_trace_width_fcn(self,x):
        return x
    def generate_polygon_spiral(self, num_points: int, initial_d: float, decrease_rate_factor: float, outermost_polygon_vertices: list[tuple[float, float]], num_sides: int):
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
            if self.square_root == "0":
                effective_decrease = decrease_rate_factor * math.sqrt(num_turns_completed)
            else:
                effective_decrease = decrease_rate_factor * num_turns_completed

            current_d = max(self.minimum_d_threshold, initial_d - effective_decrease)

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

    # --- Update Function for Plotting ---
    def update(self, val):
        """
        Updates the spiral plot based on current slider values.
        Recalculates points, updates plot limits, and redraws.
        """
        current_initial_d = self.s_initial_d.val

        # Adjusted logic for decrease_rate_factor based on spiral direction
        if self.inner_to_outer == "0": # Inner to Outer - d should generally increase
            current_decrease_rate_factor = -self.s_decrease_rate_factor.val

        else: # Outer to Inner - d should generally decrease
            current_decrease_rate_factor = self.s_decrease_rate_factor.val

        current_num_turns = self.s_num_turns.val
        initial_trace_width_from_slider = self.s_trace_width.val
        trace_width_decrease_rate = self.s_trace_width_decrease_rate.val


        num_sides_from_polygon = self._get_current_num_sides()
        initial_polygon_vertices = self._get_initial_polygon_vertices(num_sides_from_polygon)

        total_points_to_generate = 1 + int(current_num_turns * num_sides_from_polygon)

        if total_points_to_generate < num_sides_from_polygon:
            total_points_to_generate = num_sides_from_polygon

        try:
            # Generate spiral points (centerline) - still needed for guideline calculation
            generated_points = self.generate_polygon_spiral(
                total_points_to_generate,
                current_initial_d,
                current_decrease_rate_factor,
                initial_polygon_vertices,
                num_sides_from_polygon
            )
            self.current_spiral_points = generated_points

        except ValueError as e:
            print(f"Error generating spiral: {e}")
            self.current_spiral_points = []

        if self.current_spiral_points:
            # --- Calculate and plot guidelines using intersection method ---
            offset_points_line1 = []
            offset_points_line2 = []

            if len(self.current_spiral_points) < 2:
                # Not enough points to draw lines, just clear guidelines
                self.guideline1_line.set_data([], [])
                self.guideline2_line.set_data([], [])
            else:
                # Handle the very first point of the guidelines (index 0)
                # Determine initial effective trace width based on direction and per-segment rate
                if self.trace_width_decrease_direction == "0": # Inner to Outer decrease
                    # Start (innermost) has smallest width. Max width is initial_trace_width_from_slider (at outermost).
                    effective_trace_width_start = max(self.minimum_trace_width, initial_trace_width_from_slider - (trace_width_decrease_rate * (total_points_to_generate - 2)))
                else: # Outer to Inner decrease
                    # Start (outermost) has initial_trace_width_from_slider.
                    effective_trace_width_start = initial_trace_width_from_slider

                offset_val_start = effective_trace_width_start / 2.0

                p_start = self.current_spiral_points[0]
                p_next_to_start = self.current_spiral_points[1]
                tangent_start = p_next_to_start - p_start
                norm_tangent_start = np.linalg.norm(tangent_start)

                if norm_tangent_start > 1e-9:
                    unit_tangent_start = tangent_start / norm_tangent_start
                    unit_normal_start = np.array([-unit_tangent_start[1], unit_tangent_start[0]])
                    offset_points_line1.append(p_start + unit_normal_start * offset_val_start)
                    offset_points_line2.append(p_start - unit_normal_start * offset_val_start)
                else: # If the very first segment is tiny, just add p_start to lists
                     offset_points_line1.append(p_start)
                     offset_points_line2.append(p_start)


                for i in range(len(self.current_spiral_points) - 1): # Iterate up to the last segment
                    p_prev = self.current_spiral_points[i]
                    p_curr = self.current_spiral_points[i+1]
                    p_next = self.current_spiral_points[i+2] if (i+2) < len(self.current_spiral_points) else self.current_spiral_points[i+1] + (self.current_spiral_points[i+1] - self.current_spiral_points[i]) # Extrapolate if it's the very last segment

                    # Calculate effective trace width for the current point (p_curr), which corresponds to index (i+1)
                    if self.trace_width_decrease_direction == "0": # Inner to Outer decrease
                        # Width should increase from inner to outer. Initial_trace_width is max (outermost).
                        effective_trace_width_curr = max(self.minimum_trace_width, initial_trace_width_from_slider - (trace_width_decrease_rate * (total_points_to_generate - 2 - (i + 1))))
                    else: # Outer to Inner decrease
                        # Width should decrease from outer to inner. Initial_trace_width is max (outermost).
                        effective_trace_width_curr = max(self.minimum_trace_width, initial_trace_width_from_slider - (trace_width_decrease_rate * (i + 1)))

                    offset_val_curr = effective_trace_width_curr / 2.0

                    # Calculate effective trace width for the previous point (p_prev), which corresponds to index (i)
                    if self.trace_width_decrease_direction == "0": # Inner to Outer decrease
                        effective_trace_width_prev = max(self.minimum_trace_width, initial_trace_width_from_slider - (trace_width_decrease_rate * (total_points_to_generate - 2 - i)))
                    else: # Outer to Inner decrease
                        effective_trace_width_prev = max(self.minimum_trace_width, initial_trace_width_from_slider - (trace_width_decrease_rate * i))

                    offset_val_prev = effective_trace_width_prev / 2.0


                    # Segment 1: p_prev to p_curr
                    tangent_segment1 = p_curr - p_prev
                    norm_segment1 = np.linalg.norm(tangent_segment1)
                    if norm_segment1 < 1e-9: continue

                    unit_tangent_segment1 = tangent_segment1 / norm_segment1
                    unit_normal_segment1 = np.array([-unit_tangent_segment1[1], unit_tangent_segment1[0]])

                    # Offset line points for segment 1 using trace width at p_prev
                    s1_p1_offset1 = p_prev + unit_normal_segment1 * offset_val_prev
                    s1_p2_offset1 = p_curr + unit_normal_segment1 * offset_val_prev

                    s1_p1_offset2 = p_prev - unit_normal_segment1 * offset_val_prev
                    s1_p2_offset2 = p_curr - unit_normal_segment1 * offset_val_prev


                    # Segment 2: p_curr to p_next
                    tangent_segment2 = p_next - p_curr
                    norm_segment2 = np.linalg.norm(tangent_segment2)
                    if norm_segment2 < 1e-9: continue

                    unit_tangent_segment2 = tangent_segment2 / norm_segment2
                    unit_normal_segment2 = np.array([-unit_tangent_segment2[1], unit_tangent_segment2[0]])

                    # Offset line points for segment 2 using trace width at p_curr
                    s2_p1_offset1 = p_curr + unit_normal_segment2 * offset_val_curr
                    s2_p2_offset1 = p_next + unit_normal_segment2 * offset_val_curr

                    s2_p1_offset2 = p_curr - unit_normal_segment2 * offset_val_curr
                    s2_p2_offset2 = p_next - unit_normal_segment2 * offset_val_curr

                    # Find intersections
                    intersection1 = self._find_line_intersection(s1_p1_offset1, s1_p2_offset1, s2_p1_offset1, s2_p2_offset1)
                    intersection2 = self._find_line_intersection(s1_p1_offset2, s1_p2_offset2, s2_p1_offset2, s2_p2_offset2)

                    if intersection1 is not None:
                        offset_points_line1.append(intersection1)
                    else: # If parallel, just use the end point of the previous offset segment
                        offset_points_line1.append(s1_p2_offset1)

                    if intersection2 is not None:
                        offset_points_line2.append(intersection2)
                    else: # If parallel, just use the end point of the previous offset segment
                        offset_points_line2.append(s1_p2_offset2)

                line1_first_distance = np.linalg.norm(offset_points_line1[0] - offset_points_line1[1])
                line1_last_distance = np.linalg.norm(offset_points_line1[-1] - offset_points_line1[-2])
                line2_first_distance = np.linalg.norm(offset_points_line2[0] - offset_points_line2[1])
                line2_last_distance = np.linalg.norm(offset_points_line2[-1] - offset_points_line2[-2])

                # --- Adjusting offset_points_line1[0] (First point of line 1) ---
                vector_first_line1 = offset_points_line1[1] - offset_points_line1[0] #Vector from 1st to 2nd point
                norm_vector_first_line1 = np.linalg.norm(vector_first_line1)
                if norm_vector_first_line1 > 1e-9: # Check to avoid division by zero
                    unit_vector_first_line1 = vector_first_line1 / norm_vector_first_line1 # Removed np.abs()
                    # Moves the first point towards the second point by half its segment length
                    adjusted_point_line1_first = offset_points_line1[0] + unit_vector_first_line1 * (line1_first_distance / 2.0)
                    offset_points_line1[0] = adjusted_point_line1_first

                
                # --- Adjusting offset_points_line1[-1] (Last point of line 1) ---
                vector_last_line1 = offset_points_line1[-1] - offset_points_line1[-2] # Vector from 2nd-to-last to last point
                norm_vector_last_line1 = np.linalg.norm(vector_last_line1)
                if norm_vector_last_line1 > 1e-9: # Check to avoid division by zero
                    unit_vector_last_line1 = vector_last_line1 / norm_vector_last_line1 # Removed np.abs()
                    # Moves the last point towards the second-to-last point by half its segment length
                    adjusted_point_line1_last = offset_points_line1[-1] - unit_vector_last_line1 * (line1_last_distance / 2.0)
                    offset_points_line1[-1] = adjusted_point_line1_last

    
                # --- Adjusting offset_points_line2[0] (First point of line 2) ---
                vector_first_line2 = offset_points_line2[1] - offset_points_line2[0] # Vector from 1st to 2nd point
                norm_vector_first_line2 = np.linalg.norm(vector_first_line2)
                if norm_vector_first_line2 > 1e-9: # Check to avoid division by zero
                    unit_vector_first_line2 = vector_first_line2 / norm_vector_first_line2 # Removed np.abs()
                    # Moves the first point towards the second point by half its segment length
                    adjusted_point_line2_first = offset_points_line2[0] + unit_vector_first_line2 * (line2_first_distance / 2.0)
                    offset_points_line2[0] = adjusted_point_line2_first


                # --- Adjusting offset_points_line2[-1] (Last point of line 2) ---
                vector_last_line2 = offset_points_line2[-1] - offset_points_line2[-2] # Vector from 2nd-to-last to last point
                norm_vector_last_line2 = np.linalg.norm(vector_last_line2)
                if norm_vector_last_line2 > 1e-9: # Check to avoid division by zero
                    unit_vector_last_line2 = vector_last_line2 / norm_vector_last_line2 # CORRECTED DENOMINATOR, removed np.abs()
                    # Moves the last point towards the second-to-last point by half its segment length
                    adjusted_point_line2_last = offset_points_line2[-1] - unit_vector_last_line2 * (line2_last_distance / 2.0)
                    offset_points_line2[-1] = adjusted_point_line2_last
                # print(unit_vector_last_line2) # You can print it here if you still want to inspect

                        

                # These calculations should be placed AFTER offset_points_line1 and offset_points_line2
                # are fully populated in your update method.

                # Recalculate distances based on the current state of the points
                line1_first_distance = np.linalg.norm(offset_points_line1[0] - offset_points_line1[1])
                line1_last_distance = np.linalg.norm(offset_points_line1[-1] - offset_points_line1[-2])
                line2_first_distance = np.linalg.norm(offset_points_line2[0] - offset_points_line2[1])
                line2_last_distance = np.linalg.norm(offset_points_line2[-1] - offset_points_line2[-2])

                # Ensure there are enough points in the lists before attempting adjustments
                if len(offset_points_line1) >= 2 and len(offset_points_line2) >= 2:
                    # --- Set offset_points_line1[-1] as the intersection point ---
                    # Line 1 (Segment of guideline 1 that will be adjusted)
                    p1_segment1_line1 = offset_points_line1[-2]
                    p2_segment1_line1 = offset_points_line1[-1]

                    # Line 2 (Perpendicular to last segment of guideline 2, touching offset_points_line2[-1])
                    vector_last_line2_segment = offset_points_line2[-1] - offset_points_line2[-2] # Vector from 2nd-to-last to last point of line 2
                    norm_vector_last_line2_segment = np.linalg.norm(vector_last_line2_segment)

                    if norm_vector_last_line2_segment > 1e-9: # Check to avoid division by zero
                        # Unit perpendicular vector to the last segment of line 2
                        unit_perp_vector_last_line2 = np.array([-vector_last_line2_segment[1], vector_last_line2_segment[0]]) / norm_vector_last_line2_segment
                        
                        # Points defining the perpendicular line
                        p3_perp_line2 = offset_points_line2[-1]
                        # Extend the perpendicular line for intersection calculation (using a sufficiently large multiplier)
                        p4_perp_line2 = offset_points_line2[-1] + unit_perp_vector_last_line2 * 100.0 

                        intersection_last = self._find_line_intersection(p1_segment1_line1, p2_segment1_line1, p3_perp_line2, p4_perp_line2)

                        if intersection_last is not None:
                            offset_points_line1[-1] = intersection_last
                        else:
                            print("Warning: Lines are parallel or collinear, cannot find intersection for offset_points_line1[-1].")
                    else:
                        print("Warning: Last segment of offset_points_line2 is too short to calculate perpendicular for offset_points_line1[-1].")

                    # --- Set offset_points_line1[0] as the intersection point (similar logic) ---
                    # Line 1 (Segment of guideline 1 that will be adjusted)
                    p1_segment1_line1_first = offset_points_line1[0]
                    p2_segment1_line1_first = offset_points_line1[1]

                    # Line 2 (Perpendicular to first segment of guideline 2, touching offset_points_line2[0])
                    vector_first_line2_segment = offset_points_line2[1] - offset_points_line2[0] # Vector from 1st to 2nd point of line 2
                    norm_vector_first_line2_segment = np.linalg.norm(vector_first_line2_segment)

                    if norm_vector_first_line2_segment > 1e-9: # Check to avoid division by zero
                        # Unit perpendicular vector to the first segment of line 2
                        unit_perp_vector_first_line2 = np.array([-vector_first_line2_segment[1], vector_first_line2_segment[0]]) / norm_vector_first_line2_segment
                        
                        # Points defining the perpendicular line
                        p3_perp_line2_first = offset_points_line2[0]
                        # Extend the perpendicular line for intersection calculation
                        p4_perp_line2_first = offset_points_line2[0] + unit_perp_vector_first_line2 * 100.0 

                        intersection_first = self._find_line_intersection(p1_segment1_line1_first, p2_segment1_line1_first, p3_perp_line2_first, p4_perp_line2_first)

                        if intersection_first is not None:
                            offset_points_line1[0] = intersection_first
                        else:
                            print("Warning: Lines are parallel or collinear, cannot find intersection for offset_points_line1[0].")
                    else:
                        print("Warning: First segment of offset_points_line2 is too short to calculate perpendicular for offset_points_line1[0].")
                else:
                    print("Not enough points in offset_points_line1 or offset_points_line2 to perform end point adjustments.")


                # --- Connect endpoints to form a closed shape ---
                closed_shape_points = []
                # Add points from guideline 1
                closed_shape_points.extend(offset_points_line1)
                # Add points from guideline 2 in reverse to connect ends
                closed_shape_points.extend(offset_points_line2[::-1])
                # Close the loop by adding the first point of guideline 1 again
                if offset_points_line1:
                    closed_shape_points.append(offset_points_line1[0])

                self.guideline1_line.set_data([p[0] for p in closed_shape_points], [p[1] for p in closed_shape_points])
                self.guideline2_line.set_data([], []) # Clear guideline2_line as it's now part of the closed shape


            # --- End guideline calculation and plot ---


            # Adjust plot limits
            # Only consider guideline points for limits
            all_x_coords = [p[0] for p in closed_shape_points]
            all_y_coords = [p[1] for p in closed_shape_points]


            if all_x_coords and all_y_coords: # Ensure there are points to set limits with
                self.ax.set_xlim(min(all_x_coords) - 10, max(all_x_coords) + 10)
                self.ax.set_ylim(min(all_y_coords) - 10, max(all_y_coords) + 10)
            else:
                # Use initial plot limits from config if no points or single point
                self.ax.set_xlim(self.initial_plot_xlim_min, self.initial_plot_xlim_max)
                self.ax.set_ylim(self.initial_plot_ylim_min, self.initial_plot_ylim_max)


         

        self.fig.canvas.draw_idle()


    # --- Helper method to find line-line intersection ---
    def _find_line_intersection(self, p1, p2, p3, p4):
        """
        Finds the intersection point of two lines defined by (p1, p2) and (p3, p4).
        Returns the intersection point as a numpy array, or None if lines are parallel.
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        if abs(den) < 1e-9:  # Lines are parallel or collinear
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den

        intersection_x = x1 + t * (x2 - x1)
        intersection_y = y1 + t * (y2 - y1)
        return np.array([intersection_x, intersection_y])


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
        Resets common slider values to defaults. Children will implementdecrease_rate_factor
        to reset their specific sliders.
        """
        self.s_initial_d.set_val(self.initial_fixed_distance_d)
        self.s_decrease_rate_factor.set_val(self.initial_decrease_rate_factor)
        self.s_num_turns.set_val(self.initial_num_turns)
        self.s_trace_width.set_val(self.initial_trace_width)
        self.s_trace_width_decrease_rate.set_val(self.initial_trace_width_decrease_rate)
        # update(None) will be called by the child's reset after its specific sliders are set.


    # --- DXF Export Function ---
    def export_dxf(self, event):
        """
        Exports the generated spiral points and guidelines to a DXF file.
        """
        if not self.current_spiral_points:
            print("No spiral points to export.")
            return

        doc = ezdxf.new('R2010')
        doc.header['$INSUNITS'] = self.dxf_insunits

        msp = doc.modelspace()

        # Export guidelines
        if self.guideline1_line: # Only guideline1_line is used for the closed shape
            guideline1_coords = list(zip(self.guideline1_line.get_xdata(), self.guideline1_line.get_ydata()))

            if guideline1_coords:
                dxf_points1 = [(p[0], p[1], 0) for p in guideline1_coords]

                # Export as a closed polyline
                msp.add_lwpolyline(dxf_points1, close=True, dxfattribs={'color': self.dxf_guideline_color, 'layer': self.dxf_guideline_layer1})
                print("Closed guideline shape exported to DXF.")
            else:
                print("No guideline points to export.")
        else:
            print("No guideline to export.")


        filename = "polygon_spiral.dxf"
        try:
            doc.saveas(filename)
            print(f"Spiral and guideline data exported to {os.path.abspath(filename)}")
        except Exception as e:
            print(f"Error saving DXF file: {e}")
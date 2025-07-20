# regular.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox # Import TextBox

from polygon import PolygonSpiralApp

class RegularPolygonSpiralApp(PolygonSpiralApp):
    def __init__(self):
        super().__init__()

        self.ax_num_sides = plt.axes([0.1, 0.34, 0.8, 0.03], facecolor='lightgoldenrodyellow')
        self.s_num_sides = Slider(self.ax_num_sides, 'Sides', 3, self.max_num_sides, valinit=self.initial_num_sides, valstep=1)
        self.s_num_sides.on_changed(self.update)

        # Change from Slider to TextBox for Initial Length
        self.ax_initial_len = plt.axes([0.1+0.2, 0.25, 0.1, 0.03], facecolor='lightgoldenrodyellow') # Adjusted position
        self.initial_len_textbox = TextBox(self.ax_initial_len, "Init. Length", initial=str(self.initial_segment_len))
        self.initial_len_textbox.on_submit(self._submit_initial_len)
        self._current_initial_len = self.initial_segment_len # Store current value

        if self.inner_to_outer == "0":
            self.ax.set_title("Inner-to-Outer Regular Polygon Spiral")
        else:
            self.ax.set_title("Outer-to-Inner Regular Polygon Spiral")

        self.update(None)

    def _submit_initial_len(self, text):
        """
        Callback for Initial Length text box submission.
        Parses the input, updates the internal value, and triggers plot update.
        """
        try:
            val = float(text)
            # Apply min/max constraints
            if val < self.min_initial_len:
                val = self.min_initial_len
                self.initial_len_textbox.set_val(f"{val:.2f}") # Update textbox if value was clamped
            elif val > self.max_initial_len:
                val = self.max_initial_len
                self.initial_len_textbox.set_val(f"{val:.2f}") # Update textbox if value was clamped
            self._current_initial_len = val
            self.update(None) # Call update after value is set
        except ValueError:
            print("Invalid input for Initial Length. Please enter a number.")
            # Revert to last valid value
            self.initial_len_textbox.set_val(f"{self._current_initial_len:.2f}")
        self.fig.canvas.draw_idle()


    def _get_current_num_sides(self):
        return int(self.s_num_sides.val)

    def _get_initial_polygon_vertices(self, num_sides):
        current_initial_len = self._current_initial_len # Get from internal stored value
        outermost_polygon_vertices = []
        outermost_polygon_vertices.append(np.array([0.0, 0.0]))
        angle_per_side_initial = 360 / num_sides
        for i in range(1, num_sides):
            theta_degrees = (i - 1) * angle_per_side_initial
            theta_radians = np.radians(theta_degrees)
            direction_vector = np.array([np.cos(theta_radians), np.sin(theta_radians)])
            Q_i = outermost_polygon_vertices[i-1] + current_initial_len * direction_vector
            outermost_polygon_vertices.append(Q_i)
        return outermost_polygon_vertices

    def reset(self, event):
        self.s_num_sides.set_val(self.initial_num_sides)
        # Reset the TextBox for Initial Length
        self.initial_len_textbox.set_val(f"{self.initial_segment_len:.2f}")
        self._current_initial_len = self.initial_segment_len

        super().reset(event)
        # self.update(None) is called by super().reset(event)
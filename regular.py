# regular.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from polygon import PolygonSpiralApp

class RegularPolygonSpiralApp(PolygonSpiralApp):
    def __init__(self):
        super().__init__()

        self.ax_num_sides = plt.axes([0.1, 0.34, 0.8, 0.03], facecolor='lightgoldenrodyellow')
        self.s_num_sides = Slider(self.ax_num_sides, 'Sides', 3, self.max_num_sides, valinit=self.initial_num_sides, valstep=1)
        self.s_num_sides.on_changed(self.update)

        self.ax_initial_len = plt.axes([0.1, 0.30, 0.8, 0.03], facecolor='lightgoldenrodyellow')
        self.s_initial_len = Slider(self.ax_initial_len, "Init. Length", 0.1, self.max_initial_len, valinit=self.initial_segment_len, valstep=0.1)
        self.s_initial_len.on_changed(self.update)

        if self.inner_to_outer == "0":
            self.ax.set_title("Inner-to-Outer Regular Polygon Spiral")
        else:
            self.ax.set_title("Outer-to-Inner Regular Polygon Spiral")

        self.update(None)


    def _get_current_num_sides(self):
        return int(self.s_num_sides.val)

    def _get_initial_polygon_vertices(self, num_sides):
        current_initial_len = self.s_initial_len.val
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
        self.s_initial_len.set_val(self.initial_segment_len)
        super().reset(event)
        self.update(None)
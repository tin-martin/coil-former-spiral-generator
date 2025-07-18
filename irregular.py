# irregular.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import ezdxf

import selector
from polygon import PolygonSpiralApp

class IrregularPolygonSpiralApp(PolygonSpiralApp):
    def __init__(self):
        draw_points = input("To pull points from .csv, type 0. To draw points, type 1: ").strip().upper()

        while draw_points not in ["0", "1"]:
            print("\nInvalid input! Please type 0 or 1.")
            draw_points = input("To pull points from .csv, type. To draw points, type 1: ").strip().upper()

        if draw_points == "0":
            self.initial_irregular_polygon =  selector.load_points_from_csv()
        else:
            selector_instance = selector.PointSelector()
            selector_instance.show()
            self.initial_irregular_polygon = selector_instance.selected_points

        super().__init__()
        if self.inner_to_outer == "0":
            self.ax.set_title("Inner-to-Outer Irregular Polygon Spiral")
        else:
            self.ax.set_title("Outer-to-Inner Irregular Polygon Spiral")

        self.update(None)


    def _get_current_num_sides(self):
        return len(self.initial_irregular_polygon)

    def _get_initial_polygon_vertices(self, num_sides):
        return self.initial_irregular_polygon

    def reset(self, event):
        self.s_initial_d.set_val(self.initial_fixed_distance_d)
        self.s_decrease_rate_factor.set_val(self.initial_decrease_rate_factor)
        self.s_num_turns.set_val(self.initial_num_turns)
        self.update(None)

# --- Main Execution Block ---
if __name__ == "__main__":
    app = IrregularPolygonSpiralApp()
    plt.show()
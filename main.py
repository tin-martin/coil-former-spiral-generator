# main.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import os # For saving files
import ezdxf # New import for DXF export
import math # Import math for sqrtQ
from regular import RegularPolygonSpiralApp
from irregular import IrregularPolygonSpiralApp

# --- User Input for Spiral Type ---
is_regular = input("For regular polygon, type 0. For irregular polygon, type 1: ").strip().upper()
while(is_regular not in ["0", "1"]):
    print("\nInvalid input! Please type 0 or 1.")
    is_regular = input("To drawn regular polygon, type 0, otherwise type 1: ").strip().upper()

if(is_regular == "0"):
    app = RegularPolygonSpiralApp()
else:
    app = IrregularPolygonSpiralApp()

plt.show()
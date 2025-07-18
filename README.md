# PCB Polygon Spiral Generator

A Python-based tool to generate polygonal spirals for PCB design. This utility is designed for applications such as creating inductive coils, RF traces, and aesthetic spirals in KiCad PCB layouts.

---

## ✨ Features
- Generate **polygonal spirals** with customizable parameters.
- Supports **inner-to-outer** or **outer-to-inner** spiral directions.
- Adjustable **trace width**, **spacing**, and **turn count**.
- Exports results as **DXF files** or visualizes spirals using Matplotlib.
- Interactive configuration via CSV file and CLI options.

---

## 🛠 Installation
Clone the repository:
```bash
git clone https://github.com/tin-martin/pcb-polygon-spiral-generator.git
cd pcb-polygon-spiral-generator
```

Install dependencies:
```bash
pip install -r requirements.txt
```
*(Dependencies include `numpy`, `matplotlib`, `ezdxf`.)*

---

## ▶ Usage
Run the generator script:
```bash
python polygon_spiral_app_base.py
```

You will be prompted to:
- Select spiral direction (inner-to-outer or outer-to-inner).
- Load configuration from `polygon_spiral_config.csv`.
- Adjust parameters using the Matplotlib interface sliders.

### Key Parameters
- **trace_width**: Initial width of the spiral trace.
- **trace_width_decrease_rate**: How much the width decreases per turn.
- **minimum_trace_width**: Lower bound for trace width.
- **num_turns**: Number of spiral turns.
- **num_sides**: Polygon sides (e.g., 6 for hexagonal spiral).

---

## 📁 Output
- **DXF Export**: Generated spiral can be saved as a DXF file for importing into KiCad.
- **Visualization**: Real-time spiral preview in Matplotlib with sliders for tuning.

---

## ✅ Example Workflow
1. Adjust parameters in `polygon_spiral_config.csv`:
```csv
trace_width, initial_trace_width
trace_width_decrease_rate, 0.05
minimum_trace_width, 0.3
num_turns, 10
num_sides, 6
```

2. Run the script:
```bash
python polygon_spiral_app_base.py
```

3. Interactively fine-tune using sliders and export to DXF.

---

## 📌 Applications
- Wireless power transfer (WPT) coil design.
- Inductive charging prototypes.
- Aesthetic PCB spiral elements.
- Experimentation with polygonal coil geometries.

---

## 🤝 Contributing
Pull requests are welcome! Please ensure code changes are well-documented.

---

## 📜 License
This project is released under the MIT License.

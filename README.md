# KAN TAMU Capstone

Kolmogorov-Arnold Networks (KAN) for modeling transient radiation effects on diodes. Capstone project at Texas A&M University.

## Files

| File | Description |
|------|-------------|
| `myKAN.ipynb` | Main Jupyter notebook. Loads experiment CSV data, trains a KAN model ([4,10,1] architecture) to predict diode voltage from time, PCD radiation, bias voltage, load resistance. Uses Adam or LBFGS optimizer with B-spline grid updates. Includes plotting of true vs. predicted voltage and a `single_test` function for individual experiment evaluation. Newest functions `perform_training` and `perform_LOO_training` automate training over multiple experiment subsets. Zeros must be put into the subset rows for fewer than 5 experiments.|
| `start_stop_pts.csv` | Experiment metadata: voltage/radiation start/stop indices, bias voltage, load resistance, and diode type for each run. Also contains quality notes (e.g., "oscillating", "noisy rise"). |
| `scenario_notes.xlsx` | Spreadsheet with scenario-related notes for the experiments. |
| `training_subsets.csv`| Each row is a subset of experiment numbers to pull from the `TestData` folder.
| `0TestData graphs.pptx` | PowerPoint presentation with test data graphs from the experiments. |

## Input Data

The notebook expects experiment CSV files at `TestData\LFXR_Lovejoy_Sept2021_{exp_num}.csv` containing time, PCD radiation, and diode voltage columns.

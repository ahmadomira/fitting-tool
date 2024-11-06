# Project Structure

## Directories

- **notebooks/**: the Jupyter notebooks for fitting
- **data/**: sample data files for testing


## Files
- **`main_interface.py`**: the main script for running the user interface. Run this script to start the fitting tool.
- The main interface script consolidates the following files:
    - `interface_DyeAlone_fitting.py`
    - `interface_GDA_fitting.py`
    - `interface_IDA_fitting.py`

    These files contain the fitting functions for the different models. 
- **`notebooks/*.ipynb`**: the reference Jupyter notebooks. They contain the fitting routines the main interface script implements.
- **`data/*.txt`**: sample data files for testing. The data files are in the format of the data files provided by the user.
- **`data/results/*.txt`**: sample results files of the fitting. 
- **`pltstyle.py`**: defines the plot style for the fitting results.
- **`FittingApp.spec`**: the PyInstaller configuration file for creating executables.

## Creating Executables with PyInstaller
The following command is used to create an executable from the main interface script:
```bash
pyinstaller --onefile --windowed --name 'FittingApp' \
    --add-data 'interface_DyeAlone_fitting.py:.' \
    --add-data 'interface_GDA_fitting.py:.' \
    --add-data 'interface_IDA_fitting.py:.' \
    --add-data 'pltstyle.py:.' \
    --hidden-import matplotlib \
    --hidden-import tkinter \
    --noconsole \
    --icon=MyIcon.icns \
    --distpath app \
    main_interface.py"
```
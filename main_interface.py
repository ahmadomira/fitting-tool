import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')

from interface_DyeAlone_fitting import DyeAloneFittingApp
from interface_GDA_fitting import GDAFittingApp
from interface_IDA_fitting import IDAFittingApp
from interface_DBA_host_to_dye_fitting import DBAFittingAppHtoD
from interface_DBA_dye_to_host_fitting import DBAFittingAppDtoH
from interface_ida_merge_fits import IDAMergeFitsApp
from interface_dba_merge_fits import DBAMergeFitsApp
from interface_gda_merge_fits import GDAMergeFitsApp
from plot_replica import PlotReplica
from bmg_to_txt import BMGToTxtConverter

import openpyxl

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Automation Project")
    
    # Set initial window width
    window_width = 300
    
    # Create all widgets (all the existing sections and buttons will follow)
    
    # Preprocess Data Section
    preprocess_data_label = tk.Label(root, text="Preprocess Data", font=("Arial", 16, "bold"))
    preprocess_data_label.pack(pady=5)

    def plot_replica():
        # Function to handle plot replica button click
        # Implement the functionality here or open a new window
        new_window = tk.Toplevel(root)
        new_window.title("Plot Raw Replica")
        PlotReplica(new_window)  # Call the PlotReplica class for implementation

    plot_replica_button = tk.Button(root, text="Plot Raw Replica", command=plot_replica)
    plot_replica_button.pack(pady=10, padx=15, fill=tk.X)
    
    def open_bmg_to_txt_converter():
        new_window = tk.Toplevel(root)
        new_window.title("Convert BMG to TXT")
        BMGToTxtConverter(new_window)

    bmg_to_txt_button = tk.Button(root, text="Convert BMG to TXT", command=open_bmg_to_txt_converter)
    bmg_to_txt_button.pack(pady=10, padx=15, fill=tk.X)

    # Add extra vertical space before the next section
    tk.Label(root, text="").pack(pady=5)

    # Fitting Section
    fitting_label = tk.Label(root, text="Fitting", font=("Arial", 16, "bold"))
    fitting_label.pack(pady=5)

    def open_dye_alone_fitting():
        new_window = tk.Toplevel(root)
        new_window.title("Dye Alone Fitting")
        DyeAloneFittingApp(new_window)

    def open_gda_fitting():
        new_window = tk.Toplevel(root)
        new_window.title("GDA Fitting")
        GDAFittingApp(new_window)

    def open_ida_fitting():
        new_window = tk.Toplevel(root)
        new_window.title("IDA Fitting")
        IDAFittingApp(new_window)

    def open_dba_host_to_dye_fitting():
        new_window = tk.Toplevel(root)
        new_window.title("DBA Fitting (Host to Dye)")
        DBAFittingAppHtoD(new_window)

    def open_dba_dye_to_host_fitting():
        new_window = tk.Toplevel(root)
        new_window.title("DBA Fitting (Dye to Host)")
        DBAFittingAppDtoH(new_window)

    def open_ida_merge_fits():
        new_window = tk.Toplevel(root)
        new_window.title("IDA Merge Fits")
        IDAMergeFitsApp(new_window)

    def open_dba_merge_fits():
        new_window = tk.Toplevel(root)
        new_window.title("DBA Merge Fits")
        DBAMergeFitsApp(new_window)

    def open_gda_merge_fits():
        new_window = tk.Toplevel(root)
        new_window.title("GDA Merge Fits")
        GDAMergeFitsApp(new_window)
        
    dye_alone_button = tk.Button(root, text="Dye Alone Fitting", command=open_dye_alone_fitting)
    dye_alone_button.pack(pady=10, padx=20, fill=tk.X)

    gda_fitting_button = tk.Button(root, text="GDA Fitting", command=open_gda_fitting)
    gda_fitting_button.pack(pady=10, padx=20, fill=tk.X)

    ida_fitting_button = tk.Button(root, text="IDA Fitting", command=open_ida_fitting)
    ida_fitting_button.pack(pady=10, padx=20, fill=tk.X)

    dba_host_to_dye_button = tk.Button(root, text="DBA Fitting (Host to Dye)", command=open_dba_host_to_dye_fitting)
    dba_host_to_dye_button.pack(pady=10, padx=20, fill=tk.X)

    dba_dye_to_host_button = tk.Button(root, text="DBA Fitting (Dye to Host)", command=open_dba_dye_to_host_fitting)
    dba_dye_to_host_button.pack(pady=10, padx=20, fill=tk.X)

    # Add extra vertical space before the next section
    tk.Label(root, text="").pack(pady=5)

    # Merging Results Section
    merging_label = tk.Label(root, text="Merging Results", font=("Arial", 16, "bold"))
    merging_label.pack(pady=5)

    ida_merge_fits_button = tk.Button(root, text="IDA Merge Fits", command=open_ida_merge_fits)
    ida_merge_fits_button.pack(pady=10, padx=20, fill=tk.X)

    dba_merge_fits_button = tk.Button(root, text="DBA Merge Fits", command=open_dba_merge_fits)
    dba_merge_fits_button.pack(pady=10, padx=20, fill=tk.X)

    gda_merge_fits_button = tk.Button(root, text="GDA Merge Fits", command=open_gda_merge_fits)
    gda_merge_fits_button.pack(pady=10, padx=20, fill=tk.X)

    # After packing all widgets, update the window to calculate required size
    root.update()
    
    # Get the required height after all widgets are packed
    window_height = root.winfo_reqheight() + 20  # Add a small buffer
    
    # Get the screen dimensions
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    # Calculate position coordinates for the window to be centered
    position_top = int(screen_height/2 - window_height/2)
    position_right = int(screen_width/2 - window_width/2)
    
    # Set the position of the window to the center of the screen with calculated height
    root.geometry(f"{window_width}x{window_height}+{position_right}+{position_top}")

    root.mainloop()
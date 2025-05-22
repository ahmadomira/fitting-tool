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

def main():
    root = tk.Tk()
    root.title("Automation Project")
    root.geometry("300x500")  # Adjusted to fit the new sections

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

    dba_host_to_dye_button = tk.Button(root, text="DBA Fitting (Host to Dye)", command=open_dba_host_to_dye_fitting)
    dba_host_to_dye_button.pack(pady=10, padx=20, fill=tk.X)

    dba_dye_to_host_button = tk.Button(root, text="DBA Fitting (Dye to Host)", command=open_dba_dye_to_host_fitting)
    dba_dye_to_host_button.pack(pady=10, padx=20, fill=tk.X)

    ida_fitting_button = tk.Button(root, text="IDA Fitting", command=open_ida_fitting)
    ida_fitting_button.pack(pady=10, padx=20, fill=tk.X)

    gda_fitting_button = tk.Button(root, text="GDA Fitting", command=open_gda_fitting)
    gda_fitting_button.pack(pady=10, padx=20, fill=tk.X)

    # Add extra vertical space before the next section
    tk.Label(root, text="").pack(pady=5)

    # Merging Results Section
    merging_label = tk.Label(root, text="Merging Results", font=("Arial", 16, "bold"))
    merging_label.pack(pady=5)

    dba_merge_fits_button = tk.Button(root, text="DBA Merge Fits", command=open_dba_merge_fits)
    dba_merge_fits_button.pack(pady=10, padx=20, fill=tk.X)

    ida_merge_fits_button = tk.Button(root, text="IDA Merge Fits", command=open_ida_merge_fits)
    ida_merge_fits_button.pack(pady=10, padx=20, fill=tk.X)

    gda_merge_fits_button = tk.Button(root, text="GDA Merge Fits", command=open_gda_merge_fits)
    gda_merge_fits_button.pack(pady=10, padx=20, fill=tk.X)

    # Bring the window to the front
    root.lift()
    root.focus_force()
    
    root.mainloop()

if __name__ == "__main__":
    main()
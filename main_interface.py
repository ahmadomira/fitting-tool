import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')

from interface_DyeAlone_fitting import DyeAloneFittingApp
from interface_GDA_fitting import GDAFittingApp
from interface_IDA_fitting import IDAFittingApp
from interface_DBA_host_to_dye_fitting import DBAFittingAppHtoD

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Automation Project")
    root.geometry("300x200")  # Adjusted to fit the new button
    
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

    def open_dba_fitting():
        new_window = tk.Toplevel(root)
        new_window.title("DBA Fitting")
        DBAFittingAppHtoD(new_window)

    dye_alone_button = tk.Button(root, text="Dye Alone Fitting", command=open_dye_alone_fitting)
    dye_alone_button.pack(pady=10, padx=20, fill=tk.X)

    gda_fitting_button = tk.Button(root, text="GDA Fitting", command=open_gda_fitting)
    gda_fitting_button.pack(pady=10, padx=20, fill=tk.X)

    ida_fitting_button = tk.Button(root, text="IDA Fitting", command=open_ida_fitting)
    ida_fitting_button.pack(pady=10, padx=20, fill=tk.X)

    dba_fitting_button = tk.Button(root, text="DBA Fitting", command=open_dba_fitting)
    dba_fitting_button.pack(pady=10, padx=20, fill=tk.X)

    root.mainloop()
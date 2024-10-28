import tkinter as tk

from interface_IDA_fitting_replica_dye_alone import DyeAloneFittingApp
from interface_GDA_fitting_update import GDAFittingApp


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Automation Project")
    root.geometry("300x200")

    def open_dye_alone_fitting():
        new_window = tk.Toplevel(root)
        new_window.title("Dye Alone Fitting")
        DyeAloneFittingApp(new_window)

    def open_gda_fitting():
        new_window = tk.Toplevel(root)
        new_window.title("GDA Fitting")
        GDAFittingApp(new_window)

    dye_alone_button = tk.Button(root, text="Dye Alone Fitting", command=open_dye_alone_fitting)
    dye_alone_button.pack(pady=10, padx=20, fill=tk.X)

    gda_fitting_button = tk.Button(root, text="GDA Fitting", command=open_gda_fitting)
    gda_fitting_button.pack(pady=10, padx=20, fill=tk.X)

    root.mainloop()
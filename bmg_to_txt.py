#!/usr/bin/env python3

import pandas as pd
from pathlib import Path

import tkinter as tk
from tkinter import filedialog
from tkinter import ttk

def read_bmg_xlsx(excel_path: str) -> pd.DataFrame:
    """read an Excel file saved from the BMG plate reader and return the data and protocol info as dataframes"""
    
    sheets_dict = pd.read_excel(excel_path, sheet_name=None)
    dfs = {sheet_name: sheet_data for sheet_name, sheet_data in sheets_dict.items()}
    data = list(dfs.items())[0][1]
    protocol_info = list(dfs.items())[1][1]
    
    data = extract_bmg_raw_data(data)
    
    return data, protocol_info

def extract_bmg_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """assuming the data frame has a row that contains "Raw Data" followed by the flourescence readings."""
    
    # locate flourescence readings in the dataframe
    mask = df[df.map(lambda x: "Raw Data" in str(x))]
    row_idx, col_idx = mask.stack().index[0]
    
    # move to numerical index (easier to move +n  or -n rows/cols)
    row_idx = df.index.get_loc(row_idx)
    col_idx = df.columns.get_loc(col_idx)
    
    bmg_data = df.loc[df.index[row_idx + 2:], df.columns[col_idx:]]
    
    # clean the data, invalid values are set to NaN
    bmg_data =bmg_data.apply(pd.to_numeric, errors='coerce')
    
    # change columns and rows labels (to match the plate layout)
    bmg_data.columns = range(1, len(bmg_data.columns) + 1)
    bmg_data.index = [chr(i) for i in range(ord('A'), ord('A') + len(bmg_data.index))]
    
    return bmg_data

def extract_concentration_vector(robot_file: str) -> list:
        """this is for a simple robot file with one analyte. Two sheets should Analyte"""
        
        sheet_names =['Analyte (20)', 'Analyte (300)']
        
        try:
            concent_1 = pd.read_excel(robot_file, sheet_name=sheet_names[0], header=None).iloc[2, 1:13].values
            concent_2 = pd.read_excel(robot_file, sheet_name=sheet_names[1], header=None).iloc[2, 1:13].values
            return concent_1 + concent_2
        
        except ValueError as e:
            print(f"Sheet 'Analyte (20)' or 'Analyte (300)' not found: {e}")

class BMGToTxtConverter:
    def __init__(self, root):
        self.root = root
        self.root.title("Convert BMG Data to Text Files")

        self.mainframe = ttk.Frame(root, padding="3 3 12 12")
        self.mainframe.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
        
        # Configure the column to expand
        self.mainframe.columnconfigure(1, weight=1)
        
        # create a label and entry for each file
        tk.Label(self.mainframe, text="BMG file:").grid(row=0, column=0, sticky=tk.W)
        tk.Label(self.mainframe, text="Robot file:").grid(row=1, column=0, sticky=tk.W)
        tk.Label(self.mainframe, text="Save directory:").grid(row=2, column=0, sticky=tk.W)
        
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        self.bmg_file_entry = tk.Entry(self.mainframe)
        self.robot_file_entry = tk.Entry(self.mainframe)
        self.save_dir_entry = tk.Entry(self.mainframe)

        self.bmg_file_entry.grid(row=0, column=1, sticky=(tk.W, tk.E))
        self.robot_file_entry.grid(row=1, column=1, sticky=(tk.W, tk.E))
        self.save_dir_entry.grid(row=2, column=1, sticky=(tk.W, tk.E))

        # Add checkbox for custom save directory
        self.use_custom_dir = tk.BooleanVar()
        self.use_custom_dir.set(False)  # Default to using BMG file directory
        self.custom_dir_check = tk.Checkbutton(self.mainframe, text="Custom", 
                                              variable=self.use_custom_dir, 
                                              command=self.toggle_save_dir)
        self.custom_dir_check.grid(row=2, column=3, padx=5)

        # create buttons to select files
        tk.Button(self.mainframe, text="Browse", command=lambda: self.select_file(self.bmg_file_entry)).grid(row=0, column=2)
        tk.Button(self.mainframe, text="Browse", command=lambda: self.select_file(self.robot_file_entry)).grid(row=1, column=2)
        self.browse_dir_button = tk.Button(self.mainframe, text="Browse", command=lambda: self.select_directory(self.save_dir_entry))
        self.browse_dir_button.grid(row=2, column=2)

        # Initialize the save directory entry with default text
        self.save_dir_entry.insert(0, "(Default: Same as BMG file)")
        self.save_dir_entry.config(state='disabled')
        self.browse_dir_button.config(state='disabled')

        # create button to start the process
        self.status_label = tk.Label(self.mainframe, text="Ready")
        self.status_label.grid(row=4, column=0, columnspan=4)
        
        process_button = tk.Button(self.mainframe, text="Process", command=self.process_and_display_status)
        process_button.grid(row=3, column=0, columnspan=4, sticky=(tk.E, tk.W))
        
        for child in self.mainframe.winfo_children(): 
            child.grid_configure(padx=5, pady=5)

    def toggle_save_dir(self):
        """Toggle the save directory entry based on the checkbox state"""
        if self.use_custom_dir.get():
            # Enable custom directory
            self.save_dir_entry.config(state='normal')
            self.save_dir_entry.delete(0, tk.END)
            self.browse_dir_button.config(state='normal')
        else:
            # Disable and show default text
            self.save_dir_entry.delete(0, tk.END)
            self.save_dir_entry.insert(0, "(Default: Same as BMG file)")
            self.save_dir_entry.config(state='disabled')
            self.browse_dir_button.config(state='disabled')

    def select_file(self, entry_widget):
        # Open file explorer to select file  
        file = filedialog.askopenfilename()
        if file:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, file)
            
            # Update BMG file path - if the save directory is default, we show the BMG file's directory
            if entry_widget == self.bmg_file_entry and not self.use_custom_dir.get():
                self.save_dir_entry.config(state='normal')
                self.save_dir_entry.delete(0, tk.END)
                self.save_dir_entry.insert(0, "(Default: Same as BMG file)")
                self.save_dir_entry.config(state='disabled')
        return file
        
    def select_directory(self, entry_widget):
        # Open directory explorer to select directory
        directory = filedialog.askdirectory()
        if directory:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, directory)
        return directory

    def process_and_display_status(self):
        try:
            bmg_file = self.bmg_file_entry.get()
            robot_file = self.robot_file_entry.get()
            save_dir = None
            
            # Only use custom directory if the checkbox is checked
            if self.use_custom_dir.get():
                save_dir = self.save_dir_entry.get()
            
            if not bmg_file:
                self.status_label.config(text="Error: BMG file not specified", fg="red")
                return
                
            if not robot_file:
                self.status_label.config(text="Error: Robot file not specified", fg="red")
                return
                
            self.process_bmg_data(bmg_file, robot_file, save_dir)
            
            # Determine where files were saved to show in the status
            save_location = save_dir if save_dir else f"same directory as BMG file ({Path(bmg_file).parent})"
            self.status_label.config(text=f"Processing done. Files saved to {save_location}", fg="green")
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}", fg="red")

    def process_bmg_data(self, bmg_file, robot_file, save_dir=None):
        bmg_file = Path(bmg_file).resolve()
        robot_file = Path(robot_file).resolve()
        
        # Use the specified save directory or default to BMG file directory
        output_dir = Path(save_dir) if save_dir else bmg_file.parent
        
        data, info = read_bmg_xlsx(bmg_file)

        # get concentration vector in M 
        concentration_vector = extract_concentration_vector(robot_file)
        if concentration_vector is None:
            raise ValueError("Failed to extract concentration vector from robot file.")
        data.columns = concentration_vector
        
        self.save_replica_to_multiple_txt(data, output_dir, bmg_file.stem)
        self.merge_replica_to_single_txt(data, output_dir, bmg_file.stem)

    def save_replica_to_multiple_txt(self, data: pd.DataFrame, directory: Path, dir_name='for_mathematica'):
        # save replica to multiple txt files

        directory = directory / dir_name
        directory.mkdir(parents=False, exist_ok=True)
        
        for i, index in enumerate(data.index):
            data.loc[[index]].T.to_csv(directory / f'Replica_{i}.txt', sep='\t', header=False, index=True, mode='w')

    def merge_replica_to_single_txt(self, data: pd.DataFrame, directory: Path, file_name='merged_data.txt'):
        # merge replica to a single txt file (in µM)               
        data.rename(columns=lambda x: f'{x * 1e-6 :.2e}', inplace=True)    # convert to µM and apply sci. notation to concent. values
        output_file = (directory / file_name).with_suffix('.txt')
        with open(output_file, 'w') as f:
            for index in data.index:
                data.loc[[index]].T.to_csv(output_file, sep='\t', header=['signal'], index_label=['var'], mode='a', lineterminator='\n') 

if __name__ == '__main__':
    root = tk.Tk()
    app = BMGToTxtConverter(root)
    root.mainloop()

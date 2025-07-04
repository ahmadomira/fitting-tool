import os
import tkinter as tk
from tkinter import filedialog


class BaseAppGUI:
    """
    Unified base class for all GUI applications in the fitting suite.
    Provides common functionality for variable management, file/directory browsing,
    message display, and window management.
    """

    def __init__(self, root, title="Interface"):
        self.root = root
        self.root.title(title)
        self.info_label = None
        self.pad_x = 10
        self.pad_y = 5
        self.vars = {}

    def add_string_var(self, name, default=""):
        """Create and register a StringVar with default value."""
        var = tk.StringVar()
        var.set(default)
        self.vars[name] = var
        return var

    def add_double_var(self, name, default=0.0):
        """Create and register a DoubleVar with default value."""
        var = tk.DoubleVar()
        var.set(default)
        self.vars[name] = var
        return var

    def add_int_var(self, name, default=0):
        """Create and register an IntVar with default value."""
        var = tk.IntVar()
        var.set(default)
        self.vars[name] = var
        return var

    def add_bool_var(self, name, default=False):
        """Create and register a BooleanVar with default value."""
        var = tk.BooleanVar()
        var.set(default)
        self.vars[name] = var
        return var

    def browse_file(self, entry_var: tk.StringVar, callback=None):
        """Open file dialog and set the selected file path to entry_var."""
        current_path = entry_var.get()
        initialdir = os.path.dirname(current_path) if current_path else os.getcwd()
        file_path = filedialog.askopenfilename(initialdir=initialdir)
        if file_path:
            entry_var.set(file_path)
            if callback:
                callback()

    def browse_save_file(
        self,
        save_var: tk.StringVar,
        input_var: tk.StringVar = None,
        default_ext=".txt",
        filetypes=None,
    ):
        """Open save file dialog and set the selected file path to save_var."""
        current_path = save_var.get()
        initialdir = os.path.dirname(current_path) if current_path else os.getcwd()
        # Set default file name based on input file
        default_name = "results.txt"
        if input_var is not None:
            input_path = input_var.get()
            if input_path:
                input_name = os.path.splitext(os.path.basename(input_path))[0]
                default_name = f"results_{input_name}{default_ext}"
        if filetypes is None:
            filetypes = [("Text files", "*.txt"), ("All files", "*.*")]
        file_path = filedialog.asksaveasfilename(
            initialdir=initialdir,
            initialfile=default_name,
            defaultextension=default_ext,
            filetypes=filetypes,
        )
        if file_path:
            save_var.set(file_path)

    def browse_directory(self, entry_var: tk.StringVar):
        """Open directory dialog and set the selected directory path to entry_var."""
        current_path = entry_var.get()
        initialdir = current_path if current_path else os.getcwd()
        dir_path = filedialog.askdirectory(initialdir=initialdir)
        if dir_path:
            entry_var.set(dir_path)

    def get_default_save_path(self, input_var: tk.StringVar, ext=".txt"):
        """Generate a default save path based on the input file path."""
        input_path = input_var.get()
        if input_path:
            input_dir = os.path.dirname(input_path)
            input_name = os.path.splitext(os.path.basename(input_path))[0]
            default_name = f"results_{input_name}{ext}"
            return os.path.join(input_dir, default_name)
        return ""

    def set_default_save_file_path(
        self, input_var: tk.StringVar, save_var: tk.StringVar, ext: str = ".txt"
    ):
        """
        Set the default save file path for results based on the input file variable.
        """
        default_path = self.get_default_save_path(input_var, ext=ext)
        if default_path:
            save_var.set(default_path)

    def set_default_directory(self, input_var: tk.StringVar, dir_var: tk.StringVar):
        """
        Set the default directory based on the input file variable.
        """
        input_path = input_var.get()
        if input_path:
            input_dir = os.path.dirname(input_path)
            dir_var.set(input_dir)

    def show_message(self, message, is_error=False, row=99):
        """Display a message in the GUI with optional error styling."""
        if self.info_label:
            self.info_label.destroy()
        fg_color = "red" if is_error else "green"
        self.info_label = tk.Label(self.root, text=message, fg=fg_color)
        self.info_label.grid(row=row, column=0, columnspan=3, pady=10)

    def set_widget_state(self, widget, state):
        """Set the state of a widget (NORMAL, DISABLED, etc.)."""
        widget.config(state=state)

    def lift_and_focus(self):
        """Bring the window to the front and give it focus."""
        self.root.lift()
        self.root.focus_force()

    def update_widget_state_and_dir(
        self, bool_var, entry_widget, button_widget, input_var, dir_var
    ):
        """
        Enable or disable entry and button widgets based on a boolean variable.
        If enabled, set the directory variable to the directory of the input file.
        """
        state = tk.NORMAL if bool_var.get() else tk.DISABLED
        entry_widget.config(state=state)
        button_widget.config(state=state)
        if bool_var.get():
            self.set_default_directory(input_var, dir_var)

    def add_toggleable_dir_selector(
        self,
        row,
        label_text,
        bool_var,
        dir_var,
        input_file_var,
        col_offset=0,
        width=40,
        checkbutton_text=None,
    ):
        """
        Create a checkbutton, entry, and browse button for a directory, with enable/disable logic.
        Returns (entry_widget, button_widget)
        """
        if checkbutton_text is None:
            checkbutton_text = label_text
        check = tk.Checkbutton(
            self.root,
            text=checkbutton_text,
            variable=bool_var,
            command=lambda: self.update_widget_state_and_dir(
                bool_var, entry, button, input_file_var, dir_var
            ),
        )
        check.grid(
            row=row, column=col_offset, sticky=tk.W, padx=self.pad_x, pady=self.pad_y
        )
        entry = tk.Entry(
            self.root,
            textvariable=dir_var,
            width=width,
            state=tk.DISABLED,
            justify="left",
        )
        entry.grid(
            row=row,
            column=col_offset + 1,
            padx=self.pad_x,
            pady=self.pad_y,
            sticky=tk.W,
        )
        button = tk.Button(
            self.root,
            text="Browse",
            command=lambda: self.browse_directory(dir_var),
            state=tk.DISABLED,
        )
        button.grid(row=row, column=col_offset + 2, padx=self.pad_x, pady=self.pad_y)
        bool_var.trace_add(
            "write",
            lambda *args: self.update_widget_state_and_dir(
                bool_var, entry, button, input_file_var, dir_var
            ),
        )
        return entry, button

    def add_toggleable_file_selector(
        self,
        row,
        label_text,
        bool_var,
        file_var,
        col_offset=0,
        width=40,
        checkbutton_text=None,
    ):
        """
        Create a checkbutton, entry, and browse button for a file, with enable/disable logic.
        Returns (entry_widget, button_widget)
        """
        if checkbutton_text is None:
            checkbutton_text = label_text
        check = tk.Checkbutton(
            self.root,
            text=checkbutton_text,
            variable=bool_var,
            command=lambda: self.update_widget_state_and_dir(
                bool_var, entry, button, file_var, file_var
            ),
        )
        check.grid(
            row=row, column=col_offset, sticky=tk.W, padx=self.pad_x, pady=self.pad_y
        )
        entry = tk.Entry(
            self.root,
            textvariable=file_var,
            width=width,
            justify="left",
            state=tk.DISABLED,
        )
        entry.grid(
            row=row,
            column=col_offset + 1,
            padx=self.pad_x,
            pady=self.pad_y,
            sticky=tk.W,
        )
        button = tk.Button(
            self.root,
            text="Browse",
            command=lambda: self.browse_file(file_var),
            state=tk.DISABLED,
        )
        button.grid(row=row, column=col_offset + 2, padx=self.pad_x, pady=self.pad_y)
        bool_var.trace_add(
            "write",
            lambda *args: self.update_widget_state_and_dir(
                bool_var, entry, button, file_var, file_var
            ),
        )
        return entry, button

    def add_labeled_entry(
        self,
        row,
        label_text,
        var,
        col_offset=0,
        width=40,
        justify="left",
        label_kwargs=None,
        entry_kwargs=None,
    ):
        """
        Create a label and entry widget in a row. Returns the entry widget.
        """
        if label_kwargs is None:
            label_kwargs = {}
        if entry_kwargs is None:
            entry_kwargs = {}
        label = tk.Label(self.root, text=label_text, **label_kwargs)
        label.grid(
            row=row, column=col_offset, sticky=tk.W, padx=self.pad_x, pady=self.pad_y
        )
        entry = tk.Entry(
            self.root, textvariable=var, width=width, justify=justify, **entry_kwargs
        )
        entry.grid(
            row=row,
            column=col_offset + 1,
            padx=self.pad_x,
            pady=self.pad_y,
            sticky=tk.W,
        )
        return entry

    def add_file_selector(self, row, label_text, var, width=40, **kwargs):
        """Add a label, entry, and browse button for file selection."""
        tk.Label(self.root, text=label_text).grid(
            row=row, column=0, sticky=tk.W, padx=self.pad_x, pady=self.pad_y
        )
        entry = tk.Entry(self.root, textvariable=var, width=width, justify="left")
        entry.grid(row=row, column=1, padx=self.pad_x, pady=self.pad_y, sticky=tk.W)
        button = tk.Button(
            self.root,
            text="Browse",
            command=lambda: self.browse_file(var),
        )
        button.grid(row=row, column=2, padx=self.pad_x, pady=self.pad_y)
        return entry, button

    def add_directory_selector(self, row, label_text, var, width=40, **kwargs):
        """Add a label, entry, and browse button for directory selection."""
        tk.Label(self.root, text=label_text).grid(
            row=row, column=0, sticky=tk.W, padx=self.pad_x, pady=self.pad_y
        )
        entry = tk.Entry(self.root, textvariable=var, width=width, justify="left")
        entry.grid(row=row, column=1, padx=self.pad_x, pady=self.pad_y, sticky=tk.W)
        button = tk.Button(
            self.root,
            text="Browse",
            command=lambda: self.browse_directory(var),
        )
        button.grid(row=row, column=2, padx=self.pad_x, pady=self.pad_y)
        return entry, button

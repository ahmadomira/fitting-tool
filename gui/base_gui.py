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

    def browse_file(self, entry_var):
        """Open file dialog and set the selected file path to entry_var."""
        file_path = filedialog.askopenfilename()
        if file_path:
            entry_var.set(file_path)

    def browse_directory(self, entry_var):
        """Open directory dialog and set the selected directory path to entry_var."""
        directory_path = filedialog.askdirectory()
        if directory_path:
            entry_var.set(directory_path)

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

import tkinter as tk

import matplotlib.pyplot as plt


class ProgressWindow:
    """
    A centralized progress window with an optional button to close all matplotlib figures.
    The button is automatically enabled/disabled based on whether there are open figures.
    """

    def __init__(
        self,
        parent,
        title="Operation in Progress",
        message="Operation in progress, please wait...",
    ):
        """
        Initialize the progress window.

        Args:
            parent: The parent tkinter window
            title: Window title (default: "Operation in Progress")
            message: Progress message to display (default: "Operation in progress, please wait...")
        """
        self.parent = parent
        self.window = tk.Toplevel(parent)
        self.window.title(title)
        self.window.resizable(False, False)

        # Make window stay always on top of parent
        self.window.transient(parent)
        self.window.attributes("-topmost", True)

        # Ensure window stays focused with parent
        self.window.grab_set()

        # Create and pack the progress message with minimal padding
        self.progress_label = tk.Label(self.window, text=message, wraplength=200)
        self.progress_label.pack(padx=8, pady=8)

        # Create the close figures button with minimal padding and compact styling
        self.close_figures_button = tk.Button(
            self.window,
            text="Close All Plot Figures",
            command=self._close_all_figures,
            state="disabled",
            width=18,  # Fixed width to make it compact
        )
        self.close_figures_button.pack(padx=8, pady=(0, 8))

        # Update widgets to calculate proper size, then center the window
        self.window.update_idletasks()
        self._center_on_parent()

        # Start periodic updates for button state
        self._update_in_progress = True
        self._periodic_update()

    def _center_on_parent(self):
        """Center the progress window on its parent window."""
        # Force update to get accurate window dimensions
        self.window.update_idletasks()

        # Get parent window position and size
        parent_x = self.parent.winfo_x()
        parent_y = self.parent.winfo_y()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()

        # Get actual progress window size after widgets are added
        window_width = self.window.winfo_reqwidth()
        window_height = self.window.winfo_reqheight()

        # Ensure minimum reasonable size but keep it compact
        min_width = max(window_width, 220)
        min_height = max(window_height, 90)

        # Calculate center position
        x = parent_x + (parent_width // 2) - (min_width // 2)
        y = parent_y + (parent_height // 2) - (min_height // 2)

        # Set the geometry with actual calculated size
        self.window.geometry(f"{min_width}x{min_height}+{x}+{y}")

        # Force another update to ensure proper display
        self.window.update()

    def _close_all_figures(self):
        """Close all matplotlib figures and update button state."""
        plt.close("all")
        self._update_close_button_state()

    def _update_close_button_state(self):
        """Update the close figures button state based on open figures."""
        if plt.get_fignums():
            self.close_figures_button.config(state="normal")
        else:
            self.close_figures_button.config(state="disabled")

    def _periodic_update(self):
        """Periodically update the button state while the window exists."""
        if self._update_in_progress:
            try:
                if self.window.winfo_exists():
                    self._update_close_button_state()
                    self.window.after(
                        1000, self._periodic_update
                    )  # Update every second
                else:
                    self._update_in_progress = False
            except tk.TclError:
                # Window was destroyed
                self._update_in_progress = False

    def update_message(self, message):
        """
        Update the progress message.

        Args:
            message: New message to display
        """
        try:
            if self.window.winfo_exists():
                self.progress_label.config(text=message)
                self.parent.update_idletasks()
        except tk.TclError:
            # Window was destroyed
            pass

    def destroy(self):
        """Destroy the progress window and stop updates."""
        self._update_in_progress = False
        try:
            if self.window.winfo_exists():
                # Release the grab before destroying
                self.window.grab_release()
                self.window.destroy()
        except tk.TclError:
            # Window was already destroyed
            pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically destroy the window."""
        self.destroy()

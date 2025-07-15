import json
from pathlib import Path
import pandas as pd

from ...registry import register_writer
from ...base import Writer
from ...measurement_set import MeasurementSet


@register_writer("mset", ".txt")
class MsetTxtWriter(Writer):
    
    def write(self, obj: MeasurementSet, path: Path):
        """
        Save all replica rows of a MeasurementSet to a human‑readable TXT file.

        Layout:
          # <<<meta>>>
          { JSON metadata }
          # <<<data>>>
          var<TAB>signal
          0       1445
          1.5E-6  2019
          ...
          # <<<data>>>
          ...

        Requires a complete 'concentration' coordinate; call `set_concentration()` first if needed.
        """
        # ---- prerequisite checks -----------------------------------------
        if not obj.has_concentration(complete=True):
            raise ValueError(
                "MeasurementSet lacks a complete 'concentration' coordinate; "
                "use set_concentration() and provide concentration values before saving to TXT."
            )

        meta_json = json.dumps(obj.meta, indent=2)

        with Path(path).expanduser().open("w", encoding="utf-8") as f:
            # ---- write metadata section ----------------------------------
            f.write("# <<<meta>>>\n")
            f.write(meta_json + "\n")

            # ---- write each replica row ---------------------------------
            for row_label in obj.ds.coords["well_row"].values:
                conc = obj.concentration(well_row=row_label).values
                sig  = obj.row(row_label).values

                df = pd.DataFrame({"var": conc, "signal": sig})

                f.write("# <<<data>>>\n")
                # to_csv adds header by default; index=False for cleanliness
                df.to_csv(f, sep="\t", index=False, lineterminator="\n")
import json
from io import StringIO
from pathlib import Path
import pandas as pd

from ...base import Reader
from ...registry import register_reader

@register_reader(".txt")
class TxtReader(Reader):
    """
    Parse the custom TXT format written by MsetTxtWriter.

    Expected layout::
        # <<<meta>>>
        { ...JSON... }
        # <<<data>>>
        var<TAB>signal
        ...
        # <<<data>>>
        ...

    Returns
    -------
    Tuple[str, pd.DataFrame]
        ("mset", tidy_dataframe_with_meta_attr)
    (Method name aligned with Reader ABC: read())
    """

    def read(self, path: Path):
        with Path(path).expanduser().open("r", encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f]

        # ------------------------------------------------------------------
        # 1. Extract metadata block
        # ------------------------------------------------------------------
        try:
            meta_start = lines.index("# <<<meta>>>") + 1
        except ValueError as err:
            raise ValueError("TXT file missing '# <<<meta>>>' marker.") from err

        # meta ends at first '# <<<data>>>' or EOF
        try:
            meta_end = lines.index("# <<<data>>>", meta_start)
        except ValueError:
            raise ValueError("TXT file missing '# <<<data>>>' marker.") from err

        meta_json = "\n".join(lines[meta_start:meta_end]).strip()
        if not meta_json:
            raise ValueError("Metadata block is empty.")
        meta = json.loads(meta_json)

        # ------------------------------------------------------------------
        # 2. Parse each data block
        # ------------------------------------------------------------------
        data_blocks = []
        row_labels  = []
        block_start = meta_end
        expected_len = None  # will hold number of columns expected per row
        while True:
            try:
                block_start = lines.index("# <<<data>>>", block_start) + 1
            except ValueError:
                break  # no more blocks

            # find next marker or EOF
            next_marker = next((i for i in range(block_start, len(lines))
                                if lines[i].startswith("# <<<data>>>")), len(lines))
            block_lines = "\n".join(lines[block_start:next_marker]).strip()

            if not block_lines:
                block_start = next_marker
                continue  # empty block, skip

            df_block = pd.read_csv(
                StringIO(block_lines),
                sep="\t",
                engine="python"
            )

            # ensure numeric dtypes (object → float64)
            df_block = df_block.apply(pd.to_numeric, errors="coerce")

            # check consistent number of concentration points
            if expected_len is None:
                expected_len = len(df_block)
            elif len(df_block) != expected_len:
                raise ValueError(
                    f"Inconsistent point count: "
                    f"expected {expected_len}, got {len(df_block)} in row {next_row}"
                )

            # assign well_row label sequentially A–H
            next_row = chr(ord("A") + len(row_labels))
            row_labels.append(next_row)

            df_block["well_row"] = next_row
            df_block["well_col"] = range(1, len(df_block) + 1)

            data_blocks.append(df_block)
            block_start = next_marker

        if not data_blocks:
            raise ValueError("No data blocks found in TXT file.")

        tidy = pd.concat(data_blocks, ignore_index=True)

        # standard columns
        tidy = tidy.rename(columns={"var": "concentration", "signal": "signal"})
        tidy["channel"] = "FI"
        tidy["time_s"]  = 0.0

        # Reorder columns to MeasurementSet expectation
        tidy = tidy[
            ["well_row", "well_col", "channel", "time_s",
             "concentration", "signal"]
        ]

        # embed meta for dispatcher
        tidy.attrs["attrs"] = json.dumps(meta)

        # object type lives in meta → returned tag
        tag = meta.get("object_type", "mset")
        return tag, tidy
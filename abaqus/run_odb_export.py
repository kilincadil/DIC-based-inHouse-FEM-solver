# -*- coding: utf-8 -*-
"""
Launch the ODB export inside Abaqus' Python 2.7 interpreter from Python 3.

``odb_to_grid.py`` imports ``odbAccess``, which only exists inside Abaqus. This
wrapper runs it via ``abaqus python`` as a subprocess and streams its output
back, so the export can be driven from a normal Python 3 session or IDE without
opening Abaqus/CAE.

Run it either way:

    python run_odb_export.py                        # uses the CONFIG block below
    python run_odb_export.py --odb job.odb --output-dir out
    python run_odb_export.py --odb "runs/*.odb" --output-dir out   # batch

Part of: DIC-based in-house FEM (https://github.com/kilincadil/DIC-based-inHouse-FEM)
"""

import argparse
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------- #
# CONFIG - edit these, or override any of them from the command line.
# --------------------------------------------------------------------------- #
CONFIG = {
    # Abaqus launcher. "abaqus" alone works if it is on PATH; otherwise give the
    # full path, e.g. r"C:\SIMULIA\Commands\abaqus.bat" on Windows.
    "abaqus_cmd": "abaqus",

    # ODB to export. Accepts a glob pattern to process several in one go.
    "odb": "runs/job.odb",

    "output_dir": "export",

    "step": "Step-1",
    "instance": "PART-1-1",

    # Grid pitch, i.e. scale_factor * element_size used when writing the mesh.
    "m_per_pixel": 0.00184,
}

# The Abaqus-side script, resolved next to this file.
EXPORT_SCRIPT = os.path.join(HERE, "odb_to_grid.py")


def export_odb(cfg, odb_path):
    """Run the Abaqus-side export for one ODB, streaming its output."""
    if not os.path.isfile(EXPORT_SCRIPT):
        raise IOError("export script not found: {0}".format(EXPORT_SCRIPT))

    if not os.path.isfile(odb_path):
        raise IOError("ODB not found: {0}".format(odb_path))

    if not os.path.isdir(cfg["output_dir"]):
        os.makedirs(cfg["output_dir"])

    cmd = [cfg["abaqus_cmd"], "python", EXPORT_SCRIPT,
           odb_path, cfg["output_dir"], cfg["step"], cfg["instance"],
           str(cfg["m_per_pixel"])]

    print("Running: {0}".format(" ".join(cmd)))

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               universal_newlines=True)

    # Stream line by line rather than buffering, so long exports show progress.
    for line in iter(process.stdout.readline, ""):
        if not line:
            break
        print(line.rstrip())

    process.stdout.close()
    code = process.wait()

    if code != 0:
        raise RuntimeError("abaqus python exited with code {0}".format(code))


def parse_args():
    """Overlay command-line flags onto CONFIG and return the merged dict."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])

    for key, default in CONFIG.items():
        parser.add_argument("--" + key.replace("_", "-"), dest=key,
                            type=type(default), default=default)

    return vars(parser.parse_args())


if __name__ == "__main__":
    config = parse_args()

    targets = sorted(glob.glob(config["odb"])) or [config["odb"]]

    for n, target in enumerate(targets, start=1):
        print("\n[{0}/{1}] {2}".format(n, len(targets), target))
        try:
            export_odb(config, target)
        except (IOError, RuntimeError) as exc:
            # Keep going through a batch rather than losing the whole run to
            # one bad ODB; report the failures together at the end.
            print("FAILED: {0}".format(exc), file=sys.stderr)

    print("\nExport written to: {0}".format(config["output_dir"]))

"""Export functions for process models and event logs."""

from pathlib import Path

import pm4py
from pm4py.objects.bpmn.obj import BPMN
from pm4py.objects.log.obj import EventLog, Trace as PM4PyTrace

from stream_mining.io_xes import Log


def export_xes(log: Log, out_path: str) -> None:
    """Export Pydantic Log model to XES file.

    Converts a Pydantic Log model to PM4Py EventLog format and writes it
    to an XES file. This is a convenience wrapper around the conversion
    and export process.

    Args:
        log: Pydantic Log object to export.
        out_path: Output file path for the XES file.

    Raises:
        ValueError: If conversion or file writing fails.
        IOError: If the output path cannot be written to.
    """
    try:
        # Convert Pydantic Log to PM4Py EventLog
        event_log = EventLog()
        event_log.attributes["concept:name"] = "Exported Log"

        for trace in log.traces:
            pm4py_trace = PM4PyTrace()
            pm4py_trace.attributes["concept:name"] = trace.case_id

            for event in trace.events:
                pm4py_event = {}
                pm4py_event["concept:name"] = event.activity
                pm4py_event["time:timestamp"] = event.timestamp

                if event.resource:
                    pm4py_event["org:resource"] = event.resource

                pm4py_trace.append(pm4py_event)

            event_log.append(pm4py_trace)

        # Ensure output directory exists
        out_path_obj = Path(out_path)
        out_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Write XES file
        pm4py.write_xes(event_log, str(out_path))

    except Exception as e:
        raise ValueError(f"Failed to export XES file to {out_path}: {e}") from e


def export_bpmn_png(
    model: BPMN,
    out_path: str,
) -> None:
    """Export BPMN model to PNG image.

    Saves a BPMN model as a PNG image file.
    Handles graphviz/dot dependencies gracefully for Colab compatibility.

    Args:
        model: BPMN model object to export.
        out_path: Output file path for the PNG image.

    Raises:
        ValueError: If conversion or file writing fails.
        ImportError: If required dependencies (graphviz, pydot) are not available.
    """
    try:
        # Ensure output directory exists
        out_path_obj = Path(out_path)
        out_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Save BPMN model as PNG
        pm4py.save_vis_bpmn(model, out_path)

    except ImportError as e:
        raise ImportError(
            f"Required dependencies for BPMN export not available: {e}. "
            "Please install graphviz and pydot: pip install graphviz pydot"
        ) from e
    except Exception as e:
        raise ValueError(f"Failed to export BPMN PNG to {out_path}: {e}") from e


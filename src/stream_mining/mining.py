from typing import Optional

import pm4py
from pm4py.objects.bpmn.obj import BPMN
from pm4py.objects.log.obj import EventLog


def discover_heuristics(
    event_log: EventLog,
    dependency_thresh: float = 0.7,
) -> Optional[BPMN]:
    """Discover BPMN process model using Inductive Miner algorithm.

    Uses PM4Py's inductive miner to discover a BPMN model directly from an event log.
    The algorithm identifies process patterns and constructs a BPMN model
    representing the discovered process flow.

    Args:
        event_log: PM4Py EventLog containing traces and events.
        dependency_thresh: Dependency threshold (kept for API compatibility, 
                          but Inductive Miner uses its own parameters).

    Returns:
        BPMN model object if discovery succeeds.
        Returns None if the event log is empty or contains insufficient data.

    Raises:
        ValueError: If event log is empty or contains no valid traces.
    """
    if not event_log or len(event_log) == 0:
        raise ValueError("Cannot discover process model from empty event log")

    # Check if log has any events
    total_events = sum(len(trace) for trace in event_log)
    if total_events == 0:
        raise ValueError("Event log contains no events")

    try:
        # Discover BPMN model using inductive miner
        bpmn_model = pm4py.discover_bpmn_inductive(event_log)

        return bpmn_model

    except Exception as e:
        # Handle cases where discovery fails (e.g., insufficient data)
        raise ValueError(f"Failed to discover BPMN process model: {e}") from e


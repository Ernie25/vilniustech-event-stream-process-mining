from typing import Optional

import pm4py
from pm4py.objects.bpmn.obj import BPMN
from pm4py.objects.log.obj import EventLog


def discover_heuristics(
    event_log: EventLog,
    dependency_thresh: float = 0.7,
) -> Optional[BPMN]:
    if not event_log or len(event_log) == 0:
        raise ValueError("Cannot discover process model from empty event log")

    # Check if log has any events
    total_events = sum(len(trace) for trace in event_log)
    if total_events == 0:
        raise ValueError("Event log contains no events")

    try:
        net, initial_marking, final_marking = pm4py.discover_petri_net_heuristics(
            event_log,
            dependency_threshold=dependency_thresh,
        )

        try:
            bpmn_model = pm4py.convert_petri_net_to_bpmn(net, initial_marking, final_marking)
        except (AttributeError, TypeError):
            try:
                from pm4py.objects.conversion.bpmn import converter as bpmn_converter
                bpmn_model = bpmn_converter.apply(net, initial_marking, final_marking)
            except (ImportError, AttributeError):
                process_tree = pm4py.convert_to_process_tree(net, initial_marking, final_marking)
                bpmn_model = pm4py.convert_to_bpmn(process_tree)

        return bpmn_model

    except Exception as e:
        raise ValueError(f"Failed to discover BPMN process model using Heuristics Miner: {e}") from e


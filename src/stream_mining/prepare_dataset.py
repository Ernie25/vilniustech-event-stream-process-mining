from collections import defaultdict
from datetime import datetime
from typing import Iterator

import pandas as pd
from pm4py.objects.log.obj import EventLog, Trace as PM4PyTrace

from stream_mining.decay_list import DecayList
from stream_mining.io_xes import Event
from stream_mining.sliding_window import SlidingWindow


def build_directly_follows(events_iter: Iterator[Event]) -> pd.DataFrame:
    case_events: dict[str, list[Event]] = defaultdict(list)

    for event in events_iter:
        case_events[event.case_id].append(event)

    dfg_counts: dict[tuple[str, str], int] = defaultdict(int)

    for case_id, events in case_events.items():
        sorted_events = sorted(events, key=lambda e: e.timestamp)

        for i in range(len(sorted_events) - 1):
            source = sorted_events[i].activity
            target = sorted_events[i + 1].activity
            dfg_counts[(source, target)] += 1

    if not dfg_counts:
        return pd.DataFrame(columns=["source", "target", "count"])

    rows = [
        {"source": source, "target": target, "count": count}
        for (source, target), count in dfg_counts.items()
    ]

    return pd.DataFrame(rows)


def prepare_for_mining(
    window: SlidingWindow,
    decay: DecayList,
    now: datetime,
) -> EventLog:
    window_events: list[Event] = list(window.iter_events())

    if not window_events:
        return EventLog()

    window_case_ids: set[str] = {event.case_id for event in window_events}

    decay_case_ids: set[str] = set()
    for event, _ in decay._events:
        case_id = event.case_id
        if case_id not in window_case_ids:
            decay_case_ids.add(case_id)

    decay_events: list[Event] = []
    for case_id in decay_case_ids:
        case_events = decay.pull(case_id)
        if case_events:
            decay_events.extend(case_events)

    all_events = window_events + decay_events

    case_events_dict: dict[str, list[Event]] = defaultdict(list)
    for event in all_events:
        case_events_dict[event.case_id].append(event)

    try:
        event_log = EventLog()
        event_log.attributes["concept:name"] = "Stream Mining Log"

        for case_id, events in case_events_dict.items():
            # Sort events by timestamp within each case
            sorted_events = sorted(events, key=lambda e: e.timestamp)

            pm4py_trace = PM4PyTrace()
            pm4py_trace.attributes["concept:name"] = case_id

            for event in sorted_events:
                pm4py_event = {}
                pm4py_event["concept:name"] = event.activity
                pm4py_event["time:timestamp"] = event.timestamp

                if event.resource:
                    pm4py_event["org:resource"] = event.resource

                pm4py_trace.append(pm4py_event)

            event_log.append(pm4py_trace)

        return event_log

    except Exception as e:
        raise ValueError(f"Failed to convert events to PM4Py EventLog: {e}") from e


from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pm4py
from pydantic import BaseModel, Field, field_validator
from pytz import UTC


class Event(BaseModel):
    """
    Attributes:
        case_id: Unique identifier for the case/trace this event belongs to.
        activity: Name of the activity performed.
        timestamp: When the event occurred (timezone-aware).
        resource: Optional resource (person/system) that performed the activity.
    """

    case_id: str = Field(..., description="Case identifier")
    activity: str = Field(..., description="Activity name")
    timestamp: datetime = Field(..., description="Event timestamp (timezone-aware)")
    resource: Optional[str] = Field(None, description="Resource that performed the activity")

    @field_validator("timestamp", mode="before")
    @classmethod
    def ensure_timezone_aware(cls, v: datetime) -> datetime:
        """Ensure timestamp is timezone-aware."""
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=UTC)
            return v
        return v


class Trace(BaseModel):
    """Represents a trace (case) containing multiple events.

    Attributes:
        case_id: Unique identifier for this trace.
        events: List of events in chronological order.
    """

    case_id: str = Field(..., description="Case identifier")
    events: List[Event] = Field(default_factory=list, description="Events in this trace")


class Log(BaseModel):
    """Represents an event log containing multiple traces.

    Attributes:
        traces: List of traces in the log.
    """

    traces: List[Trace] = Field(default_factory=list, description="Traces in the log")


def read_xes(path: str) -> Log:
    """Read XES file and convert to Pydantic Log model.

    Args:
        path: Path to XES file.

    Returns:
        Log object containing all traces and events.

    Raises:
        FileNotFoundError: If the XES file doesn't exist.
        ValueError: If the XES file cannot be parsed.
    """
    xes_path = Path(path)
    if not xes_path.exists():
        raise FileNotFoundError(f"XES file not found: {path}")

    try:
        event_log = pm4py.read_xes(str(xes_path))
        df = pm4py.convert_to_dataframe(event_log)

        traces = []
        for case_id, group in df.groupby("case:concept:name"):
            events = []
            for _, row in group.iterrows():
                timestamp = row.get("time:timestamp")
                if timestamp is None:
                    continue

                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                elif isinstance(timestamp, datetime):
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=UTC)

                activity = row.get("concept:name", "")
                if not activity:
                    continue

                resource = row.get("org:resource")
                if pd.isna(resource):
                    resource = None
                elif isinstance(resource, str):
                    resource = resource if resource else None
                else:
                    resource = None

                event = Event(
                    case_id=str(case_id),
                    activity=str(activity),
                    timestamp=timestamp,
                    resource=resource,
                )
                events.append(event)

            if events:
                trace = Trace(case_id=str(case_id), events=events)
                traces.append(trace)

        return Log(traces=traces)

    except Exception as e:
        raise ValueError(f"Failed to parse XES file {path}: {e}") from e


def to_xes(log: Log, path: str) -> None:
    """Convert Pydantic Log model to XES file.

    Args:
        log: Log object to convert.
        path: Output path for XES file.

    Raises:
        ValueError: If conversion fails.
    """
    try:
        from pm4py.objects.log.obj import EventLog, Trace as PM4PyTrace

        event_log = EventLog()
        event_log.attributes["concept:name"] = "Imported Log"

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

        pm4py.write_xes(event_log, path)

    except Exception as e:
        raise ValueError(f"Failed to write XES file {path}: {e}") from e
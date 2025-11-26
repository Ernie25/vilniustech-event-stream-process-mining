from datetime import datetime
from pathlib import Path

import pytest
from pytz import UTC

from stream_mining.io_xes import Event, Log, Trace, read_xes, to_xes


def test_event_model():
    """Test Event Pydantic model."""
    # Test with timezone-aware timestamp
    event1 = Event(
        case_id="case1",
        activity="start",
        timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
        resource="user1",
    )
    assert event1.case_id == "case1"
    assert event1.activity == "start"
    assert event1.resource == "user1"
    assert event1.timestamp.tzinfo is not None

    # Test with timezone-naive timestamp (should be converted to UTC)
    event2 = Event(
        case_id="case2",
        activity="end",
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
    )
    assert event2.timestamp.tzinfo is not None

    # Test with optional resource
    event3 = Event(
        case_id="case3",
        activity="process",
        timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
    )
    assert event3.resource is None


def test_trace_model():
    events = [
        Event(
            case_id="case1",
            activity="start",
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
        ),
        Event(
            case_id="case1",
            activity="end",
            timestamp=datetime(2024, 1, 1, 13, 0, 0, tzinfo=UTC),
        ),
    ]
    trace = Trace(case_id="case1", events=events)
    assert trace.case_id == "case1"
    assert len(trace.events) == 2


def test_log_model():
    """Test Log Pydantic model."""
    trace1 = Trace(
        case_id="case1",
        events=[
            Event(
                case_id="case1",
                activity="start",
                timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
            )
        ],
    )
    trace2 = Trace(
        case_id="case2",
        events=[
            Event(
                case_id="case2",
                activity="start",
                timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
            )
        ],
    )
    log = Log(traces=[trace1, trace2])
    assert len(log.traces) == 2


def test_read_xes_file_not_found():
    """Test read_xes raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError):
        read_xes("nonexistent.xes")


def test_round_trip_xes(tmp_path: Path):
    """Test round-trip: read XES, write XES, read again."""
    # Create a sample XES file
    sample_xes = tmp_path / "sample.xes"
    create_sample_xes(sample_xes)

    # Read the XES file
    log1 = read_xes(str(sample_xes))

    # Verify we got data
    assert len(log1.traces) > 0
    total_events = sum(len(trace.events) for trace in log1.traces)
    assert total_events > 0

    # Write to a new file
    output_xes = tmp_path / "output.xes"
    to_xes(log1, str(output_xes))

    # Read the output file
    log2 = read_xes(str(output_xes))

    # Verify round-trip preserved data
    assert len(log2.traces) == len(log1.traces)
    assert sum(len(trace.events) for trace in log2.traces) == sum(
        len(trace.events) for trace in log1.traces
    )

    # Verify timestamps are preserved (within reasonable precision)
    for trace1, trace2 in zip(log1.traces, log2.traces):
        assert trace1.case_id == trace2.case_id
        for event1, event2 in zip(trace1.events, trace2.events):
            assert event1.activity == event2.activity
            assert event1.timestamp == event2.timestamp
            assert event1.resource == event2.resource


def test_timezone_handling(tmp_path: Path):
    """Test that timestamps are timezone-aware after reading."""
    sample_xes = tmp_path / "sample.xes"
    create_sample_xes(sample_xes)

    log = read_xes(str(sample_xes))

    # All timestamps should be timezone-aware
    for trace in log.traces:
        for event in trace.events:
            assert event.timestamp.tzinfo is not None, f"Event {event.activity} has no timezone"


def test_missing_optional_attributes(tmp_path: Path):
    """Test handling of missing optional attributes."""
    sample_xes = tmp_path / "sample.xes"
    create_sample_xes(sample_xes)

    log = read_xes(str(sample_xes))

    # Should handle missing resources gracefully
    for trace in log.traces:
        for event in trace.events:
            # Resource can be None, that's fine
            assert isinstance(event.resource, (str, type(None)))


def create_sample_xes(path: Path) -> None:
    """Create a minimal sample XES file for testing."""
    import pm4py
    from pm4py.objects.log.obj import EventLog, Trace as PM4PyTrace

    event_log = EventLog()
    event_log.attributes["concept:name"] = "Sample Log"

    # Create 3 traces with 2-4 events each
    traces_data = [
        {
            "case_id": "case1",
            "events": [
                ("start", datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC), "user1"),
                ("process", datetime(2024, 1, 1, 10, 5, 0, tzinfo=UTC), "user1"),
                ("end", datetime(2024, 1, 1, 10, 10, 0, tzinfo=UTC), None),
            ],
        },
        {
            "case_id": "case2",
            "events": [
                ("start", datetime(2024, 1, 1, 11, 0, 0, tzinfo=UTC), "user2"),
                ("end", datetime(2024, 1, 1, 11, 5, 0, tzinfo=UTC), "user2"),
            ],
        },
        {
            "case_id": "case3",
            "events": [
                ("start", datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC), None),
                ("process", datetime(2024, 1, 1, 12, 5, 0, tzinfo=UTC), "user3"),
                ("review", datetime(2024, 1, 1, 12, 10, 0, tzinfo=UTC), "user3"),
                ("end", datetime(2024, 1, 1, 12, 15, 0, tzinfo=UTC), None),
            ],
        },
    ]

    for trace_data in traces_data:
        pm4py_trace = PM4PyTrace()
        pm4py_trace.attributes["concept:name"] = trace_data["case_id"]

        for activity, timestamp, resource in trace_data["events"]:
            pm4py_event = {}
            pm4py_event["concept:name"] = activity
            pm4py_event["time:timestamp"] = timestamp
            if resource:
                pm4py_event["org:resource"] = resource
            pm4py_trace.append(pm4py_event)

        event_log.append(pm4py_trace)

    pm4py.write_xes(event_log, str(path))


"""Tests for dataset preparation functions."""

from collections import deque
from datetime import datetime, timedelta

import pandas as pd
import pytest
from pm4py.objects.log.obj import EventLog
from pytz import UTC

from stream_mining.decay_list import DecayList
from stream_mining.io_xes import Event
from stream_mining.prepare_dataset import build_directly_follows, prepare_for_mining
from stream_mining.sliding_window import SlidingWindowCount, SlidingWindowTime


def create_event(case_id: str, activity: str, timestamp: datetime, resource: str | None = None) -> Event:
    """Helper to create an Event."""
    return Event(
        case_id=case_id,
        activity=activity,
        timestamp=timestamp,
        resource=resource,
    )


class TestBuildDirectlyFollows:
    """Tests for build_directly_follows function."""

    def test_empty_iterator(self):
        """Test DFG construction with empty iterator."""
        dfg = build_directly_follows(iter([]))
        assert isinstance(dfg, pd.DataFrame)
        assert len(dfg) == 0
        assert list(dfg.columns) == ["source", "target", "count"]

    def test_single_case_simple(self):
        """Test DFG construction with single case."""
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)
        events = [
            create_event("case1", "start", base_time),
            create_event("case1", "process", base_time + timedelta(minutes=10)),
            create_event("case1", "end", base_time + timedelta(minutes=20)),
        ]

        dfg = build_directly_follows(iter(events))

        assert len(dfg) == 2
        assert set(dfg.columns) == {"source", "target", "count"}

        # Check directly-follows pairs
        dfg_dict = {(row["source"], row["target"]): row["count"] for _, row in dfg.iterrows()}
        assert dfg_dict[("start", "process")] == 1
        assert dfg_dict[("process", "end")] == 1

    def test_multiple_cases_no_cross_case_edges(self):
        """Test that DFG doesn't create cross-case edges."""
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)
        events = [
            create_event("case1", "start", base_time),
            create_event("case1", "end", base_time + timedelta(minutes=10)),
            create_event("case2", "start", base_time + timedelta(minutes=20)),
            create_event("case2", "end", base_time + timedelta(minutes=30)),
        ]

        dfg = build_directly_follows(iter(events))

        # DFG aggregates transitions - both cases have start->end, so count should be 2
        assert len(dfg) == 1
        dfg_dict = {(row["source"], row["target"]): row["count"] for _, row in dfg.iterrows()}
        # Should only have within-case edges (aggregated)
        assert ("start", "end") in dfg_dict
        assert dfg_dict[("start", "end")] == 2  # Both cases have start->end
        # Verify no cross-case edges (case1.end -> case2.start should not exist)
        assert ("end", "start") not in dfg_dict

    def test_same_activity_consecutive(self):
        """Test DFG with consecutive same activities."""
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)
        events = [
            create_event("case1", "process", base_time),
            create_event("case1", "process", base_time + timedelta(minutes=10)),
            create_event("case1", "end", base_time + timedelta(minutes=20)),
        ]

        dfg = build_directly_follows(iter(events))

        dfg_dict = {(row["source"], row["target"]): row["count"] for _, row in dfg.iterrows()}
        assert dfg_dict[("process", "process")] == 1
        assert dfg_dict[("process", "end")] == 1

    def test_unsorted_events_sorted_by_timestamp(self):
        """Test that events are sorted by timestamp within each case."""
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)
        # Provide events out of order
        events = [
            create_event("case1", "end", base_time + timedelta(minutes=20)),
            create_event("case1", "start", base_time),
            create_event("case1", "process", base_time + timedelta(minutes=10)),
        ]

        dfg = build_directly_follows(iter(events))

        dfg_dict = {(row["source"], row["target"]): row["count"] for _, row in dfg.iterrows()}
        # Should be start->process->end, not end->start->process
        assert ("start", "process") in dfg_dict
        assert ("process", "end") in dfg_dict
        assert ("end", "start") not in dfg_dict


class TestPrepareForMining:
    """Tests for prepare_for_mining function."""

    def test_empty_window_and_decay(self):
        """Test with empty window and decay list."""
        window = SlidingWindowCount(max_events=10)
        decay = DecayList(half_life=timedelta(hours=1))
        now = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)

        event_log = prepare_for_mining(window, decay, now)

        assert isinstance(event_log, EventLog)
        assert len(event_log) == 0

    def test_window_only(self):
        """Test with events only in window."""
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)
        window = SlidingWindowCount(max_events=10)
        decay = DecayList(half_life=timedelta(hours=1))

        events = [
            create_event("case1", "start", base_time),
            create_event("case1", "end", base_time + timedelta(minutes=10)),
            create_event("case2", "start", base_time + timedelta(minutes=20)),
        ]

        for event in events:
            window.push(event)

        event_log = prepare_for_mining(window, decay, base_time + timedelta(hours=1))

        assert len(event_log) == 2
        assert len(event_log[0]) == 2  # case1 has 2 events
        assert len(event_log[1]) == 1  # case2 has 1 event

        # Check case IDs
        case_ids = {trace.attributes["concept:name"] for trace in event_log}
        assert case_ids == {"case1", "case2"}

    def test_decay_only(self):
        """Test with events only in decay list."""
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)
        window = SlidingWindowCount(max_events=10)
        decay = DecayList(half_life=timedelta(hours=1))

        events = [
            create_event("case1", "start", base_time),
            create_event("case1", "end", base_time + timedelta(minutes=10)),
            create_event("case2", "start", base_time + timedelta(minutes=20)),
        ]

        for event in events:
            decay.update(event)

        event_log = prepare_for_mining(window, decay, base_time + timedelta(hours=1))

        assert len(event_log) == 2
        assert len(event_log[0]) == 2  # case1 has 2 events
        assert len(event_log[1]) == 1  # case2 has 1 event

    def test_window_and_decay_merged(self):
        """Test merging events from both window and decay."""
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)
        window = SlidingWindowCount(max_events=10)
        decay = DecayList(half_life=timedelta(hours=1))

        # Add events to window
        window.push(create_event("case1", "start", base_time))
        window.push(create_event("case2", "process", base_time + timedelta(minutes=10)))

        # Add events to decay
        decay.update(create_event("case1", "middle", base_time + timedelta(minutes=5)))
        decay.update(create_event("case3", "end", base_time + timedelta(minutes=15)))

        event_log = prepare_for_mining(window, decay, base_time + timedelta(hours=1))

        assert len(event_log) == 3  # case1, case2, case3

        # case1 should have both window and decay events merged
        case1_trace = next(trace for trace in event_log if trace.attributes["concept:name"] == "case1")
        assert len(case1_trace) == 2
        activities = [event["concept:name"] for event in case1_trace]
        assert "start" in activities
        assert "middle" in activities

    def test_events_sorted_by_timestamp(self):
        """Test that events within each case are sorted by timestamp."""
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)
        window = SlidingWindowCount(max_events=10)
        decay = DecayList(half_life=timedelta(hours=1))

        # Add events out of order
        window.push(create_event("case1", "end", base_time + timedelta(minutes=20)))
        decay.update(create_event("case1", "start", base_time))
        window.push(create_event("case1", "middle", base_time + timedelta(minutes=10)))

        event_log = prepare_for_mining(window, decay, base_time + timedelta(hours=1))

        case1_trace = next(trace for trace in event_log if trace.attributes["concept:name"] == "case1")
        assert len(case1_trace) == 3

        # Check order
        activities = [event["concept:name"] for event in case1_trace]
        assert activities == ["start", "middle", "end"]

    def test_pm4py_format_compatibility(self):
        """Test that returned EventLog is compatible with PM4Py."""
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)
        window = SlidingWindowCount(max_events=10)

        window.push(create_event("case1", "start", base_time, resource="user1"))
        window.push(create_event("case1", "end", base_time + timedelta(minutes=10)))

        event_log = prepare_for_mining(window, DecayList(half_life=timedelta(hours=1)), base_time)

        assert isinstance(event_log, EventLog)
        assert len(event_log) == 1

        trace = event_log[0]
        assert trace.attributes["concept:name"] == "case1"
        assert len(trace) == 2

        # Check event format
        event1 = trace[0]
        assert "concept:name" in event1
        assert "time:timestamp" in event1
        assert event1["concept:name"] == "start"
        assert event1["org:resource"] == "user1"

        event2 = trace[1]
        assert event2["concept:name"] == "end"
        assert "org:resource" not in event2  # No resource for second event

    def test_time_window_compatibility(self):
        """Test that prepare_for_mining works with SlidingWindowTime."""
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)
        window = SlidingWindowTime(max_age=timedelta(hours=1))

        window.push(create_event("case1", "start", base_time))
        window.push(create_event("case1", "end", base_time + timedelta(minutes=10)))

        event_log = prepare_for_mining(window, DecayList(half_life=timedelta(hours=1)), base_time)

        assert len(event_log) == 1
        assert len(event_log[0]) == 2

    def test_case_boundaries_preserved(self):
        """Test that case boundaries are preserved when merging."""
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)
        window = SlidingWindowCount(max_events=10)
        decay = DecayList(half_life=timedelta(hours=1))

        # Add events for different cases
        window.push(create_event("case1", "start", base_time))
        window.push(create_event("case2", "start", base_time + timedelta(minutes=5)))
        decay.update(create_event("case1", "end", base_time + timedelta(minutes=10)))
        decay.update(create_event("case2", "end", base_time + timedelta(minutes=15)))

        event_log = prepare_for_mining(window, decay, base_time + timedelta(hours=1))

        assert len(event_log) == 2

        # Each case should have its own events
        case1_trace = next(trace for trace in event_log if trace.attributes["concept:name"] == "case1")
        case2_trace = next(trace for trace in event_log if trace.attributes["concept:name"] == "case2")

        assert len(case1_trace) == 2
        assert len(case2_trace) == 2

        case1_activities = {event["concept:name"] for event in case1_trace}
        case2_activities = {event["concept:name"] for event in case2_trace}

        assert case1_activities == {"start", "end"}
        assert case2_activities == {"start", "end"}


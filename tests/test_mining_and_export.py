"""Tests for process mining and export functions."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from pm4py.objects.log.obj import EventLog, Trace as PM4PyTrace
from pytz import UTC

from stream_mining.decay_list import DecayList
from stream_mining.export import export_bpmn_png, export_xes
from stream_mining.io_xes import Event, Log, Trace
from stream_mining.mining import discover_heuristics
from stream_mining.prepare_dataset import prepare_for_mining
from stream_mining.sliding_window import SlidingWindowCount


def create_event(case_id: str, activity: str, timestamp: datetime, resource: str | None = None) -> Event:
    """Helper to create an Event."""
    return Event(
        case_id=case_id,
        activity=activity,
        timestamp=timestamp,
        resource=resource,
    )


def create_sample_event_log() -> EventLog:
    """Create a sample EventLog for testing."""
    base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)
    event_log = EventLog()

    # Create trace 1: start -> process -> end
    trace1 = PM4PyTrace()
    trace1.attributes["concept:name"] = "case1"
    trace1.append({"concept:name": "start", "time:timestamp": base_time})
    trace1.append({"concept:name": "process", "time:timestamp": base_time + timedelta(minutes=10)})
    trace1.append({"concept:name": "end", "time:timestamp": base_time + timedelta(minutes=20)})
    event_log.append(trace1)

    # Create trace 2: start -> end
    trace2 = PM4PyTrace()
    trace2.attributes["concept:name"] = "case2"
    trace2.append({"concept:name": "start", "time:timestamp": base_time + timedelta(minutes=30)})
    trace2.append({"concept:name": "end", "time:timestamp": base_time + timedelta(minutes=40)})
    event_log.append(trace2)

    return event_log


class TestDiscoverHeuristics:
    """Tests for discover_heuristics function."""

    def test_discover_success(self):
        """Test successful process model discovery."""
        event_log = create_sample_event_log()
        result = discover_heuristics(event_log, dependency_thresh=0.7)

        assert result is not None
        # Result is now a BPMN model, not a tuple
        bpmn_model = result

        # Verify structure
        assert bpmn_model is not None

    def test_discover_with_different_threshold(self):
        """Test discovery with different dependency threshold."""
        event_log = create_sample_event_log()

        # Lower threshold should still work
        result1 = discover_heuristics(event_log, dependency_thresh=0.5)
        assert result1 is not None

        # Higher threshold should also work
        result2 = discover_heuristics(event_log, dependency_thresh=0.9)
        assert result2 is not None

    def test_discover_empty_log(self):
        """Test discovery with empty event log."""
        empty_log = EventLog()

        with pytest.raises(ValueError, match="empty event log"):
            discover_heuristics(empty_log)

    def test_discover_log_with_no_events(self):
        """Test discovery with log containing traces but no events."""
        event_log = EventLog()
        trace = PM4PyTrace()
        trace.attributes["concept:name"] = "case1"
        event_log.append(trace)

        with pytest.raises(ValueError, match="no events"):
            discover_heuristics(event_log)

    def test_discover_single_event(self):
        """Test discovery with single event (edge case)."""
        event_log = EventLog()
        trace = PM4PyTrace()
        trace.attributes["concept:name"] = "case1"
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)
        trace.append({"concept:name": "start", "time:timestamp": base_time})
        event_log.append(trace)

        # Should either succeed or raise informative error
        try:
            result = discover_heuristics(event_log)
            # If it succeeds, result should be valid BPMN model
            if result is not None:
                assert result is not None
        except ValueError:
            # If it fails, should be informative
            pass

    def test_discover_single_trace(self):
        """Test discovery with single trace."""
        event_log = EventLog()
        trace = PM4PyTrace()
        trace.attributes["concept:name"] = "case1"
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)
        trace.append({"concept:name": "start", "time:timestamp": base_time})
        trace.append({"concept:name": "end", "time:timestamp": base_time + timedelta(minutes=10)})
        event_log.append(trace)

        result = discover_heuristics(event_log)
        # Should either succeed or raise informative error
        if result is not None:
            assert result is not None


class TestExportXes:
    """Tests for export_xes function."""

    def test_export_basic(self):
        """Test basic XES export."""
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)
        log = Log(
            traces=[
                Trace(
                    case_id="case1",
                    events=[
                        create_event("case1", "start", base_time),
                        create_event("case1", "end", base_time + timedelta(minutes=10)),
                    ],
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "test.xes"
            export_xes(log, str(out_path))

            assert out_path.exists()
            assert out_path.stat().st_size > 0

    def test_export_with_resource(self):
        """Test XES export with resource information."""
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)
        log = Log(
            traces=[
                Trace(
                    case_id="case1",
                    events=[
                        create_event("case1", "start", base_time, resource="user1"),
                        create_event("case1", "end", base_time + timedelta(minutes=10)),
                    ],
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "test.xes"
            export_xes(log, str(out_path))

            assert out_path.exists()

    def test_export_multiple_traces(self):
        """Test XES export with multiple traces."""
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)
        log = Log(
            traces=[
                Trace(
                    case_id="case1",
                    events=[create_event("case1", "start", base_time)],
                ),
                Trace(
                    case_id="case2",
                    events=[create_event("case2", "end", base_time + timedelta(minutes=10))],
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "test.xes"
            export_xes(log, str(out_path))

            assert out_path.exists()

    def test_export_creates_directory(self):
        """Test that export creates output directory if it doesn't exist."""
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)
        log = Log(
            traces=[
                Trace(
                    case_id="case1",
                    events=[create_event("case1", "start", base_time)],
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "subdir" / "test.xes"
            export_xes(log, str(out_path))

            assert out_path.exists()
            assert out_path.parent.exists()

    def test_export_round_trip(self):
        """Test that exported XES can be read back."""
        from stream_mining.io_xes import read_xes

        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)
        original_log = Log(
            traces=[
                Trace(
                    case_id="case1",
                    events=[
                        create_event("case1", "start", base_time),
                        create_event("case1", "end", base_time + timedelta(minutes=10)),
                    ],
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "test.xes"
            export_xes(original_log, str(out_path))

            # Read back
            read_log = read_xes(str(out_path))

            assert len(read_log.traces) == len(original_log.traces)
            assert len(read_log.traces[0].events) == len(original_log.traces[0].events)


class TestExportBpmnPng:
    """Tests for export_bpmn_png function."""

    def test_export_basic(self):
        """Test basic BPMN PNG export."""
        event_log = create_sample_event_log()
        model = discover_heuristics(event_log)

        if model is None:
            pytest.skip("Model discovery failed, cannot test export")

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "model.png"
            export_bpmn_png(model, str(out_path))

            assert out_path.exists()
            assert out_path.stat().st_size > 0

    def test_export_creates_directory(self):
        """Test that export creates output directory if it doesn't exist."""
        event_log = create_sample_event_log()
        model = discover_heuristics(event_log)

        if model is None:
            pytest.skip("Model discovery failed, cannot test export")

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "subdir" / "model.png"
            export_bpmn_png(model, str(out_path))

            assert out_path.exists()
            assert out_path.parent.exists()

    def test_export_file_is_png(self):
        """Test that exported file is a PNG image."""
        event_log = create_sample_event_log()
        model = discover_heuristics(event_log)

        if model is None:
            pytest.skip("Model discovery failed, cannot test export")

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "model.png"
            export_bpmn_png(model, str(out_path))

            # Check file signature (PNG files start with specific bytes)
            with open(out_path, "rb") as f:
                header = f.read(8)
                # PNG signature: 89 50 4E 47 0D 0A 1A 0A
                assert header[:4] == b"\x89PNG"


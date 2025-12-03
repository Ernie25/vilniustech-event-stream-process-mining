from datetime import datetime, timedelta

import pytest
from pytz import UTC

from stream_mining.io_xes import Event
from stream_mining.sliding_window import SlidingWindowCount, SlidingWindowTime


def create_event(case_id: str, activity: str, timestamp: datetime, resource: str | None = None) -> Event:
    """Helper to create an Event."""
    return Event(
        case_id=case_id,
        activity=activity,
        timestamp=timestamp,
        resource=resource,
    )


class TestSlidingWindowCount:
    """Tests for SlidingWindowCount."""

    def test_init_valid(self):
        """Test initialization with valid max_events."""
        window = SlidingWindowCount(max_events=5)
        assert window.max_events == 5
        assert len(window) == 0

    def test_init_invalid(self):
        """Test initialization with invalid max_events."""
        with pytest.raises(ValueError, match="max_events must be positive"):
            SlidingWindowCount(max_events=0)
        with pytest.raises(ValueError, match="max_events must be positive"):
            SlidingWindowCount(max_events=-1)

    def test_push_below_limit(self):
        """Test pushing events below the limit."""
        window = SlidingWindowCount(max_events=3)
        event1 = create_event("case1", "start", datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC))
        event2 = create_event("case1", "process", datetime(2024, 1, 1, 10, 5, 0, tzinfo=UTC))

        assert window.push(event1) is None
        assert window.push(event2) is None
        assert len(window) == 2

    def test_push_eviction(self):
        """Test eviction when limit is reached."""
        window = SlidingWindowCount(max_events=2)
        event1 = create_event("case1", "start", datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC))
        event2 = create_event("case1", "process", datetime(2024, 1, 1, 10, 5, 0, tzinfo=UTC))
        event3 = create_event("case2", "end", datetime(2024, 1, 1, 11, 0, 0, tzinfo=UTC))

        window.push(event1)
        window.push(event2)
        evicted = window.push(event3)

        assert evicted == event1  # Oldest event should be evicted
        assert len(window) == 2
        events = list(window.iter_events())
        assert events == [event2, event3]

    def test_push_fifo_order(self):
        """Test that eviction follows FIFO order."""
        window = SlidingWindowCount(max_events=3)
        events = [
            create_event(f"case{i}", f"activity{i}", datetime(2024, 1, 1, 10, i, 0, tzinfo=UTC))
            for i in range(5)
        ]

        evicted_events = []
        for event in events:
            evicted = window.push(event)
            if evicted:
                evicted_events.append(evicted)

        assert len(evicted_events) == 2  # Should evict 2 events
        assert evicted_events[0] == events[0]  # First evicted
        assert evicted_events[1] == events[1]  # Second evicted
        assert len(window) == 3

    def test_iter_events(self):
        """Test iterating over events in order."""
        window = SlidingWindowCount(max_events=5)
        events = [
            create_event(f"case{i}", f"activity{i}", datetime(2024, 1, 1, 10, i, 0, tzinfo=UTC))
            for i in range(3)
        ]

        for event in events:
            window.push(event)

        iterated = list(window.iter_events())
        assert iterated == events
        assert len(iterated) == 3

    def test_stats_empty(self):
        """Test stats on empty window."""
        window = SlidingWindowCount(max_events=5)
        stats = window.stats()

        assert stats["size"] == 0
        assert stats["max_events"] == 5
        assert stats["oldest_timestamp"] is None
        assert stats["newest_timestamp"] is None

    def test_stats_with_events(self):
        """Test stats with events."""
        window = SlidingWindowCount(max_events=5)
        event1 = create_event("case1", "start", datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC))
        event2 = create_event("case1", "end", datetime(2024, 1, 1, 11, 0, 0, tzinfo=UTC))

        window.push(event1)
        window.push(event2)
        stats = window.stats()

        assert stats["size"] == 2
        assert stats["max_events"] == 5
        assert stats["oldest_timestamp"] == event1.timestamp
        assert stats["newest_timestamp"] == event2.timestamp

    def test_rapid_eviction(self):
        """Test rapid eviction with many events."""
        window = SlidingWindowCount(max_events=3)
        events = [
            create_event(f"case{i}", f"activity{i}", datetime(2024, 1, 1, 10, i, 0, tzinfo=UTC))
            for i in range(10)
        ]

        evicted_count = 0
        for event in events:
            if window.push(event) is not None:
                evicted_count += 1

        assert evicted_count == 7  # Should evict 7 events (10 - 3)
        assert len(window) == 3
        final_events = list(window.iter_events())
        assert final_events == events[-3:]  # Last 3 events should remain

    def test_empty_window(self):
        """Test operations on empty window."""
        window = SlidingWindowCount(max_events=5)

        assert len(window) == 0
        assert list(window.iter_events()) == []
        stats = window.stats()
        assert stats["size"] == 0


class TestSlidingWindowTime:
    """Tests for SlidingWindowTime."""

    def test_init_valid(self):
        """Test initialization with valid max_age."""
        max_age = timedelta(hours=1)
        window = SlidingWindowTime(max_age=max_age)
        assert window.max_age == max_age
        assert len(window) == 0

    def test_init_invalid(self):
        """Test initialization with invalid max_age."""
        with pytest.raises(ValueError, match="max_age must be positive"):
            SlidingWindowTime(max_age=timedelta(0))
        with pytest.raises(ValueError, match="max_age must be positive"):
            SlidingWindowTime(max_age=timedelta(seconds=-1))

    def test_push_within_time_window(self):
        """Test pushing events within the time window."""
        max_age = timedelta(hours=1)
        window = SlidingWindowTime(max_age=max_age)
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)

        event1 = create_event("case1", "start", base_time)
        event2 = create_event("case1", "process", base_time + timedelta(minutes=30))

        assert window.push(event1) is None
        assert window.push(event2) is None
        assert len(window) == 2

    def test_push_eviction_old_events(self):
        """Test eviction of events older than max_age."""
        max_age = timedelta(hours=1)
        window = SlidingWindowTime(max_age=max_age)
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)

        event1 = create_event("case1", "start", base_time)
        event2 = create_event("case1", "process", base_time + timedelta(minutes=30))
        # This event is 2 hours after base_time, cutoff is 11:00
        # Both event1 (10:00) and event2 (10:30) are older than cutoff, so both should be evicted
        event3 = create_event("case2", "end", base_time + timedelta(hours=2))

        window.push(event1)
        window.push(event2)
        evicted = window.push(event3)

        # First evicted event should be event1 (oldest)
        assert evicted == event1
        # Both old events should be evicted, only event3 remains
        assert len(window) == 1
        events = list(window.iter_events())
        assert event1 not in events
        assert event2 not in events
        assert event3 in events

    def test_push_multiple_evictions(self):
        """Test eviction of multiple old events."""
        max_age = timedelta(hours=1)
        window = SlidingWindowTime(max_age=max_age)
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)

        # Add events at 10:00, 10:30, 11:00
        event1 = create_event("case1", "start", base_time)
        event2 = create_event("case2", "process", base_time + timedelta(minutes=30))
        event3 = create_event("case3", "review", base_time + timedelta(hours=1))

        window.push(event1)
        window.push(event2)
        window.push(event3)

        # Add event at 12:00 - should evict event1 and event2
        event4 = create_event("case4", "end", base_time + timedelta(hours=2))
        evicted = window.push(event4)

        assert evicted == event1  # First evicted event
        assert len(window) == 2  # Only event3 and event4 remain
        events = list(window.iter_events())
        assert event1 not in events
        assert event2 not in events
        assert event3 in events
        assert event4 in events

    def test_iter_events(self):
        """Test iterating over events in order."""
        max_age = timedelta(hours=2)
        window = SlidingWindowTime(max_age=max_age)
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)

        events = [
            create_event(f"case{i}", f"activity{i}", base_time + timedelta(minutes=i * 10))
            for i in range(3)
        ]

        for event in events:
            window.push(event)

        iterated = list(window.iter_events())
        assert iterated == events
        assert len(iterated) == 3

    def test_stats_empty(self):
        """Test stats on empty window."""
        max_age = timedelta(hours=1)
        window = SlidingWindowTime(max_age=max_age)
        stats = window.stats()

        assert stats["size"] == 0
        assert stats["max_age"] == max_age
        assert stats["oldest_timestamp"] is None
        assert stats["newest_timestamp"] is None

    def test_stats_with_events(self):
        """Test stats with events."""
        max_age = timedelta(hours=2)
        window = SlidingWindowTime(max_age=max_age)
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)

        event1 = create_event("case1", "start", base_time)
        event2 = create_event("case1", "end", base_time + timedelta(hours=1))

        window.push(event1)
        window.push(event2)
        stats = window.stats()

        assert stats["size"] == 2
        assert stats["max_age"] == max_age
        assert stats["oldest_timestamp"] == event1.timestamp
        assert stats["newest_timestamp"] == event2.timestamp

    def test_rapid_eviction(self):
        """Test rapid eviction with many events."""
        max_age = timedelta(minutes=5)
        window = SlidingWindowTime(max_age=max_age)
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)

        # Create 10 events, each 1 minute apart
        events = [
            create_event(f"case{i}", f"activity{i}", base_time + timedelta(minutes=i))
            for i in range(10)
        ]

        evicted_count = 0
        for event in events:
            if window.push(event) is not None:
                evicted_count += 1

        # Events older than 5 minutes relative to newest should be evicted
        # With newest at 10:09, events before 10:04 should be evicted
        # So events at 10:00, 10:01, 10:02, 10:03 should be evicted
        assert evicted_count >= 4
        assert len(window) <= 6  # At most 6 events (10:04 to 10:09)

    def test_empty_window(self):
        """Test operations on empty window."""
        max_age = timedelta(hours=1)
        window = SlidingWindowTime(max_age=max_age)

        assert len(window) == 0
        assert list(window.iter_events()) == []
        stats = window.stats()
        assert stats["size"] == 0

    def test_time_window_boundary(self):
        """Test events exactly at the time boundary."""
        max_age = timedelta(hours=1)
        window = SlidingWindowTime(max_age=max_age)
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)

        event1 = create_event("case1", "start", base_time)
        # Event exactly 1 hour later - should keep event1
        event2 = create_event("case2", "end", base_time + timedelta(hours=1))

        window.push(event1)
        evicted = window.push(event2)

        # event1 is exactly at the boundary (1 hour old), should be kept
        assert evicted is None
        assert len(window) == 2

        # Event slightly more than 1 hour later - should evict event1
        event3 = create_event("case3", "process", base_time + timedelta(hours=1, seconds=1))
        evicted = window.push(event3)

        assert evicted == event1
        assert len(window) == 2


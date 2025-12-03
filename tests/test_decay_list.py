"""Tests for decay list implementation."""

from datetime import datetime, timedelta

import pytest
from pytz import UTC

from stream_mining.io_xes import Event
from stream_mining.decay_list import DecayList


def create_event(case_id: str, activity: str, timestamp: datetime, resource: str | None = None) -> Event:
    """Helper to create an Event."""
    return Event(
        case_id=case_id,
        activity=activity,
        timestamp=timestamp,
        resource=resource,
    )


class TestDecayList:
    """Tests for DecayList."""

    def test_init_valid(self):
        """Test initialization with valid parameters."""
        half_life = timedelta(hours=1)
        decay_list = DecayList(half_life=half_life)
        assert decay_list.half_life == half_life
        assert decay_list.max_items is None
        assert len(decay_list._events) == 0

        decay_list_with_max = DecayList(half_life=half_life, max_items=100)
        assert decay_list_with_max.max_items == 100

    def test_init_invalid(self):
        """Test initialization with invalid half_life."""
        with pytest.raises(ValueError, match="half_life must be positive"):
            DecayList(half_life=timedelta(0))
        with pytest.raises(ValueError, match="half_life must be positive"):
            DecayList(half_life=timedelta(seconds=-1))

    def test_update_add_event(self):
        """Test adding a new event."""
        half_life = timedelta(hours=1)
        decay_list = DecayList(half_life=half_life)
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)

        event = create_event("case1", "start", base_time)
        decay_list.update(event)

        assert len(decay_list._events) == 1
        assert decay_list._events[0][0] == event
        assert decay_list._events[0][1] == base_time

    def test_update_with_timestamp(self):
        """Test updating event with explicit timestamp."""
        half_life = timedelta(hours=1)
        decay_list = DecayList(half_life=half_life)
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)
        explicit_time = datetime(2024, 1, 1, 11, 0, 0, tzinfo=UTC)

        event = create_event("case1", "start", base_time)
        decay_list.update(event, timestamp=explicit_time)

        assert decay_list._events[0][1] == explicit_time

    def test_update_multiple_events_same_case(self):
        """Test adding multiple events with same case_id."""
        half_life = timedelta(hours=1)
        decay_list = DecayList(half_life=half_life)
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)

        event1 = create_event("case1", "start", base_time)
        event2 = create_event("case1", "process", base_time + timedelta(minutes=30))

        decay_list.update(event1)
        assert len(decay_list._events) == 1

        decay_list.update(event2)
        assert len(decay_list._events) == 2  # Should add, not update
        assert event1 in [e for e, _ in decay_list._events]
        assert event2 in [e for e, _ in decay_list._events]

    def test_exponential_decay_formula(self):
        """Test that weight follows exponential decay formula."""
        half_life = timedelta(hours=1)
        decay_list = DecayList(half_life=half_life)
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)

        event = create_event("case1", "start", base_time)
        decay_list.update(event)

        # At time = 0, weight should be 1.0
        weight_at_start = decay_list._compute_weight(base_time, base_time)
        assert abs(weight_at_start - 1.0) < 1e-6

        # At time = half_life, weight should be 0.5
        time_at_half = base_time + half_life
        weight_at_half = decay_list._compute_weight(base_time, time_at_half)
        assert abs(weight_at_half - 0.5) < 1e-6

        # At time = 2 * half_life, weight should be 0.25
        time_at_double = base_time + 2 * half_life
        weight_at_double = decay_list._compute_weight(base_time, time_at_double)
        assert abs(weight_at_double - 0.25) < 1e-6

    def test_weight_monotonic_decay(self):
        """Test that weight decays monotonically as time increases."""
        half_life = timedelta(hours=1)
        decay_list = DecayList(half_life=half_life)
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)

        event = create_event("case1", "start", base_time)
        decay_list.update(event)

        times = [
            base_time,
            base_time + timedelta(minutes=15),
            base_time + timedelta(minutes=30),
            base_time + timedelta(minutes=45),
            base_time + timedelta(hours=1),
            base_time + timedelta(hours=2),
        ]

        weights = [decay_list._compute_weight(base_time, t) for t in times]

        # Weights should be monotonically decreasing
        for i in range(len(weights) - 1):
            assert weights[i] >= weights[i + 1], f"Weight increased from {weights[i]} to {weights[i+1]}"

    def test_pull_existing_case(self):
        """Test pulling events for existing case_id."""
        half_life = timedelta(hours=1)
        decay_list = DecayList(half_life=half_life)
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)

        event1 = create_event("case1", "start", base_time)
        event2 = create_event("case1", "process", base_time + timedelta(minutes=30))
        event3 = create_event("case2", "end", base_time + timedelta(hours=1))

        decay_list.update(event1)
        decay_list.update(event2)
        decay_list.update(event3)

        case1_events = decay_list.pull("case1")
        assert case1_events is not None
        assert len(case1_events) == 2
        assert event1 in case1_events
        assert event2 in case1_events

    def test_pull_nonexistent_case(self):
        """Test pulling events for nonexistent case_id."""
        half_life = timedelta(hours=1)
        decay_list = DecayList(half_life=half_life)
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)

        event = create_event("case1", "start", base_time)
        decay_list.update(event)

        result = decay_list.pull("nonexistent")
        assert result is None

    def test_sweep_removes_low_weight_events(self):
        """Test that sweep removes events with weight below threshold."""
        half_life = timedelta(hours=1)
        decay_list = DecayList(half_life=half_life)
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)

        # Add events at different times
        event1 = create_event("case1", "start", base_time)
        event2 = create_event("case2", "process", base_time + timedelta(minutes=30))
        event3 = create_event("case3", "end", base_time + timedelta(hours=1))

        decay_list.update(event1)
        decay_list.update(event2)
        decay_list.update(event3)

        # At time = base_time + 3 hours:
        # event1 weight = 0.5 ** 3 = 0.125
        # event2 weight = 0.5 ** 2.5 ≈ 0.177
        # event3 weight = 0.5 ** 2 = 0.25
        # Using eps=0.15, event1 should be removed (0.125 < 0.15), others kept
        sweep_time = base_time + timedelta(hours=3)
        removed_count = decay_list.sweep(sweep_time, eps=0.15)

        # event1 should be removed (weight ~0.125 < 0.15), event2 and event3 should remain
        assert removed_count >= 1
        remaining_events = [event for event, _ in decay_list._events]
        assert event1 not in remaining_events
        assert event2 in remaining_events
        assert event3 in remaining_events

    def test_sweep_with_different_eps(self):
        """Test sweep with different epsilon values."""
        half_life = timedelta(hours=1)
        decay_list = DecayList(half_life=half_life)
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)

        event = create_event("case1", "start", base_time)
        decay_list.update(event)

        # At 2 hours, weight should be 0.25
        sweep_time = base_time + timedelta(hours=2)

        # With eps=0.3, should remove (weight 0.25 < 0.3)
        removed1 = decay_list.sweep(sweep_time, eps=0.3)
        assert removed1 == 1

        # Reset and test with eps=0.2 (weight 0.25 > 0.2, should keep)
        decay_list = DecayList(half_life=half_life)
        decay_list.update(event)
        removed2 = decay_list.sweep(sweep_time, eps=0.2)
        assert removed2 == 0

    def test_max_items_enforcement(self):
        """Test that max_items limit is enforced."""
        half_life = timedelta(hours=1)
        max_items = 3
        decay_list = DecayList(half_life=half_life, max_items=max_items)
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)

        # Add more events than max_items
        for i in range(5):
            event = create_event(f"case{i}", f"activity{i}", base_time + timedelta(minutes=i))
            decay_list.update(event)

        # Should only keep max_items events (FIFO eviction)
        assert len(decay_list._events) == max_items

        # Oldest events should be evicted
        remaining_case_ids = {event.case_id for event, _ in decay_list._events}
        assert "case0" not in remaining_case_ids  # Oldest should be evicted
        assert "case4" in remaining_case_ids  # Newest should remain

    def test_stats_empty(self):
        """Test stats on empty decay list."""
        half_life = timedelta(hours=1)
        decay_list = DecayList(half_life=half_life)
        stats = decay_list.stats()

        assert stats["size"] == 0
        assert stats["max_items"] is None
        assert stats["min_weight"] is None
        assert stats["max_weight"] is None
        assert stats["half_life"] == half_life

    def test_stats_with_events(self):
        """Test stats with events."""
        half_life = timedelta(hours=1)
        decay_list = DecayList(half_life=half_life)
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)

        event1 = create_event("case1", "start", base_time)
        event2 = create_event("case2", "process", base_time + timedelta(minutes=30))

        decay_list.update(event1)
        decay_list.update(event2)

        reference_time = base_time + timedelta(minutes=15)
        stats = decay_list.stats(reference_time=reference_time)

        assert stats["size"] == 2
        assert stats["half_life"] == half_life
        assert stats["min_weight"] is not None
        assert stats["max_weight"] is not None
        assert stats["min_weight"] <= stats["max_weight"]
        assert stats["max_weight"] <= 1.0

    def test_tiny_half_life(self):
        """Test behavior with very small half-life (rapid decay)."""
        half_life = timedelta(seconds=1)  # Very short half-life
        decay_list = DecayList(half_life=half_life)
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)

        event = create_event("case1", "start", base_time)
        decay_list.update(event)

        # After 2 seconds, weight should be very low (~0.25)
        later_time = base_time + timedelta(seconds=2)
        weight = decay_list._compute_weight(base_time, later_time)
        assert weight < 0.3

        # Sweep should remove it
        removed = decay_list.sweep(later_time, eps=0.1)
        assert removed == 1

    def test_multiple_events_same_case(self):
        """Test handling multiple events with same case_id."""
        half_life = timedelta(hours=1)
        decay_list = DecayList(half_life=half_life)
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)

        # Add multiple events with same case_id
        event1 = create_event("case1", "start", base_time)
        event2 = create_event("case1", "process", base_time + timedelta(minutes=30))
        event3 = create_event("case1", "end", base_time + timedelta(hours=1))

        decay_list.update(event1)
        decay_list.update(event2)
        decay_list.update(event3)

        # Should have all three events
        assert len(decay_list._events) == 3

        # Pull should return all events for case1
        events = decay_list.pull("case1")
        assert events is not None
        assert len(events) == 3
        assert event1 in events
        assert event2 in events
        assert event3 in events




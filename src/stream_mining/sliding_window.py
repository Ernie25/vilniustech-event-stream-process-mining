from collections import deque
from datetime import datetime, timedelta
from typing import Dict, Iterator, Optional, Protocol

from stream_mining.io_xes import Event


class SlidingWindow(Protocol):
    """Protocol defining the common interface for sliding windows."""

    def push(self, event: Event) -> Optional[Event]:
        """Add event to window, returning evicted event if any."""
        ...

    def __len__(self) -> int:
        ...

    def iter_events(self) -> Iterator[Event]:
        ...

    def stats(self) -> Dict[str, any]:
        ...


class SlidingWindowCount:
    def __init__(self, max_events: int):
        if max_events <= 0:
            raise ValueError(f"max_events must be positive, got {max_events}")
        self.max_events = max_events
        self._events: deque[Event] = deque()

    def push(self, event: Event) -> Optional[Event]:
        """Add event to window, evicting oldest if limit exceeded.

        Args:
            event: Event to add to the window.

        Returns:
            Evicted event if window was full, None otherwise.
        """
        evicted = None
        if len(self._events) >= self.max_events:
            evicted = self._events.popleft()

        self._events.append(event)
        return evicted

    def __len__(self) -> int:
        """Return current number of events in window.

        Returns:
            Number of events currently in the window.
        """
        return len(self._events)

    def iter_events(self) -> Iterator[Event]:
        """Iterate over all events in window in order.

        Yields:
            Events in the window from oldest to newest.
        """
        yield from self._events

    def stats(self) -> Dict[str, any]:
        """Return statistics about the window.

        Returns:
            Dictionary with window statistics including:
            - size: Current number of events
            - max_events: Maximum capacity
            - oldest_timestamp: Timestamp of oldest event (if any)
            - newest_timestamp: Timestamp of newest event (if any)
        """
        stats_dict: Dict[str, any] = {
            "size": len(self._events),
            "max_events": self.max_events,
        }

        if self._events:
            timestamps = [event.timestamp for event in self._events]
            stats_dict["oldest_timestamp"] = min(timestamps)
            stats_dict["newest_timestamp"] = max(timestamps)
        else:
            stats_dict["oldest_timestamp"] = None
            stats_dict["newest_timestamp"] = None

        return stats_dict


class SlidingWindowTime:
    """Sliding window with time-based eviction.

    Maintains events within a specified time window. Events older than
    max_age relative to the newest event are automatically evicted.

    Attributes:
        max_age: Maximum age of events to keep in the window.
    """

    def __init__(self, max_age: timedelta):
        """Initialize time-based sliding window.

        Args:
            max_age: Maximum age of events to keep (relative to newest event).

        Raises:
            ValueError: If max_age is not positive.
        """
        if max_age <= timedelta(0):
            raise ValueError(f"max_age must be positive, got {max_age}")
        self.max_age = max_age
        self._events: deque[Event] = deque()

    def push(self, event: Event) -> Optional[Event]:
        """Add event to window, evicting events older than max_age.

        Args:
            event: Event to add to the window.

        Returns:
            First evicted event if any were removed, None otherwise.
        """
        # Evict events older than max_age relative to the new event
        evicted = None
        cutoff_time = event.timestamp - self.max_age

        # Capture the first evicted event
        if self._events and self._events[0].timestamp < cutoff_time:
            evicted = self._events[0]

        # Evict all events older than cutoff_time
        while self._events and self._events[0].timestamp < cutoff_time:
            self._events.popleft()

        self._events.append(event)

        return evicted

    def __len__(self) -> int:
        """Return current number of events in window.

        Returns:
            Number of events currently in the window.
        """
        return len(self._events)

    def iter_events(self) -> Iterator[Event]:
        """Iterate over all events in window in order.

        Yields:
            Events in the window from oldest to newest.
        """
        yield from self._events

    def stats(self) -> Dict[str, any]:
        """Return statistics about the window.

        Returns:
            Dictionary with window statistics including:
            - size: Current number of events
            - max_age: Maximum age of events to keep
            - oldest_timestamp: Timestamp of oldest event (if any)
            - newest_timestamp: Timestamp of newest event (if any)
        """
        stats_dict: Dict[str, any] = {
            "size": len(self._events),
            "max_age": self.max_age,
        }

        if self._events:
            timestamps = [event.timestamp for event in self._events]
            stats_dict["oldest_timestamp"] = min(timestamps)
            stats_dict["newest_timestamp"] = max(timestamps)
        else:
            stats_dict["oldest_timestamp"] = None
            stats_dict["newest_timestamp"] = None

        return stats_dict


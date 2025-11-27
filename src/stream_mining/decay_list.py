from datetime import datetime, timedelta
from typing import Dict, List, Optional

from stream_mining.io_xes import Event


class DecayList:
    """Exponential decay list with configurable half-life.

    Maintains events with exponential decay weights. Events decay over time
    according to the formula: w(Δt) = 0.5 ** (Δt / half_life), where Δt is
    the time since the event occurred.

    Attributes:
        half_life: Time period after which an event's weight halves.
        max_items: Maximum number of events to keep (None = unlimited).
    """

    def __init__(self, half_life: timedelta, max_items: Optional[int] = None):
        """Initialize decay list.

        Args:
            half_life: Time period after which an event's weight halves.
            max_items: Maximum number of events to keep (None = unlimited).

        Raises:
            ValueError: If half_life is not positive.
        """
        if half_life <= timedelta(0):
            raise ValueError(f"half_life must be positive, got {half_life}")
        self.half_life = half_life
        self.max_items = max_items
        # Store events with their timestamps: [(event, timestamp), ...]
        self._events: List[tuple[Event, datetime]] = []

    def _compute_weight(self, event_timestamp: datetime, reference_time: datetime) -> float:
        """Compute decay weight for an event at a given reference time.

        Args:
            event_timestamp: When the event occurred.
            reference_time: Reference time for weight calculation.

        Returns:
            Weight value between 0 and 1.
        """
        delta_t = (reference_time - event_timestamp).total_seconds()
        half_life_seconds = self.half_life.total_seconds()

        if half_life_seconds <= 0:
            return 0.0

        # Weight formula: w(Δt) = 0.5 ** (Δt / half_life)
        weight = 0.5 ** (delta_t / half_life_seconds)
        return max(0.0, weight)  # Ensure non-negative

    def update(self, event: Event, timestamp: Optional[datetime] = None) -> None:
        """Add event to decay list.

        Adds a new event to the decay list. Multiple events with the same
        case_id can coexist. If max_items is exceeded, the oldest event
        is evicted (FIFO).

        Args:
            event: Event to add.
            timestamp: Timestamp for the event (defaults to event.timestamp).
        """
        if timestamp is None:
            timestamp = event.timestamp

        # Add new event
        self._events.append((event, timestamp))

        # Enforce max_items limit (FIFO eviction)
        if self.max_items is not None and len(self._events) > self.max_items:
            self._events.pop(0)  # Remove oldest event

    def pull(self, case_id: str) -> Optional[List[Event]]:
        """Retrieve all events for a given case_id.

        Args:
            case_id: Case identifier to retrieve events for.

        Returns:
            List of events for the case_id, or None if not found.
        """
        events = [event for event, _ in self._events if event.case_id == case_id]
        return events if events else None

    def sweep(self, now: datetime, eps: float = 1e-3) -> int:
        """Remove events with weight below threshold.

        Args:
            now: Reference time for weight calculation.
            eps: Minimum weight threshold (default 1e-3).

        Returns:
            Number of events removed.
        """
        initial_count = len(self._events)
        self._events = [
            (event, timestamp)
            for event, timestamp in self._events
            if self._compute_weight(timestamp, now) >= eps
        ]
        return initial_count - len(self._events)

    def stats(self, reference_time: Optional[datetime] = None) -> Dict[str, any]:
        """Return statistics about the decay list.

        Args:
            reference_time: Reference time for weight calculation (defaults to now).

        Returns:
            Dictionary with statistics including:
            - size: Current number of events
            - max_items: Maximum capacity (None if unlimited)
            - min_weight: Minimum weight among all events
            - max_weight: Maximum weight among all events
            - half_life: Half-life period
        """
        if reference_time is None:
            if self._events:
                # Use timezone from first event if available
                tz = self._events[0][1].tzinfo
                reference_time = datetime.now(tz) if tz else datetime.now()
            else:
                reference_time = datetime.now()

        stats_dict: Dict[str, any] = {
            "size": len(self._events),
            "max_items": self.max_items,
            "half_life": self.half_life,
        }

        if self._events:
            weights = [
                self._compute_weight(timestamp, reference_time)
                for _, timestamp in self._events
            ]
            stats_dict["min_weight"] = min(weights)
            stats_dict["max_weight"] = max(weights)
        else:
            stats_dict["min_weight"] = None
            stats_dict["max_weight"] = None

        return stats_dict


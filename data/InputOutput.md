# Streaming Process Mining System - Input/Output Specification

## Overview

The streaming process mining system processes event streams in real-time to discover
and maintain process models. The system uses a sliding window to track recent events
and a decay list to preserve historical context. It applies Heuristics Miner to discover
BPMN process models that represent the actual process flow observed in the event stream.

The system operates in a continuous streaming mode: events arrive one at a time,
are processed through the sliding window and decay list, and process models are
periodically discovered from the current state of the system.

## Streaming Processing Phase


### Input (**x**):

- **Initial XES Event Log** (optional): `data/in/sample_small.xes`
    - Format: XES (eXtensible Event Stream) format
    - Contains: Traces (cases) with events
    - Each event has: case_id, activity name, timestamp, optional resource
    - Example structure:
      ```
      <trace>
        <string key="concept:name" value="case1"/>
        <event>
          <string key="concept:name" value="start"/>
          <date key="time:timestamp" value="2024-01-01T10:00:00+00:00"/>
        </event>
        <event>
          <string key="concept:name" value="process"/>
          <date key="time:timestamp" value="2024-01-01T10:10:00+00:00"/>
        </event>
      </trace>
      ```

- **Individual Events** (streaming input):
    - Event object with attributes:
        - `case_id`: String identifier for the case/trace
        - `activity`: String name of the activity performed
        - `timestamp`: Datetime object (timezone-aware)
        - `resource`: Optional string identifier for the resource/person
    - Events arrive sequentially in the stream
    - Events may belong to existing cases or new cases

### Process:

1. **Event Ingestion**: Load initial XES file or receive events from stream
2. **Sliding Window Processing**:
    - Events are pushed to sliding window (fixed size, e.g., 50 events)
    - When window is full, oldest events are evicted (FIFO)
    - Evicted events are added to decay list
3. **Decay List Management**:
    - Events in decay list have exponential decay weights
    - Weights decrease over time: w(Δt) = 0.5^(Δt / half_life)
    - Low-weight events can be swept (removed) periodically
4. **Dataset Preparation**:
    - Window events are PRIMARY source for mining
    - Decay list "revives" traces not currently in window
    - Events are merged and converted to PM4Py EventLog format
5. **Process Model Discovery**:
    - Heuristics Miner discovers process model from EventLog
    - Process model is converted to BPMN model
    - Dependency threshold controls model complexity
6. **Export and Visualization**:
    - BPMN model exported as PNG image
    - Current window state exported as XES snapshot

### Output (**y**):

- **BPMN Process Model** (`data/out/model_bpmn.png`):
    - Visual representation of discovered process
    - Shows activities, gateways, and flow relationships
    - Format: PNG image file
    - Example: start → process → review → end

- **XES Snapshot** (`data/out/stream_snapshot.xes`):
    - Current state of sliding window exported as XES
    - Contains all events currently in the window
    - Can be used for round-trip verification
    - Format: XES file compatible with PM4Py

## Examples

Example 1 - Simple Sequential Process:

x (Input):
- Initial XES file: `data/in/sample_small.xes`
- Contains 3 traces:
* case1: start → process → end
* case2: start → end
* case3: start → process → review → end

Process:
- Window size: 50 events
- Half-life: 24 hours
- Dependency threshold: 0.5

y (Output):
- BPMN Model: Simple flow showing start → process → review → end
- XES Snapshot: 9 events from window (3 traces)
- Statistics:
* Window: 9 events
* Decay list: 0 events (no evictions yet)
* Model activities: start, process, review, end

Example 2 - Window Full with Evictions:

x (Input):
- Stream of 100 events arriving sequentially
- Window size: 50 events
- Events belong to 20 different cases

Process:
- First 50 events fill the window
- Events 51-100 cause evictions
- Each evicted event is added to decay list

y (Output):
- BPMN Model: Process model from current 50 events in window
- XES Snapshot: Last 50 events (events 51-100)
- Statistics:
* Window: 50 events (full)
* Decay list: 50 events (evicted events)
* Evicted count: 50 events
* Cases in window: ~10-15 cases
* Cases in decay: ~5-10 cases

Example 3 - Trace Revival from Decay:

x (Input):
- Event stream where case1 is evicted from window
- Later, new event arrives for case1

Process:
1. case1 events fill window, then get evicted → added to decay
2. Window processes other cases
3. New event arrives: case1, activity="finalize"
4. System detects case1 not in window but exists in decay
5. case1 is "revived" from decay list for mining

y (Output):
- BPMN Model: Includes complete case1 trace (old events from decay + new event)
- XES Snapshot: Current window + revived case1 events
- Statistics:
* Window: 50 events
* Decay list: 45 events (case1 events included)
* Revived traces: 1 (case1)

## Key Observations


1. **Sliding Window as Primary Source**:
    - Window contains the most recent events (current state)
    - Window size determines how much recent history is considered
    - Larger windows: More context but slower processing
    - Smaller windows: Faster but less context

2. **Decay List as Historical Context**:
    - Preserves information about evicted events
    - Allows "reviving" traces when new events arrive
    - Exponential decay ensures old events have less influence
    - Sweep operation removes very old/low-weight events

3. **Process Model Discovery**:
    - Heuristics Miner is robust to noise
    - Dependency threshold controls model complexity
    - Lower threshold: More detailed, complex models
    - Higher threshold: Simpler, cleaner models

4. **Streaming Characteristics**:
    - System processes events one at a time
    - Models can be discovered at any point
    - Models evolve as new events arrive
    - Old events naturally fade out through decay

5. **Trade-offs**:
    - Window size: Memory vs. context
    - Half-life: Historical preservation vs. relevance
    - Dependency threshold: Model complexity vs. clarity
    - Sweep frequency: Decay list size vs. computational cost

## Input Data Requirements

1. **XES File Format**:
    - Standard XES format (IEEE 1849)
    - Required attributes: case:concept:name, concept:name, time:timestamp
    - Optional attributes: org:resource
    - Timestamps must be timezone-aware

2. **Event Stream Format**:
    - Events arrive sequentially
    - Each event must have: case_id, activity, timestamp
    - Timestamps should be monotonically increasing (or close to it)
    - Events can belong to any case (existing or new)

3. **Data Quality**:
    - Case IDs should be consistent within a trace
    - Activities should be standardized (same name = same activity)
    - Timestamps should be valid datetime objects
    - Missing optional attributes handled gracefully

## Configuration Parameters


1. **Sliding Window**:
    - `max_events`: Maximum number of events in window (default: 50)
    - Type: `SlidingWindowCount` (count-based) or `SlidingWindowTime` (time-based)

2. **Decay List**:
    - `half_life`: Time period for weight to halve (default: 24 hours)
    - `max_items`: Maximum events in decay list (optional, default: None)
    - `eps`: Minimum weight threshold for sweep (default: 0.001)

3. **Process Mining**:
    - `dependency_thresh`: Dependency threshold for Heuristics Miner (default: 0.7)
    - Range: 0.0-1.0
    - Lower: More dependencies included (complex model)
    - Higher: Only strong dependencies (simple model)

## Conclusion

The streaming process mining system transforms event streams into process models
through a two-stage approach: sliding window (recent events) and decay list
(historical context). The system discovers BPMN models using Heuristics Miner,
which balances model accuracy with interpretability.

Key advantages:
- Real-time processing of event streams
- Adaptive models that evolve with new data
- Historical context preservation through decay
- Configurable complexity through dependency threshold

The system is particularly valuable for:
- Monitoring business processes in real-time
- Discovering process patterns from event logs
- Maintaining up-to-date process models
- Analyzing process evolution over time


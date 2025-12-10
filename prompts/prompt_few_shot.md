# Few-Shot Prompt: Streaming Process Mining with Examples

## Task

You are an AI agent that performs end-to-end streaming process mining. Given input data X, process it through all stages and return the predicted output y. Below are examples of how the system works with known (X,y) pairs.

## Example 1: Simple Sequential Process

### Input (X):
```
File: data/in/sample_small.xes
Contains 3 traces:
- case1: start → process → end
- case2: start → end
- case3: start → process → review → end

Configuration:
- Window size: 50 events
- Half-life: 24 hours
- Dependency threshold: 0.5
```

### Processing:
1. **Data Understanding**: Parsed 3 traces with 9 total events. Activities: start, process, review, end
2. **Inference**: All 9 events fit in window (no evictions). Window size: 9 events. Decay list: 0 events
3. **Dataset Preparation**: Prepared 3 traces for mining with all events from window
4. **Reasoning**: Discovered process with two XOR gateways - one after start (skip or process), one after process (end or review)

### Output (y):
```
BPMN Model: 
- Start node → XOR Gateway 1
  - Path A: Skip directly to End (1 case)
  - Path B: Go to Process activity (2 cases)
    - Process → XOR Gateway 2
      - Path B1: Complete after Process (1 case)
      - Path B2: Require Review (1 case)
        - Review → End

XES Snapshot: 9 events from 3 traces

Statistics:
- Window: 9 events
- Decay list: 0 events
- Traces: 3
- Activities: start, process, review, end
- Model complexity: 2 XOR gateways, 4 activities
```

---

## Example 2: Window Full with Evictions

### Input (X):
```
Stream of 100 events arriving sequentially
- Window size: 50 events
- Events belong to 20 different cases
- Events arrive one at a time
```

### Processing:
1. **Data Understanding**: Received stream of 100 events from 20 cases
2. **Inference**: 
   - First 50 events fill the window
   - Events 51-100 cause oldest events to be evicted
   - Each evicted event added to decay list with exponential decay weights
3. **Dataset Preparation**: Prepared dataset from current 50 events in window
4. **Reasoning**: Discovered process model from the most recent 50 events

### Output (y):
```
BPMN Model: Process model discovered from current window state
- Represents patterns in the most recent 50 events

XES Snapshot: Last 50 events (events 51-100)

Statistics:
- Window: 50 events (full)
- Decay list: 50 events (evicted events)
- Evicted count: 50 events
- Cases in window: ~10-15 cases
- Cases in decay: ~5-10 cases
```

---

## Example 3: Trace Revival from Decay

### Input (X):
```
Event stream scenario:
1. case1 events fill window, then get evicted → added to decay
2. Window processes other cases (case2, case3, etc.)
3. New event arrives: case1, activity="finalize", timestamp=now
```

### Processing:
1. **Data Understanding**: New event for case1 detected
2. **Inference**: 
   - System detects case1 not in window but exists in decay list
   - case1 is "revived" from decay list
   - New event added to window
3. **Dataset Preparation**: Combined window events + revived case1 events from decay
4. **Reasoning**: Discovered process model includes complete case1 trace (old + new events)

### Output (y):
```
BPMN Model: Includes complete case1 trace
- Old events from decay list + new "finalize" event
- Shows full process flow for case1

XES Snapshot: Current window + revived case1 events

Statistics:
- Window: 50 events
- Decay list: 45 events (case1 events included)
- Revived traces: 1 (case1)
- Complete case1 trace: [original events] → finalize → end
```

---

## New Task: Process the Following Input

### Input (X):
```
File: data/in/new_process.xes

Traces:
- case_001: register → verify → approve → complete
- case_002: register → verify → reject
- case_003: register → verify → approve → notify → complete
- case_004: register → verify → approve → complete
- case_005: register → verify → reject

Configuration:
- Window size: 50 events
- Half-life: 24 hours
- Dependency threshold: 0.5
```

## Processing Instructions

Based on the examples above, process this new input through all four stages:

### Stage 1: Data Understanding
- Parse the XES file
- Extract all traces and event sequences
- Identify unique activities: register, verify, approve, reject, notify, complete
- Understand the temporal structure

### Stage 2: Inference (Sliding Window & Decay List)
- Initialize sliding window (size: 50)
- Process all events sequentially
- Since we have 5 traces with ~4 events each = ~20 events total, window is not full
- No evictions occur, so decay list remains empty
- Track window statistics

### Stage 3: Dataset Preparation
- Combine all events from window (no decay list events needed)
- Prepare PM4Py EventLog format
- Ensure traces are complete and ordered

### Stage 4: Reasoning & Output Generation
- Analyze patterns:
  - All cases start with register → verify
  - After verify: either approve (3 cases) or reject (2 cases)
  - After approve: either complete (2 cases) or notify → complete (1 case)
- Discover process model with XOR gateways
- Generate BPMN representation

## Expected Output (y)

Based on the examples, provide:

1. **BPMN Process Model**:
   ```
   Start → Register → Verify → XOR Gateway
     - Path 1: Reject → End (2 cases)
     - Path 2: Approve → XOR Gateway (3 cases)
       - Path 2a: Complete → End (2 cases)
       - Path 2b: Notify → Complete → End (1 case)
   ```

2. **XES Snapshot**: All events from the 5 traces (~20 events)

3. **Statistics**:
   - Window: ~20 events
   - Decay list: 0 events
   - Traces: 5
   - Activities: register, verify, approve, reject, notify, complete
   - Gateways: 2 XOR gateways

4. **Process Model Analysis**:
   - Sequential flow: register → verify (always)
   - Decision point 1: approve (60%) vs reject (40%)
   - Decision point 2: complete directly (67% of approved) vs notify first (33% of approved)

## Response Format

Follow the same structure as the examples:

```
## Stage 1: Data Understanding
[Analysis matching Example 1 format]

## Stage 2: Inference Results
[Results matching Example 1 format]

## Stage 3: Dataset Preparation
[Summary matching Example 1 format]

## Stage 4: Process Model Discovery
[Analysis matching Example 1 format]

## Output Files
- BPMN Model: [description]
- XES Snapshot: [description]
- Statistics: [summary]
```

## Instructions

Using the three examples as reference, process the new input X and generate output y. Apply the same reasoning patterns and processing logic demonstrated in the examples.


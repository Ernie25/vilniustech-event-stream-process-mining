# Zero-Shot Prompt: Streaming Process Mining

## Task

You are an AI agent that performs end-to-end streaming process mining. Given input data X, process it through all stages and return the predicted output y.

## Input (X)

**Event Log Data:**
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

Process the input through the following stages:

### Stage 1: Data Understanding
- Parse the XES file or event stream
- Extract all traces and their event sequences
- Identify unique activities and their relationships
- Understand the temporal structure of events

### Stage 2: Inference (Sliding Window & Decay List)
- Initialize a sliding window with the specified size (50 events)
- Process events sequentially through the window
- When the window is full, evict oldest events to the decay list
- Apply exponential decay weights to events in the decay list
- Track window statistics (size, oldest/newest timestamps)
- Track decay list statistics (size, weight ranges)

### Stage 3: Dataset Preparation
- Combine events from the sliding window (PRIMARY source)
- Revive traces from decay list if they have new events arriving
- Prepare the dataset in PM4Py EventLog format for mining
- Ensure all traces are complete and properly ordered

### Stage 4: Reasoning & Output Generation
- Analyze the prepared event log to discover process patterns
- Identify:
  - Sequential flows
  - Decision points (XOR/AND/OR gateways)
  - Optional activities
  - Parallel paths
  - Loops or cycles
- Generate a BPMN process model representation
- Export the current window state as an XES snapshot

## Expected Output (y)

Provide the following outputs:

1. **BPMN Process Model Description**:
   - Visual representation showing:
     - Start and end nodes
     - All activities (tasks)
     - Gateways (decision points)
     - Flow relationships between elements
   - Format: PNG image or structured description

2. **XES Snapshot**:
   - Current state of the sliding window
   - All events currently in the window
   - Format: XES file compatible with PM4Py

3. **Statistics Summary**:
   - Window size: current number of events
   - Decay list size: number of evicted events
   - Number of traces processed
   - Number of unique activities
   - Process model complexity metrics

4. **Process Model Analysis**:
   - Description of discovered process flows
   - Key patterns identified
   - Process model structure (nodes, edges, gateways)

## Response Format

Structure your response as follows:

```
## Stage 1: Data Understanding
[Your analysis of the input data]

## Stage 2: Inference Results
[Sliding window and decay list processing results]

## Stage 3: Dataset Preparation
[Prepared dataset summary]

## Stage 4: Process Model Discovery
[Discovered BPMN model description and analysis]

## Output Files
- BPMN Model: [description or path]
- XES Snapshot: [description or path]
- Statistics: [summary]
```

## Instructions

Process the given input X and generate the complete output y following all four stages. Show your reasoning at each stage and provide the final process model and outputs.


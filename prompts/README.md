# AI Agent Prompts for Streaming Process Mining

This directory contains prompt examples demonstrating how a single AI agent interacts with the streaming process mining system as an end-to-end solution.

## System Overview

The AI agent receives input data **X**, processes it through all stages of the pipeline, and returns the predicted or generated result **y**.

### Pipeline Stages

1. **Data Understanding**: Parse and understand event structure from XES files or event streams
2. **Inference**: Apply sliding window and decay list mechanisms to process events
3. **Reasoning**: Analyze process patterns and dependencies to discover the process model
4. **Output Generation**: Generate BPMN model visualization and XES snapshot

### Input (X)

- **XES Event Log File**: Standard XES format containing traces with events
- **Event Stream**: Individual events arriving sequentially with:
  - `case_id`: String identifier for the case/trace
  - `activity`: String name of the activity performed
  - `timestamp`: Datetime object (timezone-aware)
  - `resource`: Optional string identifier for the resource/person

### Output (y)

- **BPMN Process Model**: Visual representation of discovered process (PNG image)
- **XES Snapshot**: Current state of sliding window exported as XES file
- **Statistics**: Window size, decay list size, model activities, etc.

## Prompt Files

- **`prompt_zero_shot.md`**: Zero-shot example with a single new observation X requesting output y
- **`prompt_few_shot.md`**: Few-shot example with several known (X,y) pairs demonstrating the learning approach

## Usage

These prompts demonstrate how an AI agent would:
- Understand the input data structure
- Apply streaming processing mechanisms
- Reason about process patterns
- Generate structured outputs

The prompts can be used as templates for:
- Testing the AI agent system
- Demonstrating the end-to-end workflow
- Training or fine-tuning language models for process mining tasks


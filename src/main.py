from collections import deque

import pandas as pd
import pm4py
from pm4py.algo.evaluation.replay_fitness.variants import token_replay


def import_xes(file_path):
    event_log = pm4py.read_xes(file_path)
    start_activities = pm4py.get_start_activities(event_log)
    end_activities = pm4py.get_end_activities(event_log)
    print("Number of start activities: {}\nNumber of end activities: {}".format(start_activities, end_activities))
    return event_log


def export_dataframe_csv(file_path, event_log):
    df = pm4py.convert_to_dataframe(event_log)
    df.to_csv(file_path)

def trace_sliding_window(event_stream, window_size):
    current_window = deque(maxlen=window_size)
    image_index = 1
    image_path = 'path'
    decayed_traces = {}
    trace_weights = {}
    decay_factor = 0.9
    decay_threshold = 0.1

    print(event_stream)

    def find_trace(trace_id):
        for trace in current_window:
            if trace and trace[0]["case:concept:name"] == trace_id:
                return trace
        return None

    for event in event_stream:
        trace_id = event["case:concept:name"]

        trace_in_window = find_trace(trace_id)
        if trace_in_window:
            trace_in_window.append(event)
        else:
            if len(current_window) == window_size:
                oldest_trace = current_window.popleft()
                oldest_trace_id = oldest_trace[0]["case:concept:name"]
                decayed_traces[oldest_trace_id] = oldest_trace

            if trace_id in decayed_traces:
                recovered_trace = decayed_traces.pop(trace_id)
                current_window.append(recovered_trace)
                trace_weights[trace_id] = 1
            else:
                new_trace = [event]
                current_window.append(new_trace)
                trace_weights[trace_id] = 1

        apply_aging(current_window, trace_weights, decayed_traces, decay_factor, decay_threshold)

        # Once the window size is full, generate the model
        if len(current_window) == window_size:
            # Flatten the current window to get all events
            window_events = [event for trace in current_window for event in trace]
            window_df = pd.DataFrame(window_events)
            event_log = pm4py.convert_to_event_log(window_df)
            heuristics_net = discover_process_model_with_heuristics_miner(event_log)
            print(f"Added trace with event '{event['concept:name']}'")
            check_conformance(window_df)

            # print(f"Added trace with event '{event['concept:name']}'")
            pm4py.save_vis_heuristics_net(heuristics_net, f"{image_path}-trace-{image_index}.png")
            image_index += 1

    print(f"Processed {len(event_stream)} events in total, generated {image_index} process models.")


def check_conformance(data_frame):
    petri_net, initial_marking, final_marking = pm4py.discover_petri_net_heuristics(data_frame)

    fitness_alignments = pm4py.conformance.fitness_alignments(data_frame, petri_net, initial_marking, final_marking)
    fitness_token_based = pm4py.conformance.fitness_token_based_replay(data_frame, petri_net, initial_marking, final_marking)
    print(f"Fitness alignment: {fitness_alignments} Token Based fitness: {fitness_token_based}")
    precision = pm4py.conformance.precision_alignments(data_frame, petri_net, initial_marking, final_marking)
    print(f"Fitning token based: {fitness_token_based}")
    precision_token_based = pm4py.conformance.precision_alignments(data_frame, petri_net, initial_marking, final_marking)
    print(f"Token Based precision: {precision_token_based}")



def update_trace(trace, event):
    trace.append(event)

def apply_aging(window, trace_weights, decayed_traces, decay_factor, decay_threshold):
    print("Applying aging... {} = window length | {} = decayed length".format(len(window), len(decayed_traces)))
    for trace in window:
        case_id = trace[0]['case:concept:name']
        trace_weights[case_id] *= decay_factor  # Reduce weight of the trace

    # Move traces with very low weight to decayed_traces
    for trace in list(window):
        case_id = trace[0]['case:concept:name']
        if trace_weights[case_id] < decay_threshold:
            print(f"Decaying trace {case_id} due to low weight.")
            decayed_traces[case_id] = trace
            window.remove(trace)

    traces_to_remove = [trace_id for trace_id, weight in trace_weights.items() if weight < decay_threshold]
    for trace_id in traces_to_remove:
        if trace_id in decayed_traces:
            print(f"Removing permanently decayed trace {trace_id}.")
            del decayed_traces[trace_id]
            del trace_weights[trace_id]

def discover_process_model_with_heuristics_miner(event_log):
    heuristics_nets = pm4py.discover_heuristics_net(event_log)
    return heuristics_nets


if __name__ == '__main__':
    event_log = import_xes(r"path")
    print("Import completed")
    event_stream = pm4py.convert_to_dataframe(event_log).to_dict('records')

    window_size = 6

    trace_sliding_window(event_stream, window_size)


import numpy as np

states = {"s": 0, "E": 1, "5": 2, "I": 3, "e": 4}

state_transition_prob = np.array([
    [0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.9, 0.1, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.9, 0.1],
    [0.0, 0.0, 0.0, 0.0, 0.0]
])

emission_nuc_codes = {
    'A': 0,
    'C': 1,
    'G': 2,
    'T': 3
}

emission_probs = np.array([
    [0.00, 0.00, 0.00, 0.00],
    [0.25, 0.25, 0.25, 0.25],
    [0.05, 0.00, 0.95, 0.00],
    [0.40, 0.10, 0.10, 0.40],
    [0.00, 0.00, 0.00, 0.00]
])

query_sequence = "CTTCATGTGAAAGCAGACGTAAGTCA"

with np.errstate(divide='ignore'):
    log_state_transition_prob = np.log(state_transition_prob)
    log_emission_probs = np.log(emission_probs)

def calculate_prob_for_a_node(previous_column_values, current_state, emission_symbol):
    transition_log_probs = log_state_transition_prob[:, current_state]
    emission_log_prob = log_emission_probs[current_state, emission_nuc_codes[emission_symbol]]
    probabilities = previous_column_values + transition_log_probs + emission_log_prob
    max_prob = np.max(probabilities)
    best_prev_state = np.argmax(probabilities)
    return max_prob, best_prev_state

def viterbi_algorithm():
    n_states = len(states)
    seq_length = len(query_sequence)
    viterbi_value_matrix = np.full((n_states, seq_length), -np.inf)
    viterbi_trace_matrix = np.zeros((n_states, seq_length), dtype=int)
    
    initial_values = np.full(n_states, -np.inf)
    initial_values[states["s"]] = 0.0
    
    for i, symbol in enumerate(query_sequence):
        prev_col = initial_values if i == 0 else viterbi_value_matrix[:, i - 1]
        for current_state in range(n_states):
            max_prob, best_prev_state = calculate_prob_for_a_node(prev_col, current_state, symbol)
            viterbi_value_matrix[current_state, i] = max_prob
            viterbi_trace_matrix[current_state, i] = best_prev_state
            
    best_final_state = np.argmax(viterbi_value_matrix[:, -1])
    
    traceback_path = [best_final_state]
    current_state = best_final_state
    
    for i in range(seq_length - 1, 0, -1):
        current_state = viterbi_trace_matrix[current_state, i]
        traceback_path.append(current_state)
        
    traceback_path.reverse()
    
    id2state = {v: k for k, v in states.items()}
    hidden_state_sequence = [id2state[state] for state in traceback_path]
    
    print(viterbi_value_matrix)
    print(viterbi_trace_matrix)
    print("".join(hidden_state_sequence))

if __name__ == "__main__":
    viterbi_algorithm()

import math
ROUND_VALUE_3 = 3
ROUND_VALUE_6 = 6

# Inputs (x1, x2, x3, x4, x5)
x = [1.1, 2.4, 3.2, 5.1, 3.9]

# Desired Output
do = [0.52, 0.25, 0.75, 0.97]

# Learning rates
n_o = 0.3  # Learning rate for output layer
n_h = 0.1  # Learning rate for hidden layer

# Weights for hidden layer
wh1 = [0.31, 0.22, 0.33, -0.72, -0.85, 0.98]
wh2 = [-0.41, 0.55, -0.47, 0.31, 0.11, 0.37]
wh3 = [0.22, 0.16, -0.22, -0.15, 0.37, 0.84]

# Weights for output layer
wo1 = [0.17, 0.23, -0.31, 0.88]
wo2 = [0.53, 0.67, 0.92, 0.75]
wo3 = [0.11, -0.22, -0.18, 0.87]
wo4 = [0.75, 0.63, 0.66, 0.99]

array_wos = [wo1, wo2, wo3, wo4]
array_whs = [wh1, wh2, wh3]
print("\nSimulation")

print("\nHidden Layer")

# Function to calculate hidden layer outputs
def hidden_layer(inputs, weights, index):
    bias = weights[-1]
    output_expression = ""
    output_sum = 0
    for input_value, weight in zip(inputs, weights[:-1]):
        output_sum += input_value * weight
        output_expression += f"({input_value} * {weight}) + "
    
    result = output_sum - bias
    output_expression = output_expression.rstrip(" + ")
    print(f"F(v)_{index} = {output_expression} - {bias} = {round(result,ROUND_VALUE_6)}")
    
    return result

# Hidden layer outputs
hidden_output_1 = hidden_layer(x, wh1, 1)
hidden_output_2 = hidden_layer(x, wh2, 2)
hidden_output_3 = hidden_layer(x, wh3, 3)
print()

hidden_outputs = [hidden_output_1, hidden_output_2, hidden_output_3]

# Function to apply sigmoid activation
def sigmoid_array(outputs):
    sigmoid_values = []
    for index, output in enumerate(outputs, start=1):
        sigmoid_value = 1 / (1 + math.exp(-output))
        print(f"Y_h{index} = (1 / (1 + exp(-{round(output,ROUND_VALUE_3)})))")
        print(f"     = {round(sigmoid_value,ROUND_VALUE_6)}")
        sigmoid_values.append(sigmoid_value)
    return sigmoid_values

sigmoid_results = sigmoid_array(hidden_outputs)


print("\nOutput Layer")
# Function to calculate output layer values
def output_layer(Y_h, weights_o,index):
    bias = weights_o[-1]
    output_str = ""
    output = 0
    for yh, w in zip(Y_h, weights_o[:-1]):
        output += yh * w
        output_str += f"({round(yh,ROUND_VALUE_6)} * {w}) + "
    
    result = output - bias

    output_str = output_str.rstrip(" + ")
    print(f"Y_o{index} = {output_str} - {bias} = {round(result,ROUND_VALUE_6)}")
    
    final_output = 1 / (1 + math.exp(-result))
    print(f"     = (1 / (1 + exp({round(result,ROUND_VALUE_6)})))")
    print(f"     = {round(final_output,ROUND_VALUE_3)}")
    
    return final_output

# Calculate outputs for output layer neurons
output_wo1 = output_layer(sigmoid_results, wo1, 1)
output_wo2 = output_layer(sigmoid_results, wo2, 2)
output_wo3 = output_layer(sigmoid_results, wo3, 3)
output_wo4 = output_layer(sigmoid_results, wo4, 4)

output_values = [output_wo1,output_wo2,output_wo3,output_wo4]

# Error Computation Area
print("\nError Computation")
print("\nOutput Layer")

def calculate_desired_output(Y_os, do):
    result = []
    for i, (Y_o, T) in enumerate(zip(Y_os, do)):
        delta = Y_o * (1 - Y_o) * (T - Y_o)
        result.append(delta)
        
        print(f"S_o{i+1} = {round(Y_o, ROUND_VALUE_3)} * (1 - {round(Y_o, ROUND_VALUE_3)}) * ({round(T, ROUND_VALUE_3)} - {round(Y_o, ROUND_VALUE_3)})")
        print(f"     = {round(delta, ROUND_VALUE_3)}")
        
    return result

deltas = calculate_desired_output(output_values, do)

print("\nHidden Layer")
# Calculate hidden layer error deltas
def calculate_error_hidden(Y_hs, deltas, Wos):
    result = []
    for i, yh in enumerate(Y_hs):
        weighted_sum = sum(deltas[j] * Wos[j][i] for j in range(len(deltas)))
        delta = yh * (1 - yh) * weighted_sum
        result.append(delta)
        
        print(f"S_h{i+1} = {round(yh,ROUND_VALUE_6)} * (1 - {round(yh,ROUND_VALUE_3)}) * (", end="")
        value_terms = [f"{round(deltas[j],ROUND_VALUE_3)} * {Wos[j][i]}" for j in range(len(deltas))]
        print(" + ".join(value_terms))
        print(f"     = {round(delta, ROUND_VALUE_3)}")
    
    return result

# Calculate hidden layer error deltas
hidden_deltas = calculate_error_hidden(sigmoid_results, deltas, array_wos)

# Training data: (target, input1, input2)
classes = [
    (1, 1, 1),
    (-1, -1, -1),
    (1, 1, -1),
    (-1, -1, 1),
    (1, 0, -1),
    (-1, 0, 1)
]

# Initial values
threshold = 0
w1 = 0.7   # weight for input1
w2 = 0.15  # weight for input2
wb = 0.9   # bias (threshold)
learning_rate = 0.13
max_epochs = 4

epoch = 1

# Main loop for training the perceptron
while True:
    print(f"\nEpoch {epoch}:\n")

    # To track if weights have changed
    weight_changed = False
    step = 0

    for target, x1, x2 in classes:
        if x1 == 0 or x2 == 0:
            continue

        # Compute the perceptron output
        v_plus_b = round((x1 * w1) + (x2 * w2) - wb, 3)
        output = 1 if v_plus_b >= threshold else -1
        error = target - output

        # Print the current state with detailed formula
        step += 1
        print(f"Step {step}:")
        print(f"    x1, x2 = ({x1}, {x2}), target = {target}")
        print(f"    v+b = ({x1} * {w1}) + ({x2} * {w2}) - {wb} = {v_plus_b}")

        print(f"    Output (y) = {output}")
        print(f"    error = ({target}) - ({output}) = {error}")

        # Update weights and bias if there's an error
        if error != 0:
            new_w1 = round(w1 + learning_rate * error * x1, 3)
            new_w2 = round(w2 + learning_rate * error * x2, 3)
            new_wb = round(wb + learning_rate * error, 3)

            # Print how the new weights are calculated
            print(f"    Updating weights:\n")
            print(f"             w1    = {w1} + ({learning_rate} * {error} * {x1}) = {new_w1}")

            print(f"             w2    = {w2} + ({learning_rate} * {error} * {x2}) = {new_w2}")
          
            print(f"             w3    = {wb} + ({learning_rate} * {error}) = {new_wb}")

            # Update the weights and bias
            w1, w2, wb = new_w1, new_w2, new_wb
            weight_changed = True  # Mark that weights were updated

    # Stop if weights do not change or max_epochs reached
    if not weight_changed or epoch >= max_epochs:
        break

    epoch += 1
    print("---------------------------------------------------------")

print(f"\nTraining complete! Epoch limited to {max_epochs}.")

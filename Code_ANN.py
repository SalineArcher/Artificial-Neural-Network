import random
import csv
import math
import copy
import matplotlib.pyplot as plt


# FUNCTIONS
# Function to create Matrix of given size
def create_matrix(rows, cols):
    mat = []
    for i in range(rows):
        row_list = []
        for j in range(cols):
            row_list.append(0)
        mat.append(row_list)
    return mat


# Create random weight matrix
def weight_matrix(rows, cols):
    mat = []
    for i in range(rows):
        row_list = []
        for j in range(cols):
            row_list.append(random.uniform(-1, 1))
        mat.append(row_list)
    return mat


# Create bias for given matrix
def bias(mat, P):
    for p in range(P):
        row = mat.pop(p)
        row.insert(0, 1)
        mat.insert(p, row)
    return mat


# Normalising a vector
def normalising(mat):
    mini = min(mat)
    maxi = max(mat)
    for i in range(len(mat)):
        mat[i] = 0.1 + 0.8 * ((mat[i] - mini)/(maxi - mini))
    return mat


# Sigmoid
def sigmoid(x):
    ans = 1/(1 + math.exp(x))
    return ans


# Taking User input
L = int(input("Number of Inputs: "))
N = int(input("Number of Outputs: "))
P = int(input("Number of Training Patterns: "))
P_test = int(input("Number of Testing Patterns: "))
learning_rate = 0.9
momentum_coeff = 0.35
M = 25
training_data = open("Training_data.csv", "rt")
input_resource = csv.DictReader(training_data)

I = []
T = []
p = 0

# Taking input from CSV file
for row in input_resource:
    inputs = []
    outputs = []
    if p == P:
        break
    else:
        for i in range(1, L+1):
            inputs.append(float(row[f'input_{i}']))
            if i <= N:
                outputs.append(float(row[f'target_output_{i}']))
        I.append(inputs)
        T.append(outputs)
    p += 1

# Normalise input matrix
for i in range(L):
    inputs = []
    for p in range(P):
        inputs.append(I[p][i])
    inputs = normalising(inputs)

    for p in range(P):
        I[p][i] = inputs[p]

I = bias(I, P)

# Normalise output matrix
T_b4_norm = copy.deepcopy(T)
for k in range(N):
    outputs = []
    for p in range(P):
        outputs.append(T[p][k])
    outputs = normalising(outputs)
    for p in range(P):
        T[p][k] = outputs[p]

# Initialization of Matrices
V = weight_matrix(L+1, M)
W = weight_matrix(M+1, N)
delta_W = create_matrix(M+1, N)
delta_V = create_matrix(L+1, M)

MSE = 1
itr = []
err_MSE = []
iteration = 1

while MSE > 0.0001 and iteration < 5000:
    # Input of Hidden neurons
    IH = create_matrix(P, M)

    for p in range(P):
        for j in range(M):
            for i in range(L+1):
                IH[p][j] += I[p][i]*V[i][j]

    # Output of Hidden neurons
    OH = create_matrix(P, M)

    for p in range(P):
        for j in range(M):
            OH[p][j] = sigmoid(IH[p][j])

    OH = bias(OH, P)

    # Input of the Output neurons
    IO = create_matrix(P, N)

    for p in range(P):
        for k in range(N):
            for j in range(M+1):
                IO[p][k] += OH[p][j]*W[j][k]

    # Output of the Output neurons
    OO = create_matrix(P, N)

    for p in range(P):
        for k in range(N):
            OO[p][k] = sigmoid(-IO[p][k])

    # Denormalize output
    predicted_output = create_matrix(P, N)
    min_and_max = []
    for k in range(N):
        outputs = []
        for p in range(P):
            outputs.append(T_b4_norm[p][k])
        mini = min(outputs)
        maxi = max(outputs)
        min_and_max.append([mini, maxi])

    for p in range(P):
        for k in range(N):
            predicted_output[p][k] = min_and_max[k][0]+((OO[p][k]-0.1)/(0.9-0.1))*(min_and_max[k][1]-min_and_max[k][0])

    # Error for each neuron
    error_neuron = []
    for p in range(P):
        error = []
        for k in range(N):
            error.append((1/2)*((T[p][k]-OO[p][k])**2))
        error_neuron.append(error)

    # Error for each pattern for all neurons
    error_average = []
    for p in range(P):
        error = 0
        for k in range(N):
            error += error_neuron[p][k]
        error_average.append(error)

    sum_of_error = 0
    for p in range(P):
        sum_of_error += error_average[p]

    MSE = sum_of_error/P

    # Back propagation algorithm
    for k in range(N):
        for j in range(M+1):
            sum = 0
            for p in range(P):
                sum += (T[p][k] - OO[p][k]) * OO[p][k] * (1 - OO[p][k]) * OH[p][j]
            if iteration == 1:
                delta_W[j][k] = (learning_rate / P) * sum
            else:
                delta_W[j][k] = ((learning_rate / P) * sum)+(momentum_coeff * delta_W[j][k])

    # Updating W values
    for j in range(M+1):
        for k in range(N):
            W[j][k] += delta_W[j][k]

    for i in range(L+1):
        for j in range(M):
            sum1 = 0
            for p in range(P):
                for k in range(N):
                    sum1 += (T[p][k]-OO[p][k])*(OO[p][k])*(1-OO[p][k])*W[j][k]*OH[p][j]*(1-OH[p][j])*I[p][i]
            if iteration == 1:
                delta_V[i][j] = (learning_rate/(N*P)) * sum1
            else:
                delta_V[i][j] = ((learning_rate / (N * P)) * sum1) + (momentum_coeff * delta_V[i][j])

    # Updating V values
    for i in range(L+1):
        for j in range(M):
            V[i][j] += delta_V[i][j]

    print("\n=================================================================================================")
    print(f"1. M - {M}, Iteration - {iteration}, MSE - {MSE}")
    print(f"2. Output of the network \n {predicted_output}")
    print(f"3. Error in prediction \n {error_neuron}")
    print("\n=================================================================================================")
    iteration += 1
    itr.append(iteration)
    err_MSE.append(MSE)

# To plot MSE vs Iterations
plt.plot(itr, err_MSE)
plt.xlabel('Iterations')
plt.ylabel('MSE')
plt.show()

# Testing Outputs
test_data = open("Testing_data.csv", "rt")
test_input = csv.DictReader(test_data)


I = []
T = []
P = P_test

# Taking input from CSV file
for row in test_input:
    inputs = []
    outputs = []
    if p == P:
        break
    else:
        for i in range(1, L+1):
            inputs.append(float(row[f'input_{i}']))
            if i <= N:
                outputs.append(float(row[f'target_output_{i}']))
        I.append(inputs)
        T.append(outputs)
    p += 1

# Normalise input matrix
for i in range(L):
    inputs = []
    for p in range(P):
        inputs.append(I[p][i])
    inputs = normalising(inputs)

    for p in range(P):
        I[p][i] = inputs[p]

I = bias(I, P)

# Normalise output matrix
T_b4_norm = copy.deepcopy(T)
for k in range(N):
    outputs = []
    for p in range(P):
        outputs.append(T[p][k])
    outputs = normalising(outputs)
    for p in range(P):
        T[p][k] = outputs[p]

 # Input of Hidden neurons
    IH = create_matrix(P, M)

    for p in range(P):
        for j in range(M):
            for i in range(L+1):
                IH[p][j] += I[p][i]*V[i][j]

    # Output of Hidden neurons
    OH = create_matrix(P, M)

    for p in range(P):
        for j in range(M):
            OH[p][j] = sigmoid(IH[p][j])

    OH = bias(OH, P)

    # Input of the Output neurons
    IO = create_matrix(P, N)

    for p in range(P):
        for k in range(N):
            for j in range(M+1):
                IO[p][k] += OH[p][j]*W[j][k]

    # Output of the Output neurons
    OO = create_matrix(P, N)

    for p in range(P):
        for k in range(N):
            OO[p][k] = sigmoid(-IO[p][k])

    # Denormalize output
    predicted_output = create_matrix(P, N)
    min_and_max = []
    for k in range(N):
        outputs = []
        for p in range(P):
            outputs.append(T_b4_norm[p][k])
        mini = min(outputs)
        maxi = max(outputs)
        min_and_max.append([mini, maxi])

    for p in range(P):
        for k in range(N):
            predicted_output[p][k] = min_and_max[k][0]+((OO[p][k]-0.1)/(0.9-0.1))*(min_and_max[k][1]-min_and_max[k][0])

    # Error for each neuron
    error_neuron = []
    for p in range(P):
        error = []
        for k in range(N):
            error.append((1/2)*((T[p][k]-OO[p][k])**2))
        error_neuron.append(error)

    # Error for each pattern for all neurons
    error_average = []
    for p in range(P):
        error = 0
        for k in range(N):
            error += error_neuron[p][k]
        error_average.append(error)

    sum_of_error = 0
    for p in range(P):
        sum_of_error += error_average[p]

    MSE = sum_of_error/P

    print("\n=================================================================================================")
    print(f"1. Test Pattern - {P}, MSE - {MSE}")
    print(f"2. Output of the network \n {predicted_output}")
    print(f"3. Error in prediction \n {error_neuron}")
    print("\n=================================================================================================")

#Import packages and load up an easy sklearn dataset to play around with
import numpy as np
from numba import cuda
import math
from sklearn.datasets import load_diabetes

x, y = load_diabetes(return_X_y=True, as_frame=False)

print(f'Features shape: {x.shape}')
print(f'Range of features: {x.min()} - {x.max()}')
print(f'Labels shape: {y.shape}')
print(f'Range of labels: {y.min()} - {y.max()}')

#CPU versions of some basic pre-processing, loop back to GPU implementations once the basics are done. 
def min_max_scaler(x):
    x_scaled = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
    return x_scaled

def std_scaler(x):
    x_scaled = (x - x.mean(axis=0)) / x.std(axis=0)
    return x_scaled

#Perform batching on the CPU dataset
def create_batches(x, y, batch_size=64):

    if y.ndim == 1:
        y = y.reshape((y.shape[0], 1))
    else:
        pass

    pad_length = batch_size - (x.shape[0] % batch_size)
    num_batches = (x.shape[0] + pad_length) // batch_size

    x_pad_array = np.zeros((pad_length, x.shape[1]))
    x_padded = np.concatenate((x, x_pad_array), axis=0)
    x_batched = x_padded.reshape((num_batches, batch_size, x_padded.shape[1]))

    y_pad_array = np.zeros((pad_length, y.shape[1]))
    y_padded = np.concatenate((y, y_pad_array), axis=0)
    y_batched = y_padded.reshape((num_batches, batch_size, y_padded.shape[1]))

    return x_batched, y_batched

x_scaled = std_scaler(x)
y_scaled = std_scaler(y)

batch_size = 64
x_batched, y_batched = create_batches(x_scaled, y_scaled, batch_size=batch_size)

#Defining a class to hold weights and biases for linear layers on the GPU - when instantiated, creates new objects which hold weight and bias matrices of appropriate shape on GPU memory 
class NumbaLinearLayer():
    def __init__(self, batch_size, input_dim, output_dim):

        self.gpu_weights = cuda.device_array_like(np.random.normal(0, 1, (input_dim, output_dim)).astype(np.float64))
        self.gpu_bias = cuda.device_array_like(np.random.normal(0, 1, (batch_size, output_dim)).astype(np.float64))

#Standard CUDA matmul kernel implementation 
@cuda.jit
def matmul_kernel(A, B, C):
    #Get thread indices - at a lower level, .grid implements:
    #row = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    #col = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    row, col = cuda.grid(2)

    #Check bounds of indices:
    if row < C.shape[0] and col < C.shape[1]:
        #Perform computation by incrementing the 0 vals implemented in the output location when the kernel was called:
        tmp = 0.0
        for i in range(A.shape[1]): #Walks down the cols of A, which is equal to the rows of B, keeping the proper positions on the outside for output
            tmp += A[row, i] * B[i, col]
        C[row, col] = tmp

#Instantiate linear layer on GPU - stores weights and biases of appropriate shapes on GPU
linear_layer = NumbaLinearLayer(batch_size, 10, 32)
#Allocates device space for linear layer's output - matches the num_batches and batch_size of inputs, with final dimension matching output of linear layer
layer_output = cuda.device_array_like(np.zeros((x_batched.shape[0], batch_size, linear_layer.gpu_weights.shape[1])))

#Passes through a single "epoch" of matmuls on the layer over the CPU input tensor
for i in range (x_batched.shape[0]):
    #Take batch from CPU tensor and move to device
    batch = cuda.to_device(x_batched[i, :, :])
    #Allocate memory for output from batch operations
    batch_output = cuda.device_array_like(np.zeros((batch.shape[0], 32)))

    #Sets parameters to invoke numba kernel
    threads_per_block = (16,16)
    blocks_per_grid_x = math.ceil(batch_output.shape[0] / threads_per_block[0])
    blocks_per_grid_y = math.ceil(batch_output.shape[1] / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    #Performs matrix multiplication with inputs A and B as batch and layer weights, and output C as the space allocated earlier for batch outputs
    matmul_kernel[threads_per_block, blocks_per_grid](batch, linear_layer.gpu_weights, batch_output)

    #Drops the results of the matmul, stored on device, to the corresponding batch location on the pre-allocated layer output tensor
    layer_output[i, :, :] = batch_output

layer_output_cpu = layer_output.copy_to_host()




linear_1 = NumbaLinearLayer(batch_size, 10, 32)
linear_2 = NumbaLinearLayer(batch_size, 32, 64)
linear_3 = NumbaLinearLayer(batch_size, 64, 32)
linear_4 = NumbaLinearLayer(batch_size, 32, 16)
output_layer = NumbaLinearLayer(batch_size, 16, 1)

#Allocate memory of proper shape for target tensor:
pred_outputs = cuda.device_array_like(np.zeros(y_batched.shape))

#Perform the passes on the input CPU tensor
for i in range(num_batches):
    #Move batch to device
    batch = cuda.to_device(x_batched[i, :, :])

    #Allocate space for outputs of all layer passes in advance - carries same batch size consistently the whole time!
    x_1 = cuda.device_array_like(np.zeros((batch_size, linear_1.gpu_weights.shape[1])))
    x_2 = cuda.device_array_like(np.zeros((batch_size, linear_2.gpu_weights.shape[1])))
    x_3 = cuda.device_array_like(np.zeros((batch_size, linear_3.gpu_weights.shape[1])))
    x_4 = cuda.device_array_like(np.zeros((batch_size, linear_4.gpu_weights.shape[1])))
    batch_output = cuda.device_array_like(np.zeros((batch_size, output_layer.gpu_weights.shape[1])))

    #PERFORM LAYER PASSES BY INVOKING AND RUNNING KERNELS SEQUENTIALLY:
    #First layer pass - takes batch, applies linear_1 weights, results go into x_1
    threads_per_block = (2,2)
    blocks_per_grid_x = math.ceil(x_1.shape[0] / threads_per_block[0]) # 64 / 2 = 32
    blocks_per_grid_y = math.ceil(x_1.shape[1] / threads_per_block[1]) # 32 / 2 =16
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y) #(32,16)
    matmul_kernel[blocks_per_grid, threads_per_block](batch, linear_1.gpu_weights, x_1)
    #Second layer pass - takes x_1, applies linear_2 weights, results go into x_2
    threads_per_block = (2,2)
    blocks_per_grid_x = math.ceil(x_2.shape[0] / threads_per_block[0]) # 64 / 2 = 32
    blocks_per_grid_y = math.ceil(x_2.shape[1] / threads_per_block[1]) # 64 / 2 = 32
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y) # (32,32)
    matmul_kernel[blocks_per_grid, threads_per_block](x_1, linear_2.gpu_weights, x_2)
    #Third layer
    threads_per_block = (2,2)
    blocks_per_grid_x = math.ceil(x_3.shape[0] / threads_per_block[0]) # 64 / 2 = 16
    blocks_per_grid_y = math.ceil(x_3.shape[1] / threads_per_block[1]) # 32 / 2 = 8
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y) # (16,8)
    matmul_kernel[blocks_per_grid, threads_per_block](x_2, linear_3.gpu_weights, x_3)
    #Fourth layer
    threads_per_block = (2,2)
    blocks_per_grid_x = math.ceil(x_4.shape[0] / threads_per_block[0]) # 64 / 2 = 32
    blocks_per_grid_y = math.ceil(x_4.shape[1] / threads_per_block[1])  #16 / 2 = 8
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y) # (32,8)
    matmul_kernel[blocks_per_grid_x, threads_per_block](x_3, linear_4.gpu_weights, x_4)
    #Output layer
    threads_per_block = (2, 1)
    blocks_per_grid_x = math.ceil(batch_output.shape[0] / threads_per_block[0]) # 64 / 2 = 32
    blocks_per_grid_y = math.ceil(batch_output.shape[1] / threads_per_block[1]) # 1 / 1 = 1
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y) #(32,1)
    matmul_kernel[blocks_per_grid, threads_per_block](x_4, output_layer.gpu_weights, batch_output)

    #Now we assign the final batch output to our destination tensor on the GPU
    pred_outputs[i, :, :] = batch_output

#The epoch passes with all operations taking place on GPU - at the end, result is consolidated to CPU
pred_outputs_cpu = pred_outputs.copy_to_host()


#Okay, now the basic flow of a dense forward pass is done - now, I'll calculate MSE on the results:
@cuda.jit
def mse_se_kernel(y_pred, y_true, errors):
    x = cuda.grid(1)
    if x < errors.shape[0]:
        errors[x] = math.pow((y_pred[x] - y_true[x]), 2)
@cuda.jit
def scalar_division(input_array, scalar, output_array):
    x = cuda.grid(1)
    if x < output_array.shape[0]:
        output_array[x] = input_array[x] / scalar

@cuda.jit
def mse_reduce_sum_kernel(input_array, output_array):

    #First, declare the shared memory array - make sure it matches size of blocks passed to kernel:
    shared_mem = cuda.shared.array(shape = (256,), dtype=np.float64)

    #Then, get IDs for thread and block we're in, as well as global index:
    tid = cuda.threadIdx.x #Which thread in block
    bid = cuda.blockIdx.x #Which block in grid
    idx = bid * cuda.blockDim.x + tid #Which thread index in the global input array

    #Once we have the IDs, we can move every thread in the block from the global memory to shared memory in parallel
    #First we check bounds:
    if idx < input_array.shape[0]:
        shared_mem[tid] = input_array[idx]
    else:
        shared_mem[tid] = 0.0

    #Synchronize to ensure that all threads have written to shared memory:
    cuda.syncthreads()

    #Now we can implement the reduction algorithm - this algorithm processes a one dimension tensor in O(nlogn) time in parallel:
    s = cuda.blockDim.x // 2
    while s > 0:
        #Only the first s threads in each block do work:
        if tid < s:
            shared_mem[tid] += shared_mem[tid + s]

        #Synchronize threads to ensure they are done:
        cuda.syncthreads()

        #Divide stride by 2 for next iteration:
        s //= 2

    #At the end of the loop, take the resulting sum from thread 0 and set it to the output block in global memory:
    if tid == 0:
        output_array[bid] = shared_mem[0]

mse_errors_gpu = cuda.device_array(y_batched.shape[0], dtype=np.float64)
def calculate_mse(y_pred_batched, y_true_batched):
    #Iterate over batches for prediction and target
    for i in range(y_pred_batched.shape[0]):
        #Move prediction and target to device
        y_pred = cuda.to_device(y_pred_batched[i, :, :].squeeze())
        y_true = cuda.to_device(y_true_batched[i, :, :].squeeze())

        n_elements = y_pred.shape[0]
        #Allocate memory for SE kernel output
        errors = cuda.device_array_like(np.zeros((y_pred.shape)))
        #Instantiate and run the SE kernel on batch
        threads_per_block = 4
        blocks_per_grid = (n_elements + threads_per_block - 1) // threads_per_block
        mse_se_kernel[blocks_per_grid, threads_per_block](y_pred, y_true, errors)

        #Allocate memory for the output from reduce sum
        partial_sums = cuda.device_array_like(np.zeros((blocks_per_grid,)))
        #Instantiate and run the reduce sum kernel on outputs from SE kernel, accepting one returned value per block of threads:
        mse_reduce_sum_kernel[blocks_per_grid, threads_per_block](errors, partial_sums)

        #If we have multiple blocks across the input matrix, we need to perform another reduction:
        final_sum = cuda.device_array(1, dtype=np.float64)
        if blocks_per_grid > 1:
            #Configure smaller grid for second reduction
            threads_for_reduction = min(blocks_per_grid, threads_per_block)
            blocks_for_reduction = 1
            #Instantiate and run kernel for second reduction
            mse_reduce_sum_kernel[blocks_for_reduction, threads_for_reduction](partial_sums, final_sum)
        else:
            #Only one block
            final_sum = partial_sums

        #Allocate device memory to keep final division on device
        batch_mse = cuda.device_array_like(final_sum)
        scalar_division[1,1](final_sum, n_elements, batch_mse)
        mse_errors_gpu[i] = batch_mse[0]

calculate_mse(pred_outputs_cpu, y_batched)

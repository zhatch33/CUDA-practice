'''This file contains the things I learned in the forward_linear_scratch.py file, cleaned up and stitched together into a coherent forward pass through a set of dense linear layers, 
complete with per-batch loss calculation at the end of each pass. My next step will be to implement the back-propagation at the end of this "epoch" pass over the batches of inputs.''' 

import numpy as np
from numba import cuda
import math
from sklearn.datasets import load_diabetes

#--------DECLARE CUDA KERNELS AND CLASSES------
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


#--------IMPORT AND PREP DATA------------------
x, y = load_diabetes(return_X_y=True, as_frame=False)

def min_max_scaler(x):
    x_scaled = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
    return x_scaled

def std_scaler(x):
    x_scaled = (x - x.mean(axis=0)) / x.std(axis=0)
    return x_scaled

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
num_batches = x_batched.shape[0]

#----------RUN FORWARD PASS COLLECTING BATCH LOSSES ON GPU-------------
#Allocate tensor to store batch losses on GPU
batch_losses = cuda.device_array((num_batches, 1))

#Initialize layers for forward pass
layer_1 = NumbaLinearLayer(batch_size, x_batched.shape[2], 32)
layer_2 = NumbaLinearLayer(batch_size, 32, 64)
layer_3 = NumbaLinearLayer(batch_size, 64, 32)
output_layer = NumbaLinearLayer(batch_size, 32, 1)

for i in range(num_batches):
    #Move input and target to the device:
    x = cuda.to_device(x_batched[i, :, :])
    y = cuda.to_device(y_batched[i, :, :])
    #Allocate space for layer outputs - these will be reset between batches
    x_1 = cuda.device_array((batch_size, 32))
    x_2 = cuda.device_array((batch_size, 64))
    x_3 = cuda.device_array((batch_size, 32))
    y_pred = cuda.device_array((batch_size, 1))

    #Perform layer passes
    threadsperblock = (4,4)
    blockspergrid_x = math.ceil(x_1.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(x_1.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    matmul_kernel[blockspergrid, threadsperblock](x, layer_1.gpu_weights, x_1)

    threadsperblock = (4,4)
    blockspergrid_x = math.ceil(x_2.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(x_2.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    matmul_kernel[blockspergrid, threadsperblock](x_1, layer_2.gpu_weights, x_2)

    threadsperblock = (4,4)
    blockpergrid_x = math.ceil(x_3.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(x_3.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    matmul_kernel[blockspergrid, threadsperblock](x_2, layer_3.gpu_weights, x_3)

    threadsperblock = (4,1)
    blockspergrid_x = math.ceil(y_pred.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(y_pred.shape[1] / threadsperblock[1])
    matmul_kernel[blockspergrid, threadsperblock](x_3, output_layer.gpu_weights, y_pred)

    #Now, y_pred and y should both be GPU tensors of shape (batch_size, 1) when we begin calculating MSE - they should be squeezed for the MSE kernels
    errors = cuda.device_array((batch_size))
    #I'll simplify things by having every batch instance take up a whole thread in a single block grid
    threadsperblock = batch_size
    blockspergrid = 1
    mse_se_kernel[blockspergrid, threadsperblock](y_pred.squeeze(), y.squeeze(), errors)

    #Now we need to reduce the sum of the errors - this is where having a single block per grid is helpful for returning a single value
    #First allocate memory for this batch's sum SE, then calculate in kernel
    sum_SE = cuda.device_array((1))
    mse_reduce_sum_kernel[blockspergrid, threadsperblock](errors, sum_SE)
    #Now perform scalar division
    batch_mse = cuda.device_array_like(sum_SE)
    scalar_division[blockspergrid, threadsperblock](sum_SE, batch_size, batch_mse)

    batch_losses[i, :] = batch_mse[0]

batch_losses_cpu = batch_losses.copy_to_host() #Copy to CPU to inspect forward pass results

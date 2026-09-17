#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // Performs one Hillis-Steele step (offset i) over the whole array.
        __global__ void naiveScan(int n, int i, int* odata, const int* idata) {
            int index = threadIdx.x + (blockDim.x * blockIdx.x);

            if (index >= n)
                return;

            if (index >= i) {
                odata[index] = idata[index] + idata[index - i];
            }
            else {
                odata[index] = idata[index];
            }
        }

        // Does a scan within a block, then records the total of that block to blockSums array
        __global__ void blockScan(int n, int* odata, const int* idata, int* blockSums) {
            extern __shared__ int temp[]; // sized 2 * blockDim.x ints by the launch

            int tid = threadIdx.x;
            int index = threadIdx.x + (blockDim.x * blockIdx.x);

            // ping pong buffer toggles (like in GPU gems)
            int pout = 0, pin = 1;

            // Treat each block as a separate scan, so data goes
            // 0, blockElement[0], blockElement[1], etc...
            // assuming blockElement is a subarray of idata ranging from
            // (blockIdx.x * blockDim.x) to ((blockIdx.x + 1) * blockDim.x - 1)
            // NOTE: If the index is outside idata, fill with 0
            temp[pout * blockDim.x + tid] =
                (tid > 0 && index - 1 < n) ? idata[index - 1] : 0;
            __syncthreads();

            // ceil(log2(blockDim.x)) iterations, all local to this block.
            for (int d = 1; d < blockDim.x; d *= 2) {

                pout = 1 - pout;
                pin = 1 - pout;

                if (tid >= d) {
                    // add previous values (Hillis Steele step)
                    temp[pout * blockDim.x + tid] = 
                        temp[pin * blockDim.x + tid] + temp[pin * blockDim.x + tid - d];
                }
                else {
                    // copy correct value to output
                    temp[pout * blockDim.x + tid] = temp[pin * blockDim.x + tid];
                }
                __syncthreads();
            }


            if (index < n) {
                odata[index] = temp[pout * blockDim.x + tid];
            }

            // when we reach the last thread in the block, then compute the block sum
            if (tid == blockDim.x - 1) {
                // if we are past end of the data, make this a zero, else, take current value
                int ownValue = (index < n) ? idata[index] : 0;
                // block sum equals final element of shared memory plus current value 
                // (doesn't get included in exclusive scan)
                blockSums[blockIdx.x] = temp[pout * blockDim.x + tid] + ownValue;
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            unsigned blockSize = Common::getBlockSize();
            unsigned gridSize = Common::divup(n, blockSize);

            int* dev_odata;
            int* dev_idata;

            cudaMalloc((void**)&dev_odata, sizeof(int) * n);
            checkCUDAErrorFn("cudaMalloc dev_odata failed!");
            cudaMalloc((void**)&dev_idata, sizeof(int) * n);
            checkCUDAErrorFn("cudaMalloc dev_idata failed!");

            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAErrorFn("cudaMemcpy failed!");

            // set odata to 0, idata[0], idata[1]... idata[n - 1]
            cudaMemset(dev_odata, 0, sizeof(int));
            cudaMemcpy(dev_odata + 1, dev_idata, sizeof(int) * (n - 1), cudaMemcpyDeviceToDevice);
            checkCUDAErrorFn("cudaMemcpy shift failed!");

            timer().startGpuTimer();
            // This is a mechanism for doing ping pong buffers
            int pout = 0, pin = 1;
            int* buf[2] = { dev_odata, dev_idata };

            // ceil(log2n) iterations
            for (int i = 1; i < n; i *= 2) {
                pout = 1 - pout;
                pin = 1 - pout;

                naiveScan<<<gridSize, blockSize>>>(n, i, buf[pout], buf[pin]);
                checkCUDAErrorFn("naiveScan failed!");
            }
            timer().endGpuTimer();

            cudaMemcpy(odata, buf[pout], sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("cudaMemcpy failed!");

            cudaFree(dev_odata);
            cudaFree(dev_idata);
        }

        // Applies offsets we found in preceding blocks to odata
        __global__ void addBlockOffsets(int n, int* odata, const int* blockOffsets) {
            int index = threadIdx.x + (blockDim.x * blockIdx.x);

            if (index >= n)
                return;

            // blockIdx.x is the index of our block, so we add the offset we collected
            odata[index] += blockOffsets[blockIdx.x];
        }

        // In the non-shared memory version, we access global memory, since we have the entire array
        // available in the kernel. If we want to use shared memory, however, we cannot access all of the
        // work that every block has done. Therefore, each block gets a subset of the input data,
        // and also keeps track of a total per block. We then run a scan on block totals, and add them to result.
        void scanSharedMemDevice(int n, int* dev_odata, const int* dev_idata, int*& arena) {
            unsigned blockSize = Common::getBlockSize();
            unsigned gridSize = Common::divup(n, blockSize);

            // use current 
            int* dev_blockSums = arena;
            // advance to next "level" of scratch array
            arena += gridSize;

            blockScan<<<gridSize, blockSize, 2 * blockSize * sizeof(int)>>>(
                n, dev_odata, dev_idata, dev_blockSums);
            checkCUDAErrorFn("blockScan failed!");

            if (gridSize > 1) {
                // use current scratch array 
                int* dev_blockOffsets = arena;
                // advance to next "level" of scratch array
                arena += gridSize;

                scanSharedMemDevice(gridSize, dev_blockOffsets, dev_blockSums, arena);

                addBlockOffsets<<<gridSize, blockSize>>>(n, dev_odata, dev_blockOffsets);
                checkCUDAErrorFn("addBlockOffsets failed!");
            }
        }

        void scanSharedMem(int n, int* odata, const int* idata) {
            int* dev_odata;
            int* dev_idata;

            cudaMalloc((void**)&dev_odata, sizeof(int) * n);
            checkCUDAErrorFn("cudaMalloc dev_odata failed!");
            cudaMalloc((void**)&dev_idata, sizeof(int) * n);
            checkCUDAErrorFn("cudaMalloc dev_idata failed!");

            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAErrorFn("cudaMemcpy failed!");

            // this "scratch" array contains the blockOffsets for each
            // level of recursion in the scanSharedMemDevice function
            // (only relevant when the number of blocks is enough
            // that doing a scan on the block totals takes more than one block).
            // If we simply allocate a blockOffset array for each recursion in scanSharedMemDevice,
            // it causes large performance issues that make the non-shared memory version faster.
            unsigned blockSize = Common::getBlockSize();
            size_t scratchSize = 0;
            unsigned levelGrid = Common::divup((unsigned)n, blockSize);
            while (true) {
                scratchSize += levelGrid;
                if (levelGrid <= 1) break;
                scratchSize += levelGrid;
                levelGrid = Common::divup(levelGrid, blockSize);
            }

            int* dev_scratch;
            cudaMalloc((void**)&dev_scratch, sizeof(int) * scratchSize);
            checkCUDAErrorFn("cudaMalloc dev_scratch failed!");

            timer().startGpuTimer();
            // the scratch array is passed in here so it appears in recursive calls
            int* arena = dev_scratch;
            scanSharedMemDevice(n, dev_odata, dev_idata, arena);
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("cudaMemcpy failed!");

            cudaFree(dev_odata);
            cudaFree(dev_idata);
            cudaFree(dev_scratch);
        }
    }
}

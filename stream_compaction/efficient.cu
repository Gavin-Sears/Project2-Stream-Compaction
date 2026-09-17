#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define USE_CONFLICT_FREE_OFFSETS 1

// Number of shared memory bannks, and log2 of that value
// These never get changed, log2 value is so we can bit shift
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5

// When we access memory, the index mod 32 is the "bank" value.
// If all bank values are different, then it gets processed at the same time (faster).
// If they are the same, it is slower, since they all happen one after the other.
// Now, when you activate USE_CONFLICT_FREE_OFFSETS, it changes the definition of the
// macro below. By default, this macro will give you 0. 
// When using conflict free offsets, it will divide the current index value by 32 (the bitshift).
// This offset gets added to the index, so that if you for example access index 32 and 64 at the same time, the
// offset will actually add 1 to the 64 index so that the two accesses are not in the same bank. (32 and 65)
#if USE_CONFLICT_FREE_OFFSETS
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#else
#define CONFLICT_FREE_OFFSET(n) (0)
#endif

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void upSweep(int n, int i, int* odata, const int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);

            if (index >= n)
                return;

            odata[index] = idata[index];
            
            if ((index + 1) % (i * 2) == 0) {
                odata[index] += idata[index - i];
            }
        }

        // Compact upsweep implementation that has a number of threads equal to the 
        // number of elements we are changing. As such we can avoid modulo to check
        // if the thread is active, and also don't need ping pong buffers.
        __global__ void upSweepCompact(int activeCount, int i, int* data) {
            int t = threadIdx.x + blockIdx.x * blockDim.x;
            if (t >= activeCount)
                return;

            int index = (t + 1) * i * 2 - 1;
            data[index] += data[index - i];
        }

        // Compacted down-sweep, same idea as upSweepCompact above.
        __global__ void downSweepCompact(int activeCount, int i, int* data) {
            int t = threadIdx.x + blockIdx.x * blockDim.x;
            if (t >= activeCount)
                return;

            int index = (t + 1) * i * 2 - 1;
            int lNode = data[index - i];
            int rNode = data[index];

            data[index - i] = rNode;
            data[index] = lNode + rNode;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // calculates next power of two greater than n
            int n_pad = n;
            if ((n & (n - 1)) != 0) {
                n_pad = 1;
                while (n_pad < n) n_pad <<= 1;
            }

            unsigned blockSize = Common::getBlockSize();

            int* dev_data;

            // allocates power of two array, since upsweep only works with power of two
            cudaMalloc((void**)&dev_data, sizeof(int) * n_pad);
            checkCUDAErrorFn("cudaMalloc dev_data failed!");

            cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAErrorFn("cudaMemcpy failed!");

            int padding_elements = n_pad - n;
            if (padding_elements > 0) {
                cudaMemset(dev_data + n, 0, sizeof(int) * padding_elements);
                checkCUDAErrorFn("cudaMemset failed!");
            }

            timer().startGpuTimer();

            // The gridsize gets recalculated every i
            // to only reflect the active threads (which equals the number of elements 
            // that are being changed) on that iteration. This means we can avoid modulo
            // when checking threads in the kernel, and it also means no ping pong buffers.
            for (int i = 1; i < n_pad; i *= 2) {
                int activeCount = n_pad / (i * 2);
                unsigned levelGridSize = Common::divup((unsigned)activeCount, blockSize);
                upSweepCompact<<<levelGridSize, blockSize>>>(activeCount, i, dev_data);
                checkCUDAErrorFn("upSweepCompact failed!");
            }

            // set last element to 0
            cudaMemset(dev_data + n_pad - 1, 0, sizeof(int));
            checkCUDAErrorFn("cudaMemset failed!");

            for (int i = n_pad / 2; i > 0; i /= 2) {
                int activeCount = n_pad / (i * 2);
                unsigned levelGridSize = Common::divup((unsigned)activeCount, blockSize);
                downSweepCompact<<<levelGridSize, blockSize>>>(activeCount, i, dev_data);
                checkCUDAErrorFn("downSweepCompact failed!");
            }
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("cudaMemcpy failed!");

            cudaFree(dev_data);
        }

        // Runs upsweep locally on one block's shared memory buffer.
        __device__ void upSweepSharedMem(int* temp, int tid, int blockElements) {
            // Loop and increase stride exponentially
            for (int s = 1; s < blockElements; s <<= 1) {
                int activeCount = blockElements / (s * 2);
                if (tid < activeCount) {
                    int index = (tid + 1) * s * 2 - 1;
                    int a = index;
                    int b = index - s;
                    // Conflict free offset adds a value to a depending on result of int division by 32
                    temp[a + CONFLICT_FREE_OFFSET(a)] += temp[b + CONFLICT_FREE_OFFSET(b)];
                }

                __syncthreads();
            }
        }

        // Runs downsweep locally on one block's shared memory buffer.
        __device__ void downSweepSharedMem(int* temp, int tid, int blockElements) {
            for (int s = blockElements / 2; s > 0; s >>= 1) {
                int activeCount = blockElements / (s * 2);
                if (tid < activeCount) {
                    int index = (tid + 1) * s * 2 - 1;
                    int a = index;
                    int b = index - s;
                    // Adds conflict free offset to shared memory access.
                    // See upsweep or macro comments.
                    int pa = a + CONFLICT_FREE_OFFSET(a);
                    int pb = b + CONFLICT_FREE_OFFSET(b);

                    int lNode = temp[pb];
                    int rNode = temp[pa];

                    temp[pb] = rNode;
                    temp[pa] = lNode + rNode;
                }
                __syncthreads();
            }
        }

        // Does an upsweep/downsweep locally on each block, then records totals to blockSums
        __global__ void blockScanShared(int n, int* odata, const int* idata, int* blockSums) {
            extern __shared__ int temp[];

            int tid = threadIdx.x;
            int index = threadIdx.x + (blockDim.x * blockIdx.x);
            // Add offset so we can access shared memory correctly later.
            int padded = tid + CONFLICT_FREE_OFFSET(tid);

            temp[padded] = (index < n) ? idata[index] : 0;
            __syncthreads();

            // Use thread idx within block (tid) for local upsweep and downsweep
            upSweepSharedMem(temp, tid, blockDim.x);

            // Zero each block's final element before downsweep
            if (tid == blockDim.x - 1) {
                temp[padded] = 0;
            }
            __syncthreads();

            downSweepSharedMem(temp, tid, blockDim.x);

            if (index < n) {
                odata[index] = temp[padded];
            }

            // Record total to blockSums for recursive scan
            if (tid == blockDim.x - 1) {
                int ownValue = (index < n) ? idata[index] : 0;
                blockSums[blockIdx.x] = temp[padded] + ownValue;
            }
        }

        // Adds previous blocks' totals to current blocks. See naive method of the same name
        __global__ void addBlockOffsetsShared(int n, int* odata, const int* blockOffsets) {
            int index = threadIdx.x + (blockDim.x * blockIdx.x);

            if (index >= n)
                return;

            odata[index] += blockOffsets[blockIdx.x];
        }

        // work efficient shared memory scan that gets local scans per block, 
        // then recursively calls on those scan totals to get sums that we add back later.
        // Conflict free offset is also available since we are using shared memory.
        void scanSharedMemDevice(int n, int* dev_odata, const int* dev_idata, int*& arena) {
            unsigned blockSize = Common::getBlockSize();
            unsigned gridSize = Common::divup(n, blockSize);

            // In order to do the conflict free offsets, you can increase the amount of shared memory available
            // depending on the size of the maximum offset you need (otherwise you couldn't add the offset, since you
            // wouldn't have enough memory)
            size_t sharedBytes = (size_t)(blockSize + CONFLICT_FREE_OFFSET(blockSize - 1)) * sizeof(int);

            int* dev_blockSums = arena;
            arena += gridSize;

            blockScanShared<<<gridSize, blockSize, sharedBytes>>>(n, dev_odata, dev_idata, dev_blockSums);
            checkCUDAErrorFn("blockScanShared failed!");

            if (gridSize > 1) {
                int* dev_blockOffsets = arena;
                arena += gridSize;

                scanSharedMemDevice(gridSize, dev_blockOffsets, dev_blockSums, arena);

                addBlockOffsetsShared<<<gridSize, blockSize>>>(n, dev_odata, dev_blockOffsets);
                checkCUDAErrorFn("addBlockOffsetsShared failed!");
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         * Uses shared memory
         */
        void scanSharedMem(int n, int* odata, const int* idata) {
            int* dev_odata;
            int* dev_idata;

            cudaMalloc((void**)&dev_odata, sizeof(int) * n);
            checkCUDAErrorFn("cudaMalloc dev_odata failed!");
            cudaMalloc((void**)&dev_idata, sizeof(int) * n);
            checkCUDAErrorFn("cudaMalloc dev_idata failed!");

            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAErrorFn("cudaMemcpy failed!");

            // Determine size of "scratch space" needed to get every level of the scan
            // that we do our recursive calls on.
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
            int* arena = dev_scratch;
            scanSharedMemDevice(n, dev_odata, dev_idata, arena);
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("cudaMemcpy failed!");

            cudaFree(dev_odata);
            cudaFree(dev_idata);
            cudaFree(dev_scratch);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            unsigned blockSize = Common::getBlockSize();
            unsigned gridSize = Common::divup(n, blockSize);

            // calculates next power of two greater than n
            int n_pad = n;
            if ((n & (n - 1)) != 0) {
                n_pad = 1;
                while (n_pad < n) n_pad <<= 1;
            }
            int padding_elements = n_pad - n;

            unsigned gridSizePad = Common::divup(n_pad, blockSize);

            // inits
            int* dev_idata;
            int* dev_odata;
            int* dev_map;
            int* dev_sumMap;

            // mallocs
            cudaMalloc((void**)&dev_idata, sizeof(int) * n_pad);
            checkCUDAErrorFn("cudaMalloc dev_idata failed!");
            cudaMalloc((void**)&dev_odata, sizeof(int) * n);
            checkCUDAErrorFn("cudaMalloc dev_odata failed!");
            cudaMalloc((void**)&dev_map, sizeof(int) * n_pad);
            checkCUDAErrorFn("cudaMalloc dev_map failed!");
            cudaMalloc((void**)&dev_sumMap, sizeof(int) * n_pad);
            checkCUDAErrorFn("cudaMalloc dev_sumMap failed!");

            // copying and setting memory to idata
            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAErrorFn("cudaMemcpy failed!");
            if (padding_elements > 0) {
                cudaMemset(dev_idata + n, 0, sizeof(int) * padding_elements);
                checkCUDAErrorFn("cudaMemset failed!");
                cudaMemset(dev_map + n, 0, sizeof(int) * padding_elements);
                checkCUDAErrorFn("cudaMemset failed!");
            }

            timer().startGpuTimer();
            Common::kernMapToBoolean<<<gridSize, blockSize>>>(n, dev_map, dev_idata);

            // by doing this first iteration with dev_map, we can avoid a cudaMemcpy.
            // This one level still uses the old full-coverage upSweep kernel
            // (not the compacted one) because it's the only level that needs
            // to touch EVERY position - it's what seeds dev_sumMap with
            // dev_map's values in the first place, active positions and
            // untouched ones alike. Every level after this one only ever
            // touches positions dev_sumMap already holds valid data for, so
            // they can safely switch to the compacted, in-place kernel.
            upSweep<<<gridSizePad, blockSize>>>(n_pad, 1, dev_sumMap, dev_map);
            checkCUDAErrorFn("upSweep failed!");

            for (int i = 2; i < n_pad; i *= 2) {
                int activeCount = n_pad / (i * 2);
                unsigned levelGridSize = Common::divup((unsigned)activeCount, blockSize);
                upSweepCompact<<<levelGridSize, blockSize>>>(activeCount, i, dev_sumMap);
                checkCUDAErrorFn("upSweepCompact failed!");
            }

            // set last element of odata to 0
            cudaMemset(dev_sumMap + n_pad - 1, 0, sizeof(int));
            checkCUDAErrorFn("cudaMemset failed!");

            for (int i = n_pad / 2; i > 0; i /= 2) {
                int activeCount = n_pad / (i * 2);
                unsigned levelGridSize = Common::divup((unsigned)activeCount, blockSize);
                downSweepCompact<<<levelGridSize, blockSize>>>(activeCount, i, dev_sumMap);
                checkCUDAErrorFn("downSweepCompact failed!");
            }
            // END SCAN

            Common::kernScatter<<<gridSize, blockSize>>>(n, dev_odata, dev_idata, dev_map, dev_sumMap);
            timer().endGpuTimer();

            int sumMapLast;
            int mapLast;

            cudaMemcpy(&sumMapLast, dev_sumMap + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("cudaMemcpy failed!");
            cudaMemcpy(&mapLast, dev_map + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("cudaMemcpy failed!");
            int count = sumMapLast + mapLast;

            cudaMemcpy(odata, dev_odata, sizeof(int) * count, cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("cudaMemcpy failed!");

            cudaFree(dev_odata);
            cudaFree(dev_idata);
            cudaFree(dev_map);
            cudaFree(dev_sumMap);
            return count;
        }
    }
}

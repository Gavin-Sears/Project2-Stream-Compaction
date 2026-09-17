#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        __global__ void upSweep(int n, int i, int* odata, const int* idata);
        __global__ void upSweepCompact(int activeCount, int i, int* data);
        __global__ void downSweepCompact(int activeCount, int i, int* data);

        __device__ void upSweepSharedMem(int* temp, int tid, int blockElements);
        __device__ void downSweepSharedMem(int* temp, int tid, int blockElements);
        __global__ void blockScanShared(int n, int* odata, const int* idata, int* blockSums);
        __global__ void addBlockOffsetsShared(int n, int* odata, const int* blockOffsets);

        void scan(int n, int *odata, const int *idata);
        void scanSharedMem(int n, int *odata, const int *idata);

        int compact(int n, int *odata, const int *idata);
    }
}

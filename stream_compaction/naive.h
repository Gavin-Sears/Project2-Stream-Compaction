#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Naive {
        StreamCompaction::Common::PerformanceTimer& timer();

        __global__ void naiveScan(int n, int i, int* odata, const int* idata);

        void scan(int n, int *odata, const int *idata);
        void scanSharedMem(int n, int *odata, const int *idata);
    }
}

#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // Takes an input array of ints, returns a map with zeroes as 0, nonzeroes as 1
        void mapOneZero(int n, int *odata, const int *idata) {
            for (int i = 0; i < n; ++i) {
                odata[i] = abs(idata[i]) > 0 ? 1 : 0;
            }
            return;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scanNoTimer(int n, int *odata, const int *idata) {
            odata[0] = 0;
            for (int i = 1; i < n; ++i) {
                odata[i] = odata[i - 1] + idata[i - 1];
            }
        }

        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            scanNoTimer(n, odata, idata);
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int oneIndex = -1;
            for (int i = 0; i < n; ++i) {
                if (abs(idata[i]) > 0) {
                    ++oneIndex;
                    odata[oneIndex] = idata[i];
                }
            }
            timer().endCpuTimer();
            // return count of elements in resulting array, -1 if empty
            return oneIndex + 1;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int* map = new int[n];
            int* sumMap = new int[n];

            mapOneZero(n, map, idata);
            scanNoTimer(n, sumMap, map);

            int count = 0;
            for (int i = 0; i < n; ++i) {
                if (map[i]) {
                    odata[sumMap[i]] = idata[i];
                    ++count;
                }
            }
            timer().endCpuTimer();

            delete[] map;
            delete[] sumMap;
            return count;
        }
    }
}

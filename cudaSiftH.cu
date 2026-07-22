//********************************************************//
// CUDA SIFT extractor by Mårten Björkman aka Celebrandil //
//********************************************************//
#include "cudaUtils.h"
#include "cudaSift.h"

#include <cstdio>
#include <iostream>
#include <algorithm>

void InitCuda(int devNum)
{
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  if (!nDevices)
  {
    std::cerr << "No CUDA devices available" << std::endl;
    return;
  }

  devNum = std::min(nDevices - 1, devNum);
  deviceInit(devNum);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, devNum);
  printf("Device Number: %d\n", devNum);
  printf("  Device name: %s\n", prop.name);
  printf("  Memory Clock Rate (MHz): %d\n", prop.memoryClockRate / 1000);
  printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
  printf("  Peak Memory Bandwidth (GB/s): %.1f\n\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
}

void InitSiftData(SiftData &data, int num, bool host, bool dev)
{
  data.numPts = 0;
  data.maxPts = num;
  int sz = sizeof(SiftPoint) * num;
#ifdef MANAGEDMEM
  safeCall(cudaMallocManaged((void **)&data.m_data, sz));
#else
  data.h_data = NULL;
  if (host)
  {
    safeCall(cudaMallocHost((void **)&data.h_data, sz));
  }
  data.d_data = NULL;
  if (dev)
  {
    safeCall(cudaMalloc((void **)&data.d_data, sz));
  }
#endif
}

void FreeSiftData(SiftData &data)
{
#ifdef MANAGEDMEM
  safeCall(cudaFree(data.m_data));
#else
  if (data.d_data != NULL)
  {
    safeCall(cudaFree(data.d_data));
  }
  data.d_data = NULL;
  if (data.h_data != NULL)
  {
    safeCall(cudaFreeHost(data.h_data)); // matches cudaMallocHost in InitSiftData
  }
  data.h_data = NULL;
#endif
  data.numPts = 0;
  data.maxPts = 0;
}

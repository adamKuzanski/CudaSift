#ifndef CUDASIFT_H
#define CUDASIFT_H
#pragma once

#if defined(_WIN32)
#if defined(CUDASIFT_SHARED)
#if defined(CUDASIFT_EXPORTS)
#define CUDASIFT_API __declspec(dllexport)
#else
#define CUDASIFT_API __declspec(dllimport)
#endif
#else
// Static build or no explicit sharing; no decoration needed
#define CUDASIFT_API
#endif
#else
#if defined(CUDASIFT_SHARED) && defined(__GNUC__)
#define CUDASIFT_API __attribute__((visibility("default")))
#else
#define CUDASIFT_API
#endif
#endif

typedef struct
{
  float xpos;
  float ypos;
  float scale;
  float sharpness;
  float edgeness;
  float orientation;
  float score;
  float ambiguity;
  int match;
  float match_xpos;
  float match_ypos;
  float match_error;
  float subsampling;
  float empty[3];
  float data[128];
} SiftPoint;

typedef struct
{
  int numPts; // Number of available Sift points
  int maxPts; // Number of allocated Sift points
#ifdef MANAGEDMEM
  SiftPoint *m_data; // Managed data
#else
  SiftPoint *h_data; // Host (CPU) data
  SiftPoint *d_data; // Device (GPU) data
#endif
} SiftData;

struct CUstream_st;
typedef struct CUstream_st *cudaStream_t;

CUDASIFT_API void InitCuda(int devNum = 0);
CUDASIFT_API void InitSiftData(SiftData &data, int num = 1024, bool host = false, bool dev = true);
CUDASIFT_API void FreeSiftData(SiftData &data);
CUDASIFT_API double MatchSiftData(SiftData &data1, SiftData &data2, cudaStream_t stream = 0);

#endif

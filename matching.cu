#include "cudaSift.h"
#include "cudaUtils.h"

#include <cuda_runtime.h>

#define M7W 32
#define M7H 32
#define M7R 4
#define NRX 2
#define NDIM 128

int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

__global__ void FindMaxCorr10(SiftPoint *sift1, SiftPoint *sift2, int numPts1, int numPts2)
{
    // Shared Memory between all threads. Float4 stores 4 floats at once.
    __shared__ float4 buffer1[M7W * NDIM / 4];
    __shared__ float4 buffer2[M7H * NDIM / 4];

    // Thread X and Y indices.
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Block Pointer pointing to the index of the current block in sift1 point array.
    int bp1 = M7W * blockIdx.x;

    // Load SIFT1 Descriptors into Shared Memory
    for (int j = ty; j < M7W; j += M7H / M7R)
    {
        // Safe array index: base index + j taking care not to get over limit.
        int p1 = min(bp1 + j, numPts1 - 1);

        // Load SIFT1 Descriptors into Shared Memory
        for (int d = tx; d < NDIM / 4; d += M7W)
        {
            // j represents which SIFT point we're processing (0-31)
            // NDIM / 4 = 32 float4 descriptors (since we're using float4, each element holds 4 floats)
            // j * 32 gives the starting position in buffer1 for SIFT point j
            // (d + j) % (NDIM / 4) gives the correct index within the float4 using circular shift
            buffer1[j * NDIM / 4 + (d + j) % (NDIM / 4)] = ((float4 *)&sift1[p1].data)[d];

            // Read 4 consecutive floats from sift1[p1].data[4*d] through sift1[p1].data[4*d+3]
            // Store them as a single float4 in shared memory at the calculated shifted position
        }
    }

    // Initialize Score Tracking Variables
    float max_score[NRX];
    float sec_score[NRX];
    int index[NRX];
    for (int i = 0; i < NRX; i++)
    {
        max_score[i] = 0.0f;
        sec_score[i] = 0.0f;
        index[i] = -1;
    }

    // Calculate Thread Indices
    int idx = ty * M7W + tx;
    int ix = idx % (M7W / NRX);
    int iy = idx / (M7W / NRX);

    // Loop over all SIFT2 points in chunks
    for (int bp2 = 0; bp2 < numPts2 - M7H + 1; bp2 += M7H)
    {
        // Load SIFT2 Descriptors
        for (int j = ty; j < M7H; j += M7H / M7R)
        {
            int p2 = min(bp2 + j, numPts2 - 1);
            for (int d = tx; d < NDIM / 4; d += M7W)
            {
                buffer2[j * NDIM / 4 + d] = ((float4 *)&sift2[p2].data)[d];
            }
        }
        __syncthreads();

        // Determine which threads actually perform the correlation computation work: idx < 128
        if (idx < M7W * M7H / M7R / NRX)
        {
            // Initialize final scores.
            float score[M7R][NRX]; // 4x2 array for current correlations
            for (int dy = 0; dy < M7R; dy++)
            {
                for (int i = 0; i < NRX; i++)
                {
                    score[dy][i] = 0.0f;
                }
            }

            // Correlation computation loop: process all 32 float4 descriptors
            for (int d = 0; d < NDIM / 4; d++)
            {
                // Load Sift1 vectors with circular indexing
                float4 v1[NRX];
                for (int i = 0; i < NRX; i++)
                {
                    v1[i] = buffer1[((M7W / NRX) * i + ix) * NDIM / 4 + (d + (M7W / NRX) * i + ix) % (NDIM / 4)];
                }

                // Compute correlations
                for (int dy = 0; dy < M7R; dy++)
                {
                    // Load SIFT 2 vector for current row
                    float4 v2 = buffer2[(M7R * iy + dy) * (NDIM / 4) + d];

                    // Compute dot product: sum of element-wise multiplication
                    for (int i = 0; i < NRX; i++)
                    {
                        score[dy][i] += v1[i].x * v2.x;
                        score[dy][i] += v1[i].y * v2.y;
                        score[dy][i] += v1[i].z * v2.z;
                        score[dy][i] += v1[i].w * v2.w;
                    }
                }
            }

            // Update best matches
            for (int dy = 0; dy < M7R; dy++)
            {
                for (int i = 0; i < NRX; i++)
                {
                    // Check if score is greater than max_score
                    if (score[dy][i] > max_score[i])
                    {
                        sec_score[i] = max_score[i];                      // Old best becomes second
                        max_score[i] = score[dy][i];                      // New best
                        index[i] = min(bp2 + M7R * iy + dy, numPts2 - 1); // Best match index
                    }
                    else if (score[dy][i] > sec_score[i])
                    {
                        sec_score[i] = score[dy][i]; // Update second best
                    }
                }
            }
        }
        __syncthreads();
    }

    // Store Intermediate Results
    float *scores1 = (float *)buffer1;               // Reuse buffer1 as float array
    float *scores2 = &scores1[M7W * M7H / M7R];      // Second half for second scores
    int *indices = (int *)&scores2[M7W * M7H / M7R]; // Third section for indices

    // Store thread-local results in shared memory for reduction
    if (idx < M7W * M7H / M7R / NRX)
    {
        for (int i = 0; i < NRX; i++)
        {
            scores1[iy * M7W + (M7W / NRX) * i + ix] = max_score[i];
            scores2[iy * M7W + (M7W / NRX) * i + ix] = sec_score[i];
            indices[iy * M7W + (M7W / NRX) * i + ix] = index[i];
        }
    }
    __syncthreads();

    // Early exit for threads that would be out of bounds
    if (bp1 + tx >= numPts1)
    {
        return;
    }

    // Final Reduction (Thread 0 in each row)
    if (ty == 0)
    {
        // Load max scores
        float max_score = scores1[tx];
        float sec_score = scores2[tx];
        int index = indices[tx];

        // Find the best match
        for (int y = 0; y < M7H / M7R; y++)
        {
            // Take index of best matches
            if (index != indices[y * M7W + tx])
            {
                // Check if current score is better than max_score
                if (scores1[y * M7W + tx] > max_score)
                {
                    sec_score = max(max_score, sec_score);
                    max_score = scores1[y * M7W + tx];
                    index = indices[y * M7W + tx];
                }
                else if (scores1[y * M7W + tx] >= sec_score)
                {
                    sec_score = scores1[y * M7W + tx];
                }
            }
        }

        sift1[bp1 + tx].score = max_score;
        sift1[bp1 + tx].match = index;
        sift1[bp1 + tx].match_xpos = sift2[index].xpos;
        sift1[bp1 + tx].match_ypos = sift2[index].ypos;
        sift1[bp1 + tx].ambiguity = sec_score / (max_score + 1e-6f);
    }
}

__global__ void CleanMatches(SiftPoint *sift1, int numPts1)
{
    const int p1 = min(blockIdx.x * 64 + threadIdx.x, numPts1 - 1);
    sift1[p1].score = 0.0f;
}

double MatchSiftData(SiftData &data1, SiftData &data2, cudaStream_t stream)
{
    int numPts1 = data1.numPts;
    int numPts2 = data2.numPts;
    if (!numPts1 || !numPts2)
    {
        return 0.0;
    }
#ifdef MANAGEDMEM
    SiftPoint *sift1 = data1.m_data;
    SiftPoint *sift2 = data2.m_data;
#else
    if (data1.d_data == NULL || data2.d_data == NULL)
    {
        return 0.0f;
    }
    SiftPoint *sift1 = data1.d_data;
    SiftPoint *sift2 = data2.d_data;
#endif

    dim3 blocksMax3(iDivUp(numPts1, 16), iDivUp(numPts2, 512));
    dim3 threadsMax3(16, 16);
    CleanMatches<<<iDivUp(numPts1, 64), 64, 0, stream>>>(sift1, numPts1);

    blocksMax3 = dim3(iDivUp(numPts1, M7W));
    threadsMax3 = dim3(M7W, M7H / M7R);
    FindMaxCorr10<<<blocksMax3, threadsMax3, 0, stream>>>(sift1, sift2, numPts1, numPts2);

    if (data1.h_data != NULL)
    {
        float *h_ptr = &data1.h_data[0].score;
        float *d_ptr = &data1.d_data[0].score;
        // Copy back the block of SiftPoint starting at score, ending on match_ypos (4 floats + 1 int).
        safeCall(cudaMemcpy2DAsync(h_ptr, sizeof(SiftPoint), d_ptr, sizeof(SiftPoint), 4 * sizeof(float) + sizeof(int), data1.numPts, cudaMemcpyDeviceToHost, stream));
    }

    // Single synchronization point: wait for the kernels and the copy-back to finish so data1.h_data holds valid results when this returns.
    safeCall(cudaStreamSynchronize(stream));

    return 0.0f;
}

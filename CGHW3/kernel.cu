
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdint>
#include <stdio.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <vector>
#include <iostream>
#include <iomanip>

struct BVHNode {
  float3 AABBMin;
  float3 AABBMax;
  int leftChildIndex;
  int rightChildIndex;
  bool isLeaf = false;
  int parent = -1;
  int counter = 0;
  bool isLeafA = false;
  bool isLeafB = false;

  __device__ void setChildA(uint32_t idx, bool isLeaf) {
    leftChildIndex = idx;
    isLeafA = isLeaf;
  }
  __device__ void setChildB(uint32_t idx, bool isLeaf) {
    rightChildIndex = idx;
    isLeafB = isLeaf;
  }
  __device__ uint32_t getChildA() { return leftChildIndex; }
  __device__ uint32_t getChildB() { return rightChildIndex; }
  __device__ void setParent(int idx) { parent = idx; }
  __device__ bool isRoot() { return parent == -1; }
  __device__ void setAABB(float3 minpos, float3 maxpos) {
    AABBMin = minpos;
    AABBMax = maxpos;
  }
};

__device__ __forceinline__ int imin(int a, int b) { return a < b ? a : b; }
__device__ __forceinline__ int imax(int a, int b) { return a > b ? a : b; }
__host__ __device__ inline float3 fminf3(float3 a, float3 b) {
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

__host__ __device__ inline float3 fmaxf3(float3 a, float3 b) {
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

__device__ __forceinline__ uint32_t expandBits(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ int delta(const uint32_t *morton, int i, int j, int N) {
  if (i >= N || j >= N || i < 0 || j < 0)
    return -1;
  return __clz(morton[i] ^ morton[j]);
}

__device__ __forceinline__ uint32_t morton3D(uint32_t x, uint32_t y,
                                             uint32_t z) {
  return expandBits(x) | (expandBits(y) << 1) | (expandBits(z) << 2);
}

__global__ void computeMortonKernel(const float3* centers, uint32_t* morton,
    int* indices, int N,
    float3 sceneMin, float3 sceneExtent) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float3 c = centers[i];
    float fx = sceneExtent.x > 0.0f ? (c.x - sceneMin.x) / sceneExtent.x : 0.0f;
    float fy = sceneExtent.y > 0.0f ? (c.y - sceneMin.y) / sceneExtent.y : 0.0f;
    float fz = sceneExtent.z > 0.0f ? (c.z - sceneMin.z) / sceneExtent.z : 0.0f;

    int xi = imin(imax(static_cast<int>(floorf(fx * 1023.0f)), 0), 1023);
    int yi = imin(imax(static_cast<int>(floorf(fy * 1023.0f)), 0), 1023);
    int zi = imin(imax(static_cast<int>(floorf(fz * 1023.0f)), 0), 1023);

    morton[i] = morton3D((uint32_t)xi, (uint32_t)yi, (uint32_t)zi);
    indices[i] = i;
}

void mortonSort(float3 *d_centers, uint32_t *d_morton, int *d_indices, int N, float3 sceneMin, float3 sceneExtent) {
  int block = 256;
  int grid = (N + block - 1) / block;
  computeMortonKernel<<<grid, block>>>(d_centers, d_morton, d_indices, N, sceneMin, sceneExtent);

  thrust::device_ptr<uint32_t> key(d_morton);
  thrust::device_ptr<int> val(d_indices);
  thrust::sort_by_key(thrust::device, key, key + N, val);
}

__global__ void algorithm(const uint32_t *morton, int N, BVHNode *internalNodes,
                          BVHNode *leafNodes) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N-1)
    return;
  int2 range;
  int d = delta(morton, i, i + 1, N) - delta(morton, i, i - 1, N);
  d = (d > 0) ? 1 : -1;
  int delta_min = delta(morton, i, i - d, N);
  int l_max = 2;
  while (delta(morton, i, i + l_max * d, N) > delta_min)
    l_max = l_max << 1;
  int l = 0;
  for (int t = l_max >> 1; t > 0; t >>= 1) {
    if (delta(morton, i, i + (l + t) * d, N) > delta_min)
      l += t;
  }
  int j = i + l * d;
  range.x = (i <= j) ? i : j;
  range.y = (i <= j) ? j : i;

  int delta_node = delta(morton, i, j, N);
  int s = 0;
  for (int t = (l + 1) >> 1; t > 0; t >>= 1) {
    if (delta(morton, i, i + (s + t) * d, N) > delta_node)
      s += t;
  }
  int gamma = i + s * d + imin(d, 0);

  BVHNode *childA, *childB;
  bool isChildALeaf = false, isChildBLeaf = false;
  if (gamma == range.x) {
    childA = &leafNodes[gamma];
    childA->isLeaf = true;
    isChildALeaf = true;
  } else {
    childA = &internalNodes[gamma];
  }
  if (gamma + 1 == range.y) {
    childB = &leafNodes[gamma + 1];
    childB->isLeaf = true;
    isChildBLeaf = true;
  } else {
    childB = &internalNodes[gamma + 1];
  }
  internalNodes[i].setChildA(gamma, isChildALeaf);
  internalNodes[i].setChildB(gamma + 1, isChildBLeaf);
  childA->setParent(i);
  childB->setParent(i);
}

__global__ void calculateBoudingBox(int N, float3 *minpos, float3 *maxpos,
                                    int *d_indices, BVHNode *leafNodes,
                                    BVHNode *internalNodes) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N)
    return;
  BVHNode *node = &leafNodes[idx];
  int index = d_indices[idx];
  float3 minpos_ = minpos[index];
  float3 maxpos_ = maxpos[index];
  node->setAABB(minpos_, maxpos_);
  if (node->isRoot())
    return;
  node = &internalNodes[node->parent];
  int value = atomicAdd(&node->counter, 1);
  while (1) {
    if (value == 0)
      return;
    float3 minposA;
    float3 maxposA;
    float3 minposB;
    float3 maxposB;
    if (node->isLeafA)
    {
        minposA = leafNodes[node->getChildA()].AABBMin;
        maxposA = leafNodes[node->getChildA()].AABBMax;
    }
    else
    {
        minposA = internalNodes[node->getChildA()].AABBMin;
        maxposA = internalNodes[node->getChildA()].AABBMax;
    }
    if (node->isLeafB)
    {
        minposB = leafNodes[node->getChildB()].AABBMin;
        maxposB = leafNodes[node->getChildB()].AABBMax;
    }
    else
    {
        minposB = internalNodes[node->getChildB()].AABBMin;
        maxposB = internalNodes[node->getChildB()].AABBMax;
    }
    node->setAABB(fminf3(minposA, minposB), fmaxf3(maxposA, maxposB));
    if (node->isRoot())
      return;
    node = &internalNodes[node->parent];
    value = atomicAdd(&node->counter, 1);
  }
}

__global__ void computeCenters(float3* minpos,
    float3* maxpos,
    float3* centers,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    centers[i].x = (minpos[i].x + maxpos[i].x) * 0.5f;
    centers[i].y = (minpos[i].y + maxpos[i].y) * 0.5f;
    centers[i].z = (minpos[i].z + maxpos[i].z) * 0.5f;
}

__global__ void initNodesKernel(BVHNode* leafNodes, int numLeaves, BVHNode* internalNodes, int numInternals) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numLeaves) {
        leafNodes[idx].parent = -1;
        leafNodes[idx].counter = 0;
        leafNodes[idx].isLeaf = true;        
        leafNodes[idx].isLeafA = leafNodes[idx].isLeafB = false;
    }
    if (idx < numInternals) {
        internalNodes[idx].parent = -1;
        internalNodes[idx].counter = 0;
        internalNodes[idx].isLeaf = false;
        internalNodes[idx].isLeafA = internalNodes[idx].isLeafB = false;
        internalNodes[idx].leftChildIndex = -1;
        internalNodes[idx].rightChildIndex = -1;
    }
}

void build(int numLeafNode, float3* minpos_cpu, float3* maxpos_cpu, BVHNode* leafNodes_cpu, BVHNode* internalNodes_cpu, int* indices_cpu)
{
    if (numLeafNode == 0)
        return;
    if (numLeafNode == 1)
    {
        leafNodes_cpu[0].isLeaf = true;
        leafNodes_cpu[0].parent = -1;
        leafNodes_cpu[0].leftChildIndex = -1;
        leafNodes_cpu[0].rightChildIndex = -1;
        leafNodes_cpu[0].AABBMin = minpos_cpu[0];
        leafNodes_cpu[0].AABBMax = maxpos_cpu[0];
        return;
    }
    BVHNode* leafNodes = nullptr;
    BVHNode* internalNodes = nullptr;
    int* indices = nullptr;
    float3* centers = nullptr;
    uint32_t* morton = nullptr;
    float3* minpos = nullptr;
    float3* maxpos = nullptr;

    cudaError_t cudaStatus = cudaMalloc((void**)&leafNodes, numLeafNode * sizeof(BVHNode));
    if (cudaStatus != cudaSuccess)
    { 
        fprintf(stderr, "cudaMalloc failed! leafNodes\n"); 
        goto cleanup; 
    }
    cudaStatus = cudaMalloc((void**)&internalNodes, (numLeafNode - 1) * sizeof(BVHNode));
    if (cudaStatus != cudaSuccess) 
    { 
        fprintf(stderr, "cudaMalloc failed! internalNodes\n"); 
        goto cleanup; 
    }
    cudaStatus = cudaMalloc((void**)&indices, numLeafNode * sizeof(int));
    if (cudaStatus != cudaSuccess) 
    { 
        fprintf(stderr, "cudaMalloc failed! indices\n"); 
        goto cleanup; 
    }
    cudaStatus = cudaMalloc((void**)&centers, numLeafNode * sizeof(float3));
    if (cudaStatus != cudaSuccess) 
    { 
        fprintf(stderr, "cudaMalloc failed! centers\n"); 
        goto cleanup; 
    }
    cudaStatus = cudaMalloc((void**)&morton, numLeafNode * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) 
    { 
        fprintf(stderr, "cudaMalloc failed! morton\n"); 
        goto cleanup; 
    }
    cudaStatus = cudaMalloc((void**)&minpos, numLeafNode * sizeof(float3));
    if (cudaStatus != cudaSuccess) 
    { 
        fprintf(stderr, "cudaMalloc failed! minpos\n"); 
        goto cleanup; 
    }
    cudaStatus = cudaMalloc((void**)&maxpos, numLeafNode * sizeof(float3));
    if (cudaStatus != cudaSuccess) 
    { 
        fprintf(stderr, "cudaMalloc failed! maxpos\n"); 
        goto cleanup; 
    }

    cudaStatus = cudaMemcpy(maxpos, maxpos_cpu, numLeafNode * sizeof(float3), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    { 
        fprintf(stderr, "cudaMemcpy maxpos failed\n"); 
        goto cleanup; 
    }
    cudaStatus = cudaMemcpy(minpos, minpos_cpu, numLeafNode * sizeof(float3), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) 
    { 
        fprintf(stderr, "cudaMemcpy minpos failed\n"); 
        goto cleanup; 
    }

    int threadPerBlock = 256;
    int numBlocksInit = (numLeafNode + threadPerBlock - 1) / threadPerBlock;
    initNodesKernel <<<numBlocksInit, threadPerBlock >>> (leafNodes, numLeafNode, internalNodes, numLeafNode - 1);
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    { 
        fprintf(stderr, "initNodesKernel failed\n"); 
        goto cleanup; 
    }

    int numBlocks = (numLeafNode + threadPerBlock - 1) / threadPerBlock;
    computeCenters <<<numBlocks, threadPerBlock >>> (minpos, maxpos, centers, numLeafNode);
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) 
    { 
        fprintf(stderr, "computeCenters failed\n"); 
        goto cleanup; 
    }

    /* ---- 拉回并打印 centers ---- */
    float3* h_centers = new float3[numLeafNode];
    cudaMemcpy(h_centers, centers, numLeafNode * sizeof(float3), cudaMemcpyDeviceToHost);

    for (int i = 0; i < numLeafNode; ++i)
        printf("center[%d] = [%f, %f, %f]\n",
            i, h_centers[i].x, h_centers[i].y, h_centers[i].z);
    fflush(stdout);

    delete[] h_centers;


    float3 sceneMinHost = minpos_cpu[0];
    float3 sceneMaxHost = maxpos_cpu[0];

    for (int i = 1; i < numLeafNode; ++i) {
        sceneMinHost = fminf3(sceneMinHost, minpos_cpu[i]);
        sceneMaxHost = fmaxf3(sceneMaxHost, maxpos_cpu[i]);
    }
    printf("sceneMin = [%f, %f, %f]\n", sceneMinHost.x, sceneMinHost.y, sceneMinHost.z);
    printf("sceneMax = [%f, %f, %f]\n", sceneMaxHost.x, sceneMaxHost.y, sceneMaxHost.z);

    float3 sceneExtentHost = make_float3(
        sceneMaxHost.x - sceneMinHost.x,
        sceneMaxHost.y - sceneMinHost.y,
        sceneMaxHost.z - sceneMinHost.z
    );

    mortonSort(centers, morton, indices, numLeafNode, sceneMinHost, sceneExtentHost);

    uint32_t* h_morton = new uint32_t[numLeafNode];
    cudaMemcpy(h_morton, morton, numLeafNode * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    for (int i = 0; i < numLeafNode; ++i)
        printf("morton[%d] = 0x%08X\n", i, h_morton[i]);
    fflush(stdout);

    delete[] h_morton;

    numBlocks = ((numLeafNode - 1) + threadPerBlock - 1) / threadPerBlock;
    algorithm <<<numBlocks, threadPerBlock >>> (morton, numLeafNode, internalNodes, leafNodes);
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    { 
        fprintf(stderr, "algorithm kernel failed\n"); 
        goto cleanup; 
    }

    numBlocks = (numLeafNode + threadPerBlock - 1) / threadPerBlock;
    calculateBoudingBox <<<numBlocks, threadPerBlock >>> (numLeafNode, minpos, maxpos, indices, leafNodes, internalNodes);
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) 
    { 
        fprintf(stderr, "calculateBoudingBox failed\n"); 
        goto cleanup; 
    }

    cudaStatus = cudaMemcpy(internalNodes_cpu, internalNodes, (numLeafNode - 1) * sizeof(BVHNode), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) 
    { 
        fprintf(stderr, "cudaMemcpy internalNodes_cpu failed\n"); 
        goto cleanup; 
    }
    cudaStatus = cudaMemcpy(leafNodes_cpu, leafNodes, numLeafNode * sizeof(BVHNode), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) 
    { 
        fprintf(stderr, "cudaMemcpy leafNodes_cpu failed\n"); 
        goto cleanup; 
    }

    cudaStatus = cudaMemcpy(indices_cpu, indices, numLeafNode * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy leafNodes_cpu failed\n");
        goto cleanup;
    }

cleanup:
    cudaFree(leafNodes);
    cudaFree(internalNodes);
    cudaFree(indices);
    cudaFree(centers);
    cudaFree(morton);
    cudaFree(minpos);
    cudaFree(maxpos);
}

int main() {
    int N = 5;

    float3* minpos_cpu = new float3[N];
    float3* maxpos_cpu = new float3[N];
    int* indices = new int[N];

    minpos_cpu[0] = make_float3(0, 0, 0);
    maxpos_cpu[0] = make_float3(1, 1, 1);

    minpos_cpu[1] = make_float3(2, -1, 0.2);
    maxpos_cpu[1] = make_float3(3, 0.5, 1.2);

    minpos_cpu[2] = make_float3(-1, 2, 0);
    maxpos_cpu[2] = make_float3(0,3,2);

    minpos_cpu[3] = make_float3(4, 0.3, -1);
    maxpos_cpu[3] = make_float3(6, 1.7, 0.5);

    minpos_cpu[4] = make_float3(1.5, -0.5, 2);
    maxpos_cpu[4] = make_float3(2.5,0.5,3);

    BVHNode* leafNodes_cpu = new BVHNode[N];
    BVHNode* internalNodes_cpu = new BVHNode[N - 1];

    build(N, minpos_cpu, maxpos_cpu,
        leafNodes_cpu, internalNodes_cpu,indices);

    printf("=== Leaf Nodes ===\n");
    for (int i = 0; i < N; i++) {
        printf("leaf[%d]: AABB = [%f,%f,%f] - [%f,%f,%f]\n",
            i,
            leafNodes_cpu[i].AABBMin.x, leafNodes_cpu[i].AABBMin.y, leafNodes_cpu[i].AABBMin.z,
            leafNodes_cpu[i].AABBMax.x, leafNodes_cpu[i].AABBMax.y, leafNodes_cpu[i].AABBMax.z);
    }

    printf("\n=== Internal Nodes ===\n");
    for (int i = 0; i < N - 1; i++) {
        printf("internal[%d]: left=%d,%d, right=%d,%d, AABB = [%f,%f,%f] - [%f,%f,%f]\n",
            i,
            internalNodes_cpu[i].isLeafA ? indices[internalNodes_cpu[i].leftChildIndex] : internalNodes_cpu[i].leftChildIndex,
            internalNodes_cpu[i].isLeafA,
            internalNodes_cpu[i].isLeafB ? indices[internalNodes_cpu[i].rightChildIndex] :internalNodes_cpu[i].rightChildIndex,
            internalNodes_cpu[i].isLeafB,
            internalNodes_cpu[i].AABBMin.x, internalNodes_cpu[i].AABBMin.y, internalNodes_cpu[i].AABBMin.z,
            internalNodes_cpu[i].AABBMax.x, internalNodes_cpu[i].AABBMax.y, internalNodes_cpu[i].AABBMax.z);
    }
    return 0;
}

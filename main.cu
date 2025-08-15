#include <iostream>
#include <cuda.h>
#include <random>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/limits.h>
#include <cfloat>  
#include <iomanip>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/binary_search.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/inner_product.h>
#include <limits>
#include <fstream>

using namespace std;


#define CUDA_CHECK(err) do { if (err != cudaSuccess) { printf("CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); exit(1); } } while(0) 

#define BATCH_SIZE 2560 
//#define BATCH_SIZE  32768 


#define THRUST_CHECK(cmd) try { cmd; } \
    catch(thrust::system_error &e) { \
        std::cerr << "Thrust error: " << e.what() << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        exit(EXIT_FAILURE); \
    }


void cost(int* data, double* centroids, int* assignment, int N, int d, int k);

__device__ double atomicMin_double(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        double assumed_val = __longlong_as_double(assumed);
        if (val >= assumed_val)
            break;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));                                                                                 
    } while (assumed != old);

    return __longlong_as_double(old);
}
  
__global__ void compute_s(double* centroid, double* s, int d, int k){
     int tid = blockIdx.x*blockDim.x + threadIdx.x;
     if(tid >= k) return;
     s[tid] = 1e12; 
     for(int i = 0; i<k; i++){
         if(tid == i) continue;
         double temp = 0.0;
         for(int ii = 0; ii < d; ii++){
            double diff = centroid[tid*d + ii] - centroid[i*d + ii]; 
            temp += diff * diff;
         }
         temp = sqrt(temp);
         if(temp < s[tid]) s[tid] = temp;
     }
     s[tid] /= 2.0;

}

__global__ void DistanceAndClusterUpdate(int* data, int* assignment, double* u, double* l, double* centroids,int N, int d, int k, unsigned long long* sum, int* count ){
   int id = blockIdx.x * blockDim.x + threadIdx.x;
   if(id >= N) return;
   int ind = 0; 
   double minDist =  1e13; 
   for(int i = 0; i<k; i++){
       double dist = 0.0;
        for(int ii = 0; ii < d; ii++){
            double diff = centroids[i*d + ii] - data[ii*N + id];
            dist += diff*diff;
        }
        dist = sqrt(dist);
        if(minDist > dist ){
            ind = i;
            minDist = dist;
        }
   }
  
   u[id] = 0.0; 
   for(int ii = 0; ii < d; ii++){
       double diff = centroids[ind*d + ii] - data[ii*N + id];
       u[id] += diff*diff;
   }
   u[id] = sqrt(u[id]);
   l[id] = 1e13;
   for(int i = 0; i<k; i++){
       if(i == ind) continue;
       double dist = 0.0;
       for(int ii = 0; ii < d; ii++){
           double diff = centroids[i*d + ii] - data[ii*N + id];
           dist += diff*diff;
       }
       dist = sqrt(dist);
       if(l[id] > dist) l[id] = dist; 
   }

   if(ind == assignment[id]) return;
   int old = assignment[id];
   
   atomicAdd(&count[old], -1); 
   atomicAdd(&count[ind], 1); 
   for(int ii = 0; ii < d; ii++){
       atomicAdd(&sum[old*d + ii], -data[ii*N + id]);
       atomicAdd(&sum[ind*d + ii], data[ii*N + id]);
   }
  
   assignment[id] = ind;
}


__global__ void compute_min_distances(int* data, double* centroids, int num_clusters, int n, int d, double* distances) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
                
   // double min_dist_sq = INFINITY;
   // for (int c = 0; c < num_clusters; ++c) {
   //     double dist_sq = 0.0;
   //     for (int j = 0; j < d; ++j) {
   //         double diff = static_cast<double>(data[i * d + j]) - centroids[c * d + j];
   //         dist_sq += diff * diff;
   //     }
   //     if (dist_sq < min_dist_sq) min_dist_sq = dist_sq;
   // }
   // distances[i] = min_dist_sq;
   

    if(num_clusters == 1) distances[i] = INFINITY;
     double dist_sq = 0.0;
     for (int j = 0; j < d; ++j) {
         double diff = static_cast<double>(data[i * d + j]) - centroids[(num_clusters-1)* d + j];
         dist_sq += diff * diff;
     }
     if(dist_sq < distances[i]) distances[i] = dist_sq;
}       
    

__global__ void InitStat(int* data, double* centroids, int num_clusters, int n, int d, int* count, unsigned long long* sum, int* assignment, int b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
                
    int off = b*BATCH_SIZE;
    double min_dist_sq = INFINITY;
    for (int c = 0; c < num_clusters; ++c) {
        double dist_sq = 0.0;
        for (int j = 0; j < d; ++j) {
            double diff = static_cast<double>(data[i * d + j]) - centroids[c * d + j];
            dist_sq += diff * diff;
        }
        if (dist_sq < min_dist_sq){
            min_dist_sq = dist_sq;
            assignment[i+off] = c;
        }
    }
    int cid = assignment[i+off];
    atomicAdd(&count[cid], 1);
    for (int ii = 0; ii < d; ii++){
        atomicAdd(&sum[cid*d + ii], data[i*d + ii]); 
    }

}       
__global__ void compute_distances_to_candidate(int* data, double* candidate_data, int d, int n, double* distances) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double dist_sq = 0.0;
    for (int j = 0; j < d; ++j) {
        double diff = static_cast<double>(data[i * d + j]) - candidate_data[j];
        dist_sq += diff * diff;
    }
    distances[i] = dist_sq;
}


void kmeans_plusplus_init(int* h_data, int N, int d, int k, double* h_centroids) {
    cudaSetDevice(0);
    const int NUM_STREAMS = 2;
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    // Allocate device memory
    double* d_centroids;
    double* d_distances;
    double* d_cumulative;
    CUDA_CHECK(cudaMalloc(&d_centroids, k * d * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_distances, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cumulative, N * sizeof(double)));
   
    //cudaMemset(d_distances, 0xff, N*sizeof(double));
    
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> init_dist(0, N-1);

    // Select first centroid
    int first_idx = init_dist(gen);
    thrust::device_ptr<double> d_centroids_ptr(d_centroids);
    for (int i = 0; i < d; ++i) {
        d_centroids_ptr[i] = static_cast<double>(h_data[first_idx * d + i]);
    }

    const int NUM_BATCHES = (N + BATCH_SIZE - 1) / BATCH_SIZE;
    thrust::device_ptr<double> d_distances_ptr(d_distances);
    thrust::device_ptr<double> d_cumulative_ptr(d_cumulative);
    
    for (int cluster = 1; cluster < k; ++cluster) {
        // Compute minimum distances to existing centroids
        for (int b = 0; b < NUM_BATCHES; ++b) {
            const int stream_idx = b % NUM_STREAMS;
            cudaStream_t stream = streams[stream_idx];

            const int start = b * BATCH_SIZE;
            const int end = std::min((b+1)*BATCH_SIZE, N);
            const int n = end - start;

            int* d_batch;
            CUDA_CHECK(cudaMallocAsync(&d_batch, n * d * sizeof(int), stream));
            CUDA_CHECK(cudaMemcpyAsync(d_batch, h_data + start*d, n*d*sizeof(int), cudaMemcpyHostToDevice, stream));

            const int block_size = 256;
            const int grid_size = (n + block_size - 1) / block_size;

            compute_min_distances<<<grid_size, block_size, 0, stream>>>(
                    d_batch, d_centroids, cluster, n, d, d_distances + start
                    );
            CUDA_CHECK(cudaFreeAsync(d_batch, stream));
        }

        // Synchronize all streams
        for (int i = 0; i < NUM_STREAMS; ++i) {
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        }

        // Calculate total weight
        double total_weight = thrust::reduce(thrust::device, d_distances_ptr, d_distances_ptr + N, 0.0);
        if (total_weight <= 1e-10) {
            printf("Warning: All points have zero distance to existing centroids , cluster valus is %d\n", cluster);
            break;
        }
       // cout << "Cluster nu. " << cluster << "\n";
       // for(int i = 0; i<5; i++){
       //    cout << d_distances_ptr[i] << ((i == 4)?"\n":" "); 
       // }
        // Compute cumulative distribution
        thrust::inclusive_scan(thrust::device, d_distances_ptr, d_distances_ptr + N, d_cumulative_ptr);

        // Select candidates
        int num_trials = 2 + static_cast<int>(std::log(cluster + 1));

        std::vector<double> rand_vals(num_trials);
        std::uniform_real_distribution<> real_dist(0.0, total_weight);
        for (auto& val : rand_vals) val = real_dist(gen);

        std::vector<int> candidate_ids(num_trials);
        for (int t = 0; t < num_trials; ++t) {
            auto iter = thrust::upper_bound(thrust::device, d_cumulative_ptr, d_cumulative_ptr + N, rand_vals[t]);
            candidate_ids[t] = std::min<int>(iter - d_cumulative_ptr, N-1);
        }

        // Evaluate candidates
        double min_potential = INFINITY;
        int best_candidate = candidate_ids[0];

        for (int t = 0; t < num_trials; ++t) {
            const int candidate_idx = candidate_ids[t];
            
            // Copy candidate data to device
            double* d_candidate_data;
            CUDA_CHECK(cudaMalloc(&d_candidate_data, d * sizeof(double)));
            std::vector<double> h_candidate(d);
            for (int j = 0; j < d; ++j) {
                h_candidate[j] = static_cast<double>(h_data[candidate_idx * d + j]);
            }
            CUDA_CHECK(cudaMemcpy(d_candidate_data, h_candidate.data(), d * sizeof(double), cudaMemcpyHostToDevice));

            // Compute distances to candidate
            double* d_candidate_dist;
            CUDA_CHECK(cudaMalloc(&d_candidate_dist, N * sizeof(double)));

            for (int b = 0; b < NUM_BATCHES; ++b) {
                const int stream_idx = b % NUM_STREAMS;
                cudaStream_t stream = streams[stream_idx];

                const int start = b * BATCH_SIZE;
                const int end = std::min((b+1)*BATCH_SIZE, N);
                const int n = end - start;
                int* d_batch;
                CUDA_CHECK(cudaMallocAsync(&d_batch, n * d * sizeof(int), stream));
                CUDA_CHECK(cudaMemcpyAsync(d_batch, h_data + start*d, n*d*sizeof(int), cudaMemcpyHostToDevice, stream));

                const int block_size = 256;
                const int grid_size = (n + block_size - 1) / block_size;

                compute_distances_to_candidate<<<grid_size, block_size, 0, stream>>>(
                        d_batch, d_candidate_data, d, n, d_candidate_dist + start
                        );

                CUDA_CHECK(cudaFreeAsync(d_batch, stream));
            }

            // Wait for all streams to complete
            for (int i = 0; i < NUM_STREAMS; ++i) {
                CUDA_CHECK(cudaStreamSynchronize(streams[i]));
            }

            thrust::device_ptr<double> d_cand_ptr(d_candidate_dist);
            double potential = thrust::inner_product(
                    thrust::device,
                    d_distances_ptr,
                    d_distances_ptr + N,
                    d_cand_ptr,          // Second input range start
                    0.0,                 // Initial value
                    thrust::plus<double>(),      // Reduction operator
                    thrust::minimum<double>()    // Binary transformation
                    );

            if (potential < min_potential) {
                min_potential = potential;
                best_candidate = candidate_idx;
            }

            CUDA_CHECK(cudaFree(d_candidate_dist));
            CUDA_CHECK(cudaFree(d_candidate_data));
        }

        // Add new centroid
        for (int j = 0; j < d; ++j) {
            d_centroids_ptr[cluster*d + j] = static_cast<double>(h_data[best_candidate * d + j]);
        }

    }

    // Copy centroids back to host
    CUDA_CHECK(cudaMemcpy(h_centroids, d_centroids, k*d*sizeof(double), cudaMemcpyDeviceToHost));

    // Cleanup
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    CUDA_CHECK(cudaFree(d_centroids));
    CUDA_CHECK(cudaFree(d_distances));
    CUDA_CHECK(cudaFree(d_cumulative));
}


__device__ double atomicMaxDouble(double* address, double val) {
    unsigned long long int* addr_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *addr_as_ull;
    unsigned long long int assumed;
    
    do {
        assumed = old;
        double current_val = __longlong_as_double(assumed);
        double new_val = fmax(current_val, val);
        old = atomicCAS(addr_as_ull,
                        assumed,
                        __double_as_longlong(new_val));
    } while (assumed != old);

    return __longlong_as_double(old);
}

__global__ void FindMax4all(double* deltas, double* max_other){
   int i = threadIdx.x; 
   int j = blockIdx.x;
   if(i == j) return; 
   atomicMaxDouble(&max_other[i], deltas[j]);

}

__global__ void deltaCompute(unsigned long long* sum, int* count, double* deltas, double* centroids, int d) {
    int cid = blockIdx.x;
    int coordinate = threadIdx.x;

    if (coordinate >= d) return;
    long long sm = sum[cid*d + coordinate];
    // Compute new centroid value and difference
    double val = (double)(sm) / count[cid];
    double diff = centroids[cid*d + coordinate] - val;
    centroids[cid*d + coordinate] = val;
    diff *= diff;

    atomicAdd(&deltas[cid], diff);

    __syncthreads();

    if (threadIdx.x == 0) {
        deltas[cid] = sqrt(deltas[cid]);
    }
}

__global__ void Update(int* data, double* u, double* l, double* s, int* assignment, double* centroids, unsigned long long* sum, int* count, int b, int n, int d, int k, int* counter){
    int id = threadIdx.x + blockIdx.x*blockDim.x;
    if(id >= n) return;
  
    int offset = b*BATCH_SIZE;

    int cid = assignment[id + offset];
    double val = max(s[cid], l[id]);

    if(u[id] < val) return;
    u[id] = 0;  
    for(int ii = 0; ii < d; ii++){
        double diff = centroids[cid*d + ii] - (double)data[id*d + ii];
        u[id] += diff*diff;    
    }
    u[id] = sqrt(u[id]);
    if(u[id] < val) return;
    
    atomicAdd(counter, 1);
   

    double minDist = 1e13, secondMinDist = 1e13;
    int newid = 0, secondid = 0;

    for(int i = 0; i < k; i++) {
        double dist = 0.0;
        for(int ii = 0; ii < d; ii++) {
            double diff = centroids[i*d + ii] - data[id*d + ii];
            dist += diff * diff;
        }
        dist = sqrt(dist);

        if(dist < minDist) {
            secondMinDist = minDist;
            secondid = newid;
            minDist = dist;
            newid = i;
        } else if(dist < secondMinDist) {
            secondMinDist = dist;
            secondid = i;
        }
    }
    u[id] = minDist;
    l[id] = secondMinDist;

    if(newid == cid) return;

    atomicAdd(&count[cid], -1); 
    atomicAdd(&count[newid], 1); 
    for(int ii = 0; ii < d; ii++){ 
        atomicAdd(&sum[cid*d + ii], -data[id*d + ii]);
        atomicAdd(&sum[newid*d + ii], data[id*d + ii]);
    }
    assignment[offset + id] = newid;

}


__global__ void BoundUpdate(double* u, double* l, double* deltas, double* max_deltas, int* assignment, int n, int b){
    int id = threadIdx.x + blockIdx.x*blockDim.x;
    if(id >= n) return;
    int cid = assignment[id + b*BATCH_SIZE];
        
    u[id] += deltas[cid];
    l[id] -= max_deltas[cid];
    if(l[id] < 0) l[id] = 0;
}

int main(int argc, char* argv[]){
    cout << fixed << setprecision(6);
    int N, d, k; cin >> N >> d >> k;
    int *data,  *assignment,  *count, *d_count;
    long long *sum;
    unsigned long long *d_sum;
    double *centroids, *d_centroids, *u, *l; 

    data = (int*)malloc(N*d*sizeof(int));
    assignment = (int*)malloc(N*sizeof(int));
    count = (int*)malloc(k*sizeof(int));

    sum = (long long*)malloc(k*d*sizeof(long long));

    centroids = (double*)malloc(k*d*sizeof(double));
    u = (double*)malloc(N*sizeof(double));
    l = (double*)malloc(N*sizeof(double));

    for(int i = 0; i<N; i++){
        for(int ii = 0; ii < d; ii++){
            int val; cin >> val;
            data[i*d + ii] = val;
        }
        u[i] = 1e15; 
        l[i] = 0.0;
        assignment[i] = 0;
    }

    cudaMalloc(&d_count, k*sizeof(int));
    cudaMalloc(&d_sum, d*k*sizeof(long long));
    cudaMalloc(&d_centroids, d*k*sizeof(double));

    memset(count, 0, k*sizeof(int));
    memset(sum, 0, d*k*sizeof(long long));

    auto start = std::chrono::high_resolution_clock::now();
    kmeans_plusplus_init(data, N, d, k, centroids);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end - start; 
    cout << "Time for K++means on GPU = " << elapsed1.count() << "\n";
    
    cudaMemcpy(d_centroids, centroids, k*d*sizeof(double), cudaMemcpyHostToDevice);
    
    start  =  std::chrono::high_resolution_clock::now();
    int NUM_BATCHES = (N + BATCH_SIZE-1)/BATCH_SIZE;
    int NUM_STREAMS = 2;
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    } 
    int* d_assignment;
    cudaMalloc(&d_assignment, N*sizeof(int));

    for (int b = 0; b < NUM_BATCHES; ++b) {
        const int stream_idx = b % NUM_STREAMS;
        cudaStream_t stream = streams[stream_idx];

        const int start = b * BATCH_SIZE;
        const int end = std::min((b+1)*BATCH_SIZE, N);
        const int n = end - start;

        int* d_batch;
        CUDA_CHECK(cudaMallocAsync(&d_batch, n * d * sizeof(int), stream));
        CUDA_CHECK(cudaMemcpyAsync(d_batch, data + start*d, n*d*sizeof(int), cudaMemcpyHostToDevice, stream));

        const int block_size = 256;
        const int grid_size = (n + block_size - 1) / block_size;

        InitStat<<<grid_size, block_size, 0, stream>>>(
                d_batch, d_centroids, k, n, d, d_count, d_sum, d_assignment, b 
                );
        CUDA_CHECK(cudaFreeAsync(d_batch, stream));
    }
    
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }
    
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }

    cudaMemcpy(sum, d_sum, k*d*sizeof(long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(count, d_count, k*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(assignment, d_assignment, N*sizeof(int), cudaMemcpyDeviceToHost);

    end = std::chrono::high_resolution_clock::now();
    elapsed1 = end-start;
    cout << "For Bound Initialization on CPU: " << elapsed1.count() << "\n";


    cout << "After K++ Means\n";
    cost(data, centroids, assignment, N, d, k);
    
    double *s, *d_s;
    s = (double*)malloc(k*sizeof(double));
    cudaMalloc(&d_s, k*sizeof(double));

    const double epsilon = 1e-6;
    int max_iter = 45; 
    int iter = 0;
    double alpha = 0.0001;

    start = std::chrono::high_resolution_clock::now();
    while(iter++ < max_iter)
    {
        /* Step 1 - Compute s[i]... */ 
        compute_s<<<1, k>>>(d_centroids, d_s, d, k); 
        cudaMemcpy(s, d_s, k*sizeof(double), cudaMemcpyDeviceToHost);
        
        int NUM_BATCHES = (N + BATCH_SIZE-1)/BATCH_SIZE;
        
        const int NUM_STREAMS = 3;
        cudaStream_t streams[NUM_STREAMS];
        for (int i = 0; i < NUM_STREAMS; ++i) {
            CUDA_CHECK(cudaStreamCreate(&streams[i]));
        }
        int* counter;
        cudaMallocManaged(&counter, sizeof(int));
        *counter = 0;
        const int block_size = 256;
        for (int b = 0; b < NUM_BATCHES; ++b) {
            const int stream_idx = b % NUM_STREAMS;
            cudaStream_t stream = streams[stream_idx];

            const int start = b * BATCH_SIZE;
            const int end = std::min((b+1)*BATCH_SIZE, N);
            const int n = end - start;

            int* d_data;
            double *d_l, *d_u;

            CUDA_CHECK(cudaMallocAsync(&d_data, n * d * sizeof(int), stream));
            CUDA_CHECK(cudaMallocAsync(&d_u, n*sizeof(double), stream));
            CUDA_CHECK(cudaMallocAsync(&d_l, n*sizeof(double), stream));

            CUDA_CHECK(cudaMemcpyAsync(d_data, data + start*d, n*d*sizeof(int), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(d_u, u + start, n*sizeof(double), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(d_l, l + start, n*sizeof(double), cudaMemcpyHostToDevice, stream));
            const int grid_size = (n + block_size - 1) / block_size;
            
            Update<<<grid_size, block_size, 0, stream>>>(d_data, d_u, d_l, d_s, d_assignment, d_centroids, d_sum, d_count, b, n, d, k, counter);

            CUDA_CHECK(cudaMemcpyAsync(u+start, d_u, n*sizeof(double), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(l+start, d_l, n*sizeof(double), cudaMemcpyDeviceToHost, stream));

            CUDA_CHECK(cudaFreeAsync(d_data, stream));
            CUDA_CHECK(cudaFreeAsync(d_u, stream));
            CUDA_CHECK(cudaFreeAsync(d_l, stream));
        }

        for (int jj = 0; jj < NUM_STREAMS; ++jj) {
            CUDA_CHECK(cudaStreamSynchronize(streams[jj]));
        }
       // cout << "Iter no. " << iter << "\n";
       // cout << "No. of points updated " << *counter << "\n"; 
        
        // Step - 4 : Update Centroids. 
        
        double *deltas;
        deltas =(double*)malloc(k*sizeof(double));
        memset(deltas, 0, k*sizeof(double));
        double *d_deltas;
        cudaMalloc(&d_deltas, k*sizeof(double));

        cudaMemset(d_deltas, 0, k*sizeof(double));
        deltaCompute<<<k, d>>>(d_sum, d_count, d_deltas, d_centroids, d); 
        cudaMemcpy(deltas, d_deltas, k*sizeof(double), cudaMemcpyDeviceToHost);  
        cudaMemcpy(centroids, d_centroids, d*k*sizeof(double), cudaMemcpyDeviceToHost);

        double *max_deltas, *d_max_deltas;
        max_deltas = (double*)malloc(k*sizeof(double));
        cudaMalloc(&d_max_deltas, k*sizeof(double));
        cudaMemset(d_max_deltas, 0, k*sizeof(double));
        FindMax4all<<<k, k>>>(d_deltas, d_max_deltas);
        cudaMemcpy(max_deltas, d_max_deltas, k*sizeof(double), cudaMemcpyDeviceToHost);  

        double mxVal = 0.0;
        for(int i = 0;i <k; i++) mxVal = max(deltas[i], mxVal); 
        if(mxVal < epsilon){
            cout << "Early convergence\n";
            break;
        }
        for (int b = 0; b < NUM_BATCHES; ++b) {
            const int stream_idx = b % NUM_STREAMS;
            cudaStream_t stream = streams[stream_idx];

            const int start = b * BATCH_SIZE;
            const int end = std::min((b+1)*BATCH_SIZE, N);
            const int n = end - start;

            double *d_l, *d_u;

            CUDA_CHECK(cudaMallocAsync(&d_u, n*sizeof(double), stream));
            CUDA_CHECK(cudaMallocAsync(&d_l, n*sizeof(double), stream));

            CUDA_CHECK(cudaMemcpyAsync(d_u, u + start, n*sizeof(double), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(d_l, l + start, n*sizeof(double), cudaMemcpyHostToDevice, stream));
            const int grid_size = (n + block_size - 1) / block_size;
            
            BoundUpdate<<<grid_size, block_size, 0, stream>>>(d_u, d_l, d_deltas, d_max_deltas, d_assignment, n, b);

            CUDA_CHECK(cudaMemcpyAsync(u+start, d_u, n*sizeof(double), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(l+start, d_l, n*sizeof(double), cudaMemcpyDeviceToHost, stream));

            CUDA_CHECK(cudaFreeAsync(d_u, stream));
            CUDA_CHECK(cudaFreeAsync(d_l, stream));
        }

        for (int jj = 0; jj < NUM_STREAMS; ++jj) {
            CUDA_CHECK(cudaStreamSynchronize(streams[jj]));
        }

        
        for (int i = 0; i < NUM_STREAMS; ++i) {
            CUDA_CHECK(cudaStreamDestroy(streams[i]));
        }
       
        free(deltas);
        cudaFree(d_deltas);
        alpha = 5*alpha;
        alpha = max(alpha, 1.0);
    }
    cudaMemcpy(assignment, d_assignment, N*sizeof(int), cudaMemcpyDeviceToHost);

    end = std::chrono::high_resolution_clock::now();
    elapsed1 = end - start;
    cout << "Time for iteration : " << elapsed1.count() << "\n";

    cout << "After Convergence \n";
    cost(data, centroids, assignment, N, d, k);
    
    if(argc == 1){
        cudaFree(d_assignment);
        cudaFree(d_s);
        free(s);
        return 0;
    }
    string filename = argv[1];
    ofstream outfile(filename);
    if (!outfile.is_open()) {
        cerr << "Error: Could not open file " << filename << " for writing." << endl;
        return 1;
    }
    
    for (int i = 0; i < N; ++i) {
        outfile << assignment[i] << "\n";
    }

    outfile.close();
    
    cudaFree(d_assignment);
    cudaFree(d_s);
    free(s);
}


void cost(int* data, double* centroids, int* assignment, int N, int d, int k)
{
    double ans = 0;
    for(int i = 0; i<N; i++){
        int cid = assignment[i];
        for(int ii = 0; ii < d; ii++){
            double diff = centroids[cid*d + ii] - data[i*d + ii];
            ans += diff*diff;
        }
    }
    cout << ans << "\n";
}

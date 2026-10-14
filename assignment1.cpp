%%cuda_group_save -g "source" -n "radix_solution.cu"
/**
 * The file in which you will implement your GPU solutions!
 */

#include "algorithm_choices.h"

#include <chrono>    // for timing
#include <iostream>  // std::cout, std::endl

#include <algorithm>

#include "cuda_common.h"

namespace{
    int const MAX_WARPS = 32;
    int const MAX_LANES = 32;
    int const RADIX_DOMAIN= 10;
}

namespace csc485b {
namespace a1      {
namespace radix     {

/**
 * The CPU baseline benefits from warm caches because the data was generated on
 * the CPU. Run the data through the GPU once with some arbitrary logic to
 * ensure that the GPU cache is warm too and the comparison is more fair.
 */
__global__
void warm_the_gpu( element_t * data, std::size_t invert_at_pos, std::size_t num_elements )
{
    int const th_id = blockIdx.x * blockDim.x + threadIdx.x;

    // We know this will never be true, because of the data generator logic,
    // but I doubt that the compiler will figure it out. Thus every element
    // should be read, but none of them should be modified.
    if( th_id < num_elements && data[ th_id ] > num_elements * 100 )
    {
        ++data[ th_id ]; // should not be possible.
    }
}

/**
 * Outputs a histogram of size RADIX_DOMAIN that counts how many times a digit (of a certain place value) appears in d_data
 * Outputs a flattened 2D array block_digit_counts such that each "row" is the block number and each "column" is the digit. Used for 
 */
__global__
void histogram( element_t * data, element_t * output_histogram, element_t * block_digit_counts, std::size_t num_data, int num_blocks, int place)
{
    //init shared memory as buckets
    __shared__ element_t smem_histogram [RADIX_DOMAIN]; 

    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    int const local_th_id = (warp_id * blockDim.x) + lane_id;
    int const global_th_id = (blockIdx.x * (blockDim.x * blockDim.y) + ((warp_id * blockDim.x) + lane_id));

    // init buckets to 0
    if (local_th_id < RADIX_DOMAIN){
        smem_histogram[local_th_id] = 0; //initialize buckets to 0
    }

    __syncthreads();

    if (global_th_id < num_data){
        int place_data_val = (data[global_th_id] / place) % 10; 
        atomicAdd(&smem_histogram[place_data_val], 1);
    }

    __syncthreads();

    if (local_th_id < RADIX_DOMAIN){
        int digit = local_th_id;
        block_digit_counts[digit * num_blocks + blockIdx.x] = smem_histogram[digit];
        atomicAdd(&output_histogram[digit], smem_histogram[digit]);
    }

    return;
}

__global__
void initialize_exclusive_scan(element_t *input_data, element_t *scan_data, std::size_t num_elements)
{
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    int const local_th_id = (warp_id * blockDim.x) + lane_id;
    int const global_th_id = blockIdx.x * (blockDim.x * blockDim.y) + local_th_id;

    if (global_th_id < num_elements) {
        if (global_th_id == 0)
            scan_data[global_th_id] = 0;
        else
            scan_data[global_th_id] = input_data[global_th_id - 1];
    }
}

/**
 * Outputs an exclusive prefix sum of the histogram
 */
__global__
void exclusive_scan( element_t * input_data, element_t * output_exclusive, std::size_t num_elements, int stride)
{
    __shared__ element_t smem [MAX_WARPS * MAX_LANES]; 

    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    int const local_th_id = (warp_id * blockDim.x) + lane_id;
    int const global_th_id = (blockIdx.x * (blockDim.x * blockDim.y) + ((warp_id * blockDim.x) + lane_id));

    if (global_th_id < num_elements){
        smem[local_th_id] = input_data[global_th_id]; 
    }

    __syncthreads();

    if (global_th_id < num_elements) {

        element_t new_value = smem[local_th_id];

        if (global_th_id >= stride) {

            // value we need is inside this block
            if (local_th_id >= stride) {
                new_value += smem[local_th_id - stride];
            }

            // value we need is in an earlier block
            else {
                new_value += input_data[global_th_id - stride];
            }
        }

        output_exclusive[global_th_id] = new_value;
    }

    return;
}


/**
 * Outputs an exclusive prefix sum of the histogram
 */
__global__
void scan_block_digit_counts(element_t *block_digit_counts, element_t *block_prefix, int num_blocks)
{
    int digit = threadIdx.x;

    if (digit < RADIX_DOMAIN) {

        int sum = 0;

        for (int block = 0; block < num_blocks; block++) {

            int index =
                digit * num_blocks + block;

            // exclusive scan:
            // save previous total BEFORE adding current block
            block_prefix[index] = sum;

            sum += block_digit_counts[index];
        }
    }
}


__global__
void rank( element_t * data, element_t * output_rank, std::size_t num_elements, int place)
{
    __shared__ int warp_count[MAX_WARPS][RADIX_DOMAIN];

    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    int const local_th_id = (warp_id * blockDim.x) + lane_id;
    int const global_th_id = (blockIdx.x * (blockDim.x * blockDim.y) + ((warp_id * blockDim.x) + lane_id));

    // initialize all warp/digit counts to 0
    if (local_th_id < MAX_WARPS * RADIX_DOMAIN) {
        warp_count[local_th_id / RADIX_DOMAIN][local_th_id % RADIX_DOMAIN] = 0;
    }
    __syncthreads();

    int place_value = 0;
    int rank_in_warp = 0;

    if (global_th_id < num_elements) {
        place_value = (data[global_th_id] / place) % 10;

        unsigned same = __match_any_sync(__activemask(), place_value); // find every lane in my warp with the same digit
        unsigned before_me = (1u << lane_id) - 1; // make a mask of lanes before me
        unsigned matching_before_me = same & before_me; // keep only matching lanes that are before me

        // count them
        rank_in_warp = __popc(matching_before_me);
        int count_in_warp = __popc(same);

        if (rank_in_warp == 0) {
            warp_count[warp_id][place_value] = count_in_warp;
        }

    }
    __syncthreads();

    if (global_th_id < num_elements) {
        int earlier_warp_count = 0;

        for (int w = 0; w < warp_id; w++) {
            earlier_warp_count += warp_count[w][place_value];
        }

        int rank_in_block = earlier_warp_count + rank_in_warp;

        output_rank[global_th_id] = rank_in_block;
    }
}


__global__
void scatter( element_t * data, element_t *output, element_t * exclusive_scan_data, element_t * rank, element_t * block_prefix,
std::size_t num_data, std::size_t num_exclusive_scan_data, int num_blocks, int place)
{

    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    int const local_th_id = (warp_id * blockDim.x) + lane_id; // only use local th_id because we only need 1 block
    int const global_th_id = (blockIdx.x * (blockDim.x * blockDim.y) + ((warp_id * blockDim.x) + lane_id));

    if (global_th_id < num_data){
        //read first
        int target_data = data[global_th_id];
        int place_val = (data[global_th_id] / place) % 10; 

        int earlier_blocks_count = block_prefix[place_val * num_blocks + blockIdx.x];
        int target_index = exclusive_scan_data[place_val] + rank[global_th_id] + earlier_blocks_count;

        //update next 
        output[target_index] = target_data;

    }

    return;
}


/**
 * Performs all the logic of allocating device vectors and copying host/input
 * vectors to the device. Times the opposing_sort() kernel with wall time,
 * but excludes set up and tear down costs such as mallocs, frees, and memcpies.
 */
void run_radix_gpu_soln( std::vector< element_t > data, std::size_t switch_at, std::size_t n )
{
    // Kernel launch configurations. Feel free to change these.
    // This is set to maximise the size of a thread block on a T4, but it hasn't
    // been tuned. It's not known if this is optimal.
    dim3 threads(MAX_LANES, MAX_WARPS);
    std::size_t const threads_per_block = threads.x * threads.y;
    std::size_t const num_blocks =  ( n + threads_per_block - 1 ) / threads_per_block;

    // Allocate arrays on the device/GPU
    element_t * d_data;
    element_t * d_histogram;
    element_t * d_exclusive_scan;
    element_t * d_rank;
    element_t * d_output;
    element_t * d_block_digit_counts;
    element_t * d_block_prefix;

    cudaMalloc( (void**) & d_data, sizeof( element_t ) * n );
    cudaMalloc( (void**) & d_histogram, sizeof( element_t ) * RADIX_DOMAIN);
    cudaMalloc( (void**) & d_exclusive_scan, sizeof( element_t ) * RADIX_DOMAIN);
    cudaMalloc( (void**) & d_rank, sizeof( element_t ) * n );
    cudaMalloc( (void**) & d_output, sizeof( element_t ) * n );
    cudaMalloc( (void**) & d_block_digit_counts, sizeof(element_t) * num_blocks * RADIX_DOMAIN);
    cudaMalloc( (void**) & d_block_prefix, sizeof(element_t) * num_blocks * RADIX_DOMAIN);
    CHECK_ERROR("Allocating input array on device");

    // Copy the input from the host to the device/GPU
    cudaMemcpy( d_data, data.data(), sizeof( element_t ) * n, cudaMemcpyHostToDevice );
    cudaMemset(d_histogram, 0, sizeof(element_t) * RADIX_DOMAIN); // set to 0
    cudaMemset(d_exclusive_scan, 0, sizeof(element_t) * RADIX_DOMAIN); 
    cudaMemset(d_rank, 0, sizeof(element_t) * n);
    cudaMemset(d_output, 0, sizeof(element_t) * n);
    CHECK_ERROR("Copying input array to device");

    // Warm the cache on the GPU for a more fair comparison
    warm_the_gpu<<< num_blocks, threads>>>( d_data, switch_at, n );

    element_t max_value = *std::max_element(data.begin(), data.end());
    int last_place = 1;
    while (max_value / last_place >= 10){
        last_place *= 10;
    }

    // Time the execution of the kernel that you implemented
    auto const kernel_start = std::chrono::high_resolution_clock::now();


    // start host loop
    for (int place=1; place<=last_place; place*=10){
        //reset histogram to 0s every itteration
        cudaMemset( d_histogram, 0, sizeof(element_t) * RADIX_DOMAIN);

        histogram<<<num_blocks, threads>>>(d_data, d_histogram, d_block_digit_counts, n, num_blocks, place);

        //use hillis steele to do an exclusive rolling prefix sum
        initialize_exclusive_scan<<<1, threads>>>(d_histogram, d_scan_data, RADIX_DOMAIN);
        for (int stride = 1; stride < RADIX_DOMAIN; stride *= 2) {
            exclusive_scan<<<1, threads>>>(d_scan_data, d_exclusive_scan, RADIX_DOMAIN, stride);
            std::swap(d_scan_data, d_exclusive_scan);
        }

        //use hillis steele to do an exclusive rolling prefix sum
        for (int digit = 0; digit < RADIX_DOMAIN; digit++) {
            initialize_exclusive_scan<<<scan_num_blocks, threads>>>(d_block_digit_counts + digit * num_blocks, d_scan_data, num_blocks);
            for (int stride = 1; stride < num_blocks; stride *= 2) {
                exclusive_scan<<<scan_num_blocks, threads>>>(d_scan_data, d_block_prefix + digit * num_blocks, num_blocks, stride);
                std::swap(d_scan_data, d_block_prefix);
            }
        }

        rank<<<num_blocks, threads>>>(d_data, d_rank, n, place);
        scatter<<<num_blocks, threads>>>(d_data, d_output, d_exclusive_scan, d_rank, d_block_prefix, n, RADIX_DOMAIN, num_blocks, place);

        d_data = d_output;
    }


    cudaDeviceSynchronize();

    auto const kernel_end = std::chrono::high_resolution_clock::now();
    CHECK_ERROR("Executing kernel on device");

    // After the timer ends, copy the result back, free the device vector,
    // and echo out the timings and the results.
    cudaMemcpy( data.data(), d_data, sizeof( element_t ) * n, cudaMemcpyDeviceToHost );
    CHECK_ERROR("Transferring result back to host");
    cudaFree( d_data );
    cudaFree( d_histogram );
    cudaFree( d_exclusive_scan );
    cudaFree( d_rank );
    cudaFree( d_output );
    cudaFree( d_block_digit_counts );
    cudaFree( d_block_prefix );
    CHECK_ERROR("Freeing device memory");

    std::cout << "GPU Radix Solution time: "
              << std::chrono::duration_cast<std::chrono::nanoseconds>(kernel_end - kernel_start).count()
              << " ns" << std::endl;

    for( auto const x : data ) std::cout << x << " "; std::cout << std::endl;
}

} // namespace radix
} // namespace a1
} // namespace csc485b
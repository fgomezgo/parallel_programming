#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#define ITERATIONS 100
#define ARR_SIZE 1000000

#define ARR_SIZE_PRINT_LIMIT 100

//CUDA values
#define NUM_BLOCKS 4096
#define NUM_THREADS 1

//Random values range
const int MIN_RAND_NUM = -100;
const int MAX_RAND_NUM = 100;

void print_array(int *arr, int size){
    if(size > ARR_SIZE_PRINT_LIMIT){
        return;
    }
    for(int i = 0; i < size; ++i){
        printf("%d ", arr[i]);
    }
    printf("\n");
}

__device__ void sort_2_preordered_segments(const int &start_idx, const int &middle_idx, const int &end_idx, int* from, int* to){
    //Define the starting indexes of the leftmost segment, the rightmost segment and the new segment 
    //that will store the new sorted segment as a combination of the 2 previous ones
    int left_idx  = start_idx;
    int right_idx = middle_idx+1;
    int idx = start_idx;

    //Sort the new segment by sorting the two previously sorted leftmost and rightmost segments
    while(left_idx <= middle_idx || right_idx <= end_idx){

        if(left_idx <= middle_idx && right_idx <= end_idx){

            if(from[left_idx] < from[right_idx]){
                to[idx++] = from[left_idx++];
            } else{
                to[idx++] = from[right_idx++];
            }

        } else if(left_idx <= middle_idx){
            to[idx++] = from[left_idx++];
        } else{
            to[idx++] = from[right_idx++];
        }
    }
}

__global__ void process_layer(int *from, int *to, int layer_size, int num_segments, int segments_per_block){

    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int range_size = index*segments_per_block;

    //End process if blockId surpasses the number of segments to process
    if(range_size >= num_segments){
        return;
    }

    //Iterate over each segment in charge of the current Block in the array [FROM] of size "layer_size" and 
    //sort it as 2 separated ordered lists into the new array [TO]
    for(int range_idx = range_size; 
        range_idx < range_size+segments_per_block && range_idx < num_segments; ++range_idx){
        
        int segment_idx = range_idx*layer_size;

        //Current segment is complete (it has a size of at least "layer_size")
        if( segment_idx+layer_size-1 < ARR_SIZE ){

            //Get start, end and middle indexes of the current segment to sort it as 2 previously ordered segments
            //The first segment from start_idx to middle_idx
            //The second segment from middle_idx+1 to end_idx
            int start_idx   = segment_idx;
            int end_idx     = segment_idx+layer_size-1;
            int middle_idx  = (start_idx+end_idx)>>1;

            //Sort the preordered segments into a new combined ordered segment
            sort_2_preordered_segments(start_idx, middle_idx, end_idx, from, to);

        } else if( ARR_SIZE-segment_idx > (layer_size>>1) ){     //Last segment in layer is not complete but is greater than the previous "layer_size"

            //Get start, end and middle index of the leftmost and rightmost segments
            int start_idx    = segment_idx;
            int end_idx      = ARR_SIZE-1;
            int middle_idx   = segment_idx+(layer_size>>1)-1;

            //Sort the preordered segments into a new combined ordered segment
            sort_2_preordered_segments(start_idx, middle_idx, end_idx, from, to);
        } else{     //Last segment in layer is not complete but is smaller than the previous "layer_size"

            //Copy the last segment directly from [FROM] to [TO]
            for(int i = segment_idx; i < ARR_SIZE; ++i){
                to[i] = from[i];
            }
        }
    }  
}

__global__ void process_last_layer(int *from, int *to, int layer_size){

    //Get start, end and middle index of the leftmost and rightmost segments
    int start_idx    = 0;
    int end_idx      = ARR_SIZE-1;
    int middle_idx   = (layer_size>>1)-1;

    //Sort the preordered segments into a new combined ordered segment
    sort_2_preordered_segments(start_idx, middle_idx, end_idx, from, to);
}

int* iterative_merge_sort(int *arrA, int *arrB){

    //Auxiliar pointers to know to which array; the segments to sort are being passed during each layer
    //Each layer will always try to get the elements to sort from the [FROM] array and put the sorted elements to the [TO] array
    int *from   = arrA;
    int *to     = arrB;

    //The size of the segments in each processed layer
    int layer_size;

    //Iterate over all the segment sizes that are powers of two (2, 4, 8, 16, ...) to sort the array by layers of that sizes
    for(layer_size = 2; layer_size <= ARR_SIZE; layer_size <<= 1){
        
        //Each layer swaps the [FROM] and [TO] pointers so the array that was previously the target to store the newly ordered segments
        //is now the one were the ordered segments are extracted from
        std::swap(from,to);

        //Number of segments of the total array size to be processed in the current layer
        int num_segments        = ARR_SIZE/layer_size+(ARR_SIZE%layer_size != 0);
        //Number of segments to be processed by each block
        int segments_per_block  = num_segments/(NUM_BLOCKS*NUM_THREADS)+
                                (num_segments%(NUM_BLOCKS*NUM_THREADS) != 0); 

        //CUDA Kernel
        process_layer<<<NUM_BLOCKS,NUM_THREADS>>>(from, to, layer_size, num_segments, segments_per_block);
        //cudaDeviceSynchronize();
    }

    //An extra layer has to be processed due to the fact that ARR_SIZE is not a perfect power of 2
    if(layer_size > ARR_SIZE){

        std::swap(from, to);

        //CUDA Kernel
        process_last_layer<<<1,1>>>(from, to, layer_size);
        //cudaDeviceSynchronize();
    }

    //Return pointer to final sorted values of the array
    return to;
}

int main(){

    //Random Seed
    srand(time(NULL));

    //Allocate array's A memory
    int *arrA = (int*)malloc(sizeof(int)*ARR_SIZE);

    //Allocate array's B memory
    int *arrB = (int*)malloc(sizeof(int)*ARR_SIZE);

    //Device arrays reference pointers
    int *deviceA, *deviceB;
    //Allocate space for device copies of array A, B

    cudaMalloc((void**) &deviceA, sizeof(int)*ARR_SIZE);
    cudaMalloc((void**) &deviceB, sizeof(int)*ARR_SIZE);

    for(int iteration = 0; iteration < ITERATIONS; ++iteration){

        //Initialize array A and array B with random values
        for(int i = 0; i < ARR_SIZE; ++i){
            arrA[i] = arrB[i] = (rand()%(MAX_RAND_NUM-MIN_RAND_NUM+1))+MIN_RAND_NUM;
        }

        //Copy initialized arrays to device
        cudaMemcpy(deviceA, arrA, sizeof(int)*ARR_SIZE, cudaMemcpyHostToDevice);
        cudaMemcpy(deviceB, arrB, sizeof(int)*ARR_SIZE, cudaMemcpyHostToDevice);

        //Print array A
        print_array(arrA, ARR_SIZE);

        //Iterative merge sort
        int *sorted_array = iterative_merge_sort(deviceA, deviceB);

        //Copy sorted array from device to host
        cudaMemcpy(arrB, sorted_array, sizeof(int)*ARR_SIZE, cudaMemcpyDeviceToHost);

        //Print any error regarding the GPU
        //printf("%s\n", cudaGetErrorString(cudaGetLastError()));

        //Print whatever array the pointer "to" ended up pointing to (this one stores the final sorted values)
        print_array(arrB, ARR_SIZE);
    }

    //Free allocated memory
    free(arrA);
    free(arrB);

    //Free device allocated memory
    cudaFree(deviceA);
    cudaFree(deviceB);

    return 0;
}
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#define ITERATIONS 10000
#define ARR_SIZE 101

#define ARR_SIZE_PRINT_LIMIT 100

//CUDA values
#define NUM_BLOCKS 2048
#define NUM_THREADS 512

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

__device__ inline void device_swap(int *a, int *b){
    int aux = *a;
    *a = *b;
    *b = aux;
}

__global__ void merge_sort_swap(int *device, bool odd, bool *device_sorted){

    //Assign unique index Id to each thread
    int index = blockIdx.x*blockDim.x + threadIdx.x;

    //Return if index gets past ARR_SIZE 
    if(index >= ARR_SIZE){
        return;
    }

    //printf("index %d\n",index);

    //Odd sort stage
    if(odd){
        if(!(index & 1) && (index < ARR_SIZE-1)){ //Even number
            if(device[index+1] < device[index]){
                device_swap(&device[index],&device[index+1]);
                *device_sorted = false;
            }
        }
    } else{  //Even sort stage
        if((index & 1) && (index < ARR_SIZE-1)){ //Odd number
            if(device[index+1] < device[index]){
                device_swap(&device[index],&device[index+1]);
                *device_sorted = false;
            }
        }
    }
} 

void iterative_merge_sort(int *device){

    //Flag to check if the array is already sorted
    bool sorted = false;
    bool *device_sorted;

    //Allocate flag in device
    cudaMalloc((void**) &device_sorted, sizeof(bool));

    while(!sorted){
        sorted = true;
        //Copy sorted var to device
        cudaMemcpy(device_sorted, &sorted, sizeof(bool), cudaMemcpyHostToDevice);

        merge_sort_swap<<<NUM_BLOCKS, NUM_THREADS>>>(device, true, device_sorted);
        merge_sort_swap<<<NUM_BLOCKS, NUM_THREADS>>>(device, false, device_sorted);

        //Get sorted flag result from device
        cudaMemcpy(&sorted, device_sorted, sizeof(bool), cudaMemcpyDeviceToHost);
    }
}

int main(){

    //Random Seed
    srand(time(NULL));

    //Allocate array's memory
    int *arr = (int*)malloc(sizeof(int)*ARR_SIZE);

    //Device's array reference pointer
    int *device;

    //Allocate space for device copy of array
    cudaMalloc((void**) &device, sizeof(int)*ARR_SIZE);

    for(int iteration = 0; iteration < ITERATIONS; ++iteration){

        printf("[ITERATION]: %d\n", iteration);

        //Initialize array with random values
        for(int i = 0; i < ARR_SIZE; ++i){
            arr[i] = (rand()%(MAX_RAND_NUM-MIN_RAND_NUM+1))+MIN_RAND_NUM;
        }

        //Copy initialized array to device
        cudaMemcpy(device, arr, sizeof(int)*ARR_SIZE, cudaMemcpyHostToDevice);

        //Print array
        print_array(arr, ARR_SIZE);

        //Iterative merge sort
        iterative_merge_sort(device);

        //Copy sorted array from device to host
        cudaMemcpy(arr, device, sizeof(int)*ARR_SIZE, cudaMemcpyDeviceToHost);

        //Print sorted array
        print_array(arr, ARR_SIZE);

        //Print any error regarding the GPU
        printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    }

    //Free allocated memory
    free(arr);
    //Free device allocated memory
    cudaFree(device);

    return 0;
}
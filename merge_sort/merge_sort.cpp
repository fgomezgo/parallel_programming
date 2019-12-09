#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#define ITERATIONS 100
#define ARR_SIZE 1000000

#define ARR_SIZE_PRINT_LIMIT 50

//Random values range
#define MIN_RAND_NUM -100
#define MAX_RAND_NUM 100

void print_array(int *arr, int size){
    if(size > ARR_SIZE_PRINT_LIMIT){
        return;
    }
    for(int i = 0; i < size; ++i){
        printf("%d ", arr[i]);
    }
    printf("\n");
}

void merge_sort(int start_idx, int end_idx, int *arrA, int *arrB){

    //Corner case, segment with length 1
    if(start_idx == end_idx){
        arrB[start_idx] = arrA[start_idx];
        return;
    }

    //Sort each half of the segment separately
    int middle_idx = (start_idx+end_idx)/2;
    merge_sort(start_idx, middle_idx, arrB, arrA);
    merge_sort(middle_idx+1, end_idx, arrB, arrA);

    int left_idx  = start_idx;
    int right_idx = middle_idx+1;
    int idx = start_idx;

    //Sort 2 preordered left and right segments of array A and put the sorted whole segment in array B 
    while(left_idx <= middle_idx || right_idx <= end_idx){

        if(left_idx <= middle_idx && right_idx <= end_idx){
            if(arrA[left_idx] < arrA[right_idx]){
                arrB[idx++] = arrA[left_idx++];
            } else{
                arrB[idx++] = arrA[right_idx++];
            }
        } else if(left_idx <= middle_idx){
            arrB[idx++] = arrA[left_idx++];
        } else{
            arrB[idx++] = arrA[right_idx++];
        }

    }
}


int main(){

    //Random Seed
    srand(time(NULL));

    //Allocate array's A memory
    int *arrA = (int*)malloc(sizeof(int)*ARR_SIZE);

    //Allocate array's B memory
    int *arrB = (int*)malloc(sizeof(int)*ARR_SIZE);

    for(int iteration = 0; iteration < ITERATIONS; ++iteration){

        //Initialize array A and array B with random values
        for(int i = 0; i < ARR_SIZE; ++i){
            arrA[i] = arrB[i] = (rand()%(MAX_RAND_NUM-MIN_RAND_NUM+1))+MIN_RAND_NUM;
        }

        //Print array A
        print_array(arrA, ARR_SIZE);

        //Merge Sort
        merge_sort(0, ARR_SIZE-1, arrA, arrB);

        //Print array B with sorted values
        print_array(arrB, ARR_SIZE);
    }

    //Free allocated memory
    free(arrA);
    free(arrB);

    return 0;
}
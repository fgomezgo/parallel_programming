import java.util.Random;

class merge_sort {

    private static int ITERATIONS = 100;
    private static int ARR_SIZE = 1000000;

    private static int ARR_SIZE_PRINT_LIMIT = 100;

    //Random Values Range
    private static int MIN_RAND_NUM = -100;
    private static int MAX_RAND_NUM = 100;

    //Prints array
    private static void print_array(int[] arr, int size){
        if(size > ARR_SIZE_PRINT_LIMIT){
            return;
        }
        for(int i = 0; i < size; ++i){
            System.out.printf("%d ", arr[i]);
        }
        System.out.println("");
    } 

    //Recursive merge sort
    private static void merge_sort_run(int start_idx, int end_idx, int[] arrA, int[] arrB){

        //Corner case, segment with length 1
        if(start_idx == end_idx){
            arrB[start_idx] = arrA[start_idx];
            return;
        }
    
        //Sort each half of the segment separately
        int middle_idx = (start_idx+end_idx)/2;
        merge_sort_run(start_idx, middle_idx, arrB, arrA);
        merge_sort_run(middle_idx+1, end_idx, arrB, arrA);
    
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

    public static void main(String[] args){

        //Initialize random seed
        Random random = new Random();

        //Declare arrays A and B
        int arrA[] = new int[ARR_SIZE];
        int arrB[] = new int[ARR_SIZE];

        for(int iteration = 0; iteration < ITERATIONS; ++iteration){

            //Initialize array A and array B with random values
            for(int i = 0; i < ARR_SIZE; ++i){
                arrA[i] = arrB[i] = random.nextInt(MAX_RAND_NUM-MIN_RAND_NUM+1)+MIN_RAND_NUM;
            }
    
            //Print array A
            print_array(arrA, ARR_SIZE);
    
            //Recursive merge sort
            merge_sort_run(0, ARR_SIZE-1, arrA, arrB);
    
            //Print whatever array the pointer "to" ended up pointing to (this one stores the final sorted values)
            print_array(arrB, ARR_SIZE);
        }
    }


}
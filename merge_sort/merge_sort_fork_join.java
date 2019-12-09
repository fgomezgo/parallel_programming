import java.util.Random;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

public class merge_sort_fork_join extends RecursiveAction {

    private static int ITERATIONS = 100;
    private static int ARR_SIZE = 1000000;

    private static int ARR_SIZE_PRINT_LIMIT = 100;

    //Random Values Range
    private static int MIN_RAND_NUM = -100;
    private static int MAX_RAND_NUM = 100;

    //Variables for each forked task
    private int[] arrA;
    private int[] arrB;
    private int start_idx, end_idx;

    public merge_sort_fork_join(int start_idx, int end_idx, int[] arrA, int[] arrB){
        this.start_idx  = start_idx;
        this.end_idx    = end_idx;
        this.arrA       = arrA;
        this.arrB       = arrB;
    }

    @Override 
	protected void compute() {
        //Corner case, segment with length 1
        if(start_idx == end_idx){
            arrB[start_idx] = arrA[start_idx];
            return;
        }
    
        //Sort each half of the segment separately
        int middle_idx = (start_idx+end_idx)/2;

        merge_sort_fork_join leftSegment = new merge_sort_fork_join(start_idx, middle_idx, arrB, arrA);
        leftSegment.fork();

        merge_sort_fork_join rightSegment = new merge_sort_fork_join(middle_idx+1, end_idx, arrB, arrA);
        rightSegment.fork();

        leftSegment.join();
        rightSegment.join();
    
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

    public static void main(String[] args){

        //Create fork-join pool
        ForkJoinPool pool;

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
            //merge_sort_run(0, ARR_SIZE-1, arrA, arrB);
            pool = new ForkJoinPool(8);
            pool.invoke(new merge_sort_fork_join(0, ARR_SIZE-1, arrA, arrB));

            //Print whatever array the pointer "to" ended up pointing to (this one stores the final sorted values)
            print_array(arrB, ARR_SIZE);
        }
    }


}
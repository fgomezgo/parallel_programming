import java.util.Random;

class iterative_merge_sort {

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

    private static void sort_2_preordered_segments(int start_idx, int middle_idx, int end_idx, int[] from, int[] to){
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

    private static int[] iterative_merge_sort_run(int[] arrA, int[] arrB){

        //Auxiliar pointers to know to which array; the segments to sort are being passed during each layer
        //Each layer will always try to get the elements to sort from the [FROM] array and put the sorted elements to the [TO] array
        int from[]  = arrA;
        int to[]    = arrB;

        //The size of the segments in each processed layer
        int layer_size;

        //Iterate over all the segment sizes that are powers of two (2, 4, 8, 16, ...) to sort the array by layers of that sizes
        for(layer_size = 2; layer_size <= ARR_SIZE; layer_size <<= 1){
            
            //Each layer swaps the [FROM] and [TO] pointers so the array that was previously the target to store the newly ordered segments
            //is now the one were the ordered segments are extracted from
            int aux[] = from;
            from = to;
            to = aux; 

            //Iterate over each segment in the array [FROM] of size "layer_size" and sort it as 2 separated ordered lists into the new array [TO]
            for(int segment_idx = 0; segment_idx < ARR_SIZE; segment_idx += layer_size){

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

        //An extra layer has to be processed due to the fact that ARR_SIZE is not a perfect power of 2
        if(layer_size > ARR_SIZE){

            //Last layer swaps the [FROM] and [TO] pointers so the array that was previously the target to store the newly ordered segments
            //is now the one were the ordered segments are extracted from
            int aux[] = from;
            from = to;
            to = aux; 

            //Get start, end and middle index of the leftmost and rightmost segments
            int start_idx    = 0;
            int end_idx      = ARR_SIZE-1;
            int middle_idx   = (layer_size>>1)-1;

            //Sort the preordered segments into a new combined ordered segment
            sort_2_preordered_segments(start_idx, middle_idx, end_idx, from, to);
        }

        //Return pointer to final sorted values of the array
        return to;

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
    
            //Iterative merge sort
            int sorted_array[] = iterative_merge_sort_run(arrA, arrB);
    
            //Print whatever array the pointer "to" ended up pointing to (this one stores the final sorted values)
            print_array(sorted_array, ARR_SIZE);
        }
    }
}

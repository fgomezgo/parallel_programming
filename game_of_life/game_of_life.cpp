/*
*   Game Of Life
*
*
*/

#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <string.h>

//STL libraries shortcut
using namespace std;

//Board Dimensions
#define ROWS 10000
#define COLUMNS 10000

//Number of generations to simulate
#define GENERATIONS 20

int* createRow(){

    //Allocate memory as an array
    int* row = (int*) malloc(COLUMNS*sizeof(int));

    //Checks if row has allocated memory, throws error otherwise
    if(row == NULL){
        cout<<"Row not created."<<endl;
        exit(1);
    }

    return row;
}

int** createBoard(){

    //Allocate memory as a matrix
    int** board = (int**) malloc(ROWS*sizeof(int*));

    //Checks if board has allocated memory, throws error otherwise
    if(board == NULL){
        cout<<"Board not created."<<endl;
        exit(1);
    }

    for(int i = 0; i < ROWS; ++i){
        board[i] = createRow(); 
    }

    return board;
}

void initializeBoard(int** board){

    //Checks if board has allocated memory, throws error otherwise
    if(board == NULL){
        cout<<"Board is not created."<<endl;
        exit(1);
    }
    
    // Initialize random seed
    srand (time(NULL));

    //Randomized initialization of alive and dead cells in the board
    //Uses individual bits of the randomized value to reduce by a factor of 31 the number of calls to rand()

    int randBitIdx = 0;
    int randomBitMask = rand();

    for(int i = 0; i < ROWS; ++i){
        for(int j = 0; j < COLUMNS; ++j){
            if(randBitIdx == 31){
                randBitIdx = 0;
                randomBitMask = rand();
            }
            board[i][j] = ( randomBitMask & (1<<randBitIdx) )? 1:0;
            ++randBitIdx; 
        }
    }

}

void showBoard(int** board){
    //Checks if board has allocated memory, throws error otherwise
    if(board == NULL){
        cout<<"Board is not created."<<endl;
        exit(1);
    }

    //Prints board representing alive cells with a '*' and dead cells with a '.'
    for(int i = 0; i < ROWS; ++i){
        for(int j = 0; j < COLUMNS; ++j){
            if(board[i][j]){
                cout<<"*";
            } else{
                cout<<".";
            }
        }
        cout<<endl;
    }
}

int checkRules(int neighbors, int state){
    //state = 0 -> cell is dead
    //state = 1 -> cell is alive

    //Cell is dead
    if(state == 0){
        //Is born
        if(neighbors == 3){
            return 1;
        } else{
            //Remains dead
            return 0;
        }
    } else{ //Cell is alive
        //Remains alive
        if(neighbors == 2 || neighbors == 3){
            return 1;
        } else{ //Dies
            return 0;
        }
    }

    return 0;
}

//Checks whether such position exists in the board or not
bool validPosition(const int &row, const int &col){
    return (row >= 0 && col >= 0 && row < ROWS && col < COLUMNS);
}

int getNeighbors(const int &row, const int &col, int** board, int* prevRow){

    int numberOfNeighbors = 0;

    int dr[8] = {0,0,1,-1,1,-1,1,-1};
    int dc[8] = {1,-1,0,0,-1,1,1,-1};

    //Iterates over each possible neighbor (8)

    for(int neighbor = 0; neighbor < 8; ++neighbor){
        int nr = row+dr[neighbor];
        int nc = col+dc[neighbor];

        //Checks if current neighbor exists or if its out of board's boundaries
        if(validPosition(nr,nc)){

            //if it's a neighbor from above, check the stored Row (which keeps the previous value) instead of the current board
            if(dr[neighbor] == -1){
                if(prevRow[nc])    ++numberOfNeighbors;
            } else{
                if(board[nr][nc])  ++numberOfNeighbors;
            }
        }
    }

    return numberOfNeighbors;
}

void nextGeneration(int** board, int* prevRow, int* currRow){

    //Initializes prevRow with zeroes
    memset(prevRow, 0, COLUMNS*sizeof(int));

    for(int i = 0; i < ROWS; ++i){
        
        //Makes a copy of the current row
        memcpy(currRow, board[i], COLUMNS*sizeof(int));

        for(int j = 0; j < COLUMNS; ++j){

            //Get how many neighbors are alive around this cell
            int neighbors = getNeighbors(i,j,board,prevRow);

            //Apply rules to determinte the current cell's state
            board[i][j] = checkRules(neighbors, board[i][j]);
        }

        //Swaps currRow and prevRow pointers
        int* auxPtr = prevRow;
        prevRow = currRow;
        currRow = auxPtr;
    }
}

void runGame(int generations, int** board){

    //Checks if board has allocated memory, throws error otherwise
    if(board == NULL){
        cout<<"Board is not created."<<endl;
        exit(1);
    }

    //Allocates blocks of memory for auxiliar variables
    int* auxRow  = createRow();
    int* auxRow2 = createRow();

    //Iterates through generations one by one

    for(int generation = 0; generation <= generations; ++generation){
        
        //Show board in current generation if board size is less than 1000
        if(ROWS*COLUMNS <= 1000){
            cout<<"Generation "<<generation<<":"<<endl;
            showBoard(board);
        } else{
            cout<<"Generation "<<generation<<"..."<<endl;
        }

        //Simulates next generation in the board
        nextGeneration(board, auxRow, auxRow2);
    }
}


int main(){

    int **board = createBoard();
    initializeBoard(board);
    runGame(GENERATIONS, board);

    return 0;
}

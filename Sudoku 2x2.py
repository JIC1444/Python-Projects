import numpy as np

a=np.array([[0,2,4,0],[1,0,0,3],[4,0,0,2],[0,1,3,0]])
print(a)
row = 2
setcol=0
setrow=0
col = 0
poss = np.array([1,2,3,4])
blacklist = []
#setup, poss is all possible numbers that can appear in 2x2 sudoku
#checkmat is the the  blacklist matrix


zeroes=np.array([])
zeroes = np.where(a==0)
zeroes_row=zeroes[0]
zeroes_col=zeroes[1]
print(zeroes_row)
print(zeroes_col)
#gives 2 arrays with matrix positions of all the zeroes in the 4x4 grid

current_idx=0

row=zeroes_row[current_idx]
col=zeroes_col[current_idx]
print(row, col)


#checks if the indexed number is in the same row and adds it to the blacklist
def solve_index():
    global row
    global col
    global zeroes_row
    global zeroes_col
    global blacklist
    global poss
    global a
    global current_idx
    poss = np.array([1,2,3,4])
    blacklist = []

    #row and col funtions are constant and correspond to current index being solved
    row=zeroes_row[current_idx]
    col=zeroes_col[current_idx]

    #current row and col array gives the numbers of the row of the current index
    current_row_arr=[]
    current_col_arr=[]
    current_row_arr =np.array([a[row,:]])
    current_col_arr = np.array([a[:,col]])

    for x in range(0,4):
        if current_row_arr[0,x]!=0:
            swap = a[row,x]
            if swap not in blacklist:
                blacklist=np.append(blacklist, swap)
        elif a[row,col] == 0:
            print("")

    #checks if the indexed number is in the same column and adds it to the blacklist

    for y in range(0,4):
        if current_col_arr[0,y]!=0:
            swap =current_col_arr[0,y]
            if swap not in blacklist:
                blacklist=np.append(blacklist, swap)
        elif a[row,col] == 0:
             print("")
             
    #creates the index's box 


    index = [row,col]
    if row<=1:
        if col<=1:
            index_box=np.array([a[0,0],a[0,1],a[1,0],a[1,1]])
        else:
            index_box=np.array([a[0,2],a[0,3],a[1,2],a[1,3]])
    else:
        if col<=1:
            index_box=np.array([a[2,0],a[2,1],a[3,0],a[3,1]])
        else:
            index_box=np.array([a[2,2],a[2,3],a[3,2],a[3,3]])

    #checks the 2x2 square for numbers to blacklist  

    for f in range(0,4):
        checkb=index_box[(f)]
        if checkb!=0:
            swap = checkb
            if swap not in blacklist:
                blacklist=np.append(blacklist, swap)
        elif a[row,col] == 0:
             print("")


    #if the length of the array is one then just inserts the number
    blacklist=np.delete(blacklist, np.where(blacklist==0))
    length=len(blacklist)
    #poss2 is the possible numbers minus checkmat (done through the remove variable)

    for z in range(0,length):
        remove = blacklist[z]
        poss = np.delete(poss, np.where(poss == remove))
        
    length2=len(poss)
    if length2 == 1:
        a[row,col] = poss
    else:
        poss3=poss

    zeroes_row=np.delete(zeroes_row,current_idx)
    zeroes_col=np.delete(zeroes_col,current_idx)
    


notsolved=True
while notsolved==True:
    solve_index()
    print(a)
    if 0 not in a:
        notsolved=False
print("Solved!")









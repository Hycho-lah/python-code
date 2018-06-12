# NW.py
# HW1, Computational Genomics, Fall 2018
# andrewid: eye
# Implementation of Needleman-Wunsch algorithm for Sequence Alignment with the scoring function below:
#           Match = 0
#           Mismatch = -1
#           Gap = -1
#The script can be run with command line: puthon NW.py example.fasta
#The function return a single value, a string with three lines with the following format:
#           alignment score
#           alignment in the first sequence
#           alignment in the second sequence
#Each of the three items are in separate lines

import sys

def ReadFASTA(filename):
    fp=open(filename, 'r')
    Sequences={}
    tmpname=""
    tmpseq=""
    for line in fp:
        if line[0]==">":
            if len(tmpseq)!=0:
                Sequences[tmpname]=tmpseq
            tmpname=line.strip().split()[0][1:]
            tmpseq=""
        else:
            tmpseq+=line.strip()
    Sequences[tmpname]=tmpseq
    fp.close()
    return Sequences

# You may define any helper functions for Needleman-Wunsch algorithm here

def scores(di_score,ho_score, ve_score, match):
    if match == True:
        di_update = di_score + 0
    elif match == False:
        di_update = di_score - 1
    ho_update = ho_score - 1
    ve_update = ve_score - 1
    return di_update, ho_update, ve_update

def traceback(score_matrix,pointer_matrix, seq1, seq2):
    rows = len(score_matrix)
    columns = len(score_matrix[0])
    align_score = score_matrix[-1][-1]
    pos_y = rows-1
    pos_x = columns-1
    align_seq1 = []
    align_seq2 = []
    seq_pos = -1
    while pos_y != 0 and pos_x != 0:
        pointer = pointer_matrix[pos_y][pos_x]
        score = score_matrix[pos_y][pos_x]
        if pointer == 'D':
            align_seq1 = [seq1[pos_y-1]] + align_seq1
            align_seq2 = [seq2[pos_x-1]] + align_seq2
            pos_y -= 1
            pos_x -= 1
        elif pointer =='V':
            align_seq1 = [seq1[pos_y - 1]] + align_seq1
            align_seq2 = ["-"] + align_seq2
            pos_y -= 1
        elif pointer =='H':
            align_seq1 = ["-"] + align_seq1
            align_seq2 = [seq2[pos_x - 1]] + align_seq2
            pos_x -= 1
        seq_pos -= 1
    align_seq1 = ''.join(align_seq1)#convert list to string
    align_seq2 = ''.join(align_seq2)
    return align_score, align_seq1, align_seq2

def NW_matrix(seq1,seq2):
    height = len(seq1) + 1
    width = len(seq2) + 1

    # Creates a list containing "height" lists, each of "width" items, all set to 0
    score_matrix = [[0 for x in range(width)] for y in range(height)]
    pointer_matrix = [[0 for x in range(width)] for y in range(height)]
    # Fill up gaps
    for i in range(0, height):
        score_matrix[i][0] = -1 * i
    for i in range(0, width):
        score_matrix[0][i] = -1 * i

    for y in range(1, height):
        for x in range(1, width):
            di_score = score_matrix[y - 1][x - 1]
            ve_score = score_matrix[y - 1][x]
            ho_score = score_matrix[y][x - 1]
            if seq1[y - 1] == seq2[x - 1]:
                match = True
            else:
                match = False
            di_update, ho_update, ve_update = scores(di_score, ho_score, ve_score, match)
            new_score = max(di_update, ho_update, ve_update)
            score_matrix[y][x] = new_score
            d = {'D': di_update, 'H': ho_update, 'V': ve_update}
            pointer = max(d, key=d.get)
            pointer_matrix[y][x] = pointer
            # if new_score == di_update:
            # pointer_matrix[y][x] = 'D'
            # elif new_score == ho_update:
            # pointer_matrix[y][x] = 'H'
            # elif new_score == ve_update:
            # pointer_matrix[y][x] = 'V'
    return score_matrix, pointer_matrix

# Do not change this function signature
def needleman_wunsch(seq1, seq2): #takes in two strings seq1 and seq2
    """Find the global alignment for seq1 and seq2
    Returns: a string of three lines like so:
    '<alignment score>\n<alignment in seq1>\n<alignment in seq2>'
    """
    score_matrix, pointer_matrix = NW_matrix(seq1, seq2)
    align_score, align_seq1, align_seq2 = traceback(score_matrix, pointer_matrix, seq1, seq2)
    output_string = str(align_score) + "\n" + str(align_seq1) + "\n" + str(align_seq2)
    return output_string

if __name__=="__main__":
    Sequences=ReadFASTA(sys.argv[1])
    assert len(Sequences.keys())==2, "fasta file contains more than 2 sequences."
    Sequences = ReadFASTA("example.fasta") # just for testing purposes
    seq1=Sequences[list(Sequences.keys())[0]] #converts sequences to strings
    seq2=Sequences[list(Sequences.keys())[1]]
    #output_string, align_seq1, align_seq2 = needleman_wunsch(seq1, seq2)
    print(needleman_wunsch(seq1, seq2))

#print(ReadFASTA("example.fasta"))
#a = ReadFASTA("example.fasta")
#print(len(a))
#seq1=a[list(a.keys())[0]]
#seq2=a[list(a.keys())[1]]
#seq1 = "ABCDE"
#seq2 = "CDEFG"
#print(type(seq1))
#print(type(seq1))
#print(seq1)
#score_matrix, pointer_matrix = NW_matrix(seq1,seq2)
#print(score_matrix)
#print(len(score_matrix))
#print(len(score_matrix[0]))
#print(len(seq1))
#print(len(seq2))

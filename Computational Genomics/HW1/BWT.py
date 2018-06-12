# BWT.py
# HW1, Computational Genomics, Fall 2018
# andrewid: eye

# This code contains a series of functions related to the Burrows-Wheeler Transform (BWT)
# BWT is a technique to transform a sequence into one that has more repeated runs of characters.

def rle(s):
    """Run Length Encoder
    Args: s, string to be encoded
    Returns: RLE(s)
    """
    s_rle = ""
    repeats = 0
    for i in range(0,len(s)):
        if i != 0:
            if s[i] == s[i-1]:
                repeats +=1
            else:
                if repeats != 0:
                    s_rle += str(s[i - 1]) + str(repeats + 1) + str(s[i])
                elif repeats == 0:
                    s_rle += str(s[i])
                repeats = 0
        elif i == 0:
            s_rle += str(s[i])
    return s_rle

def bwt_encode(s):
    """Burrows-Wheeler Transform
    Args: s, string, which must not contain '{' or '}'
    Returns: BWT(s), which contains '{' and '}'
    """
    #"""Apply Burrows-Wheeler transform to input string. Not indicated by a unique byte but use index list"""
    # Table of rotations of string
    s = "{" + s + "}"
    table = [s[i:] + s[:i] for i in range(len(s))]
    # Sorted table
    table_sorted = table[:]
    table_sorted.sort()
    # Get index list of ((every string in sorted table)'s next string in unsorted table)'s index in sorted table
    indexlist = []
    for t in table_sorted:
        index1 = table.index(t)
        index1 = index1 + 1 if index1 < len(s) - 1 else 0
        index2 = table_sorted.index(table[index1])
        indexlist.append(index2)
    # Join last characters of each row into string
    r = ''.join([row[-1] for row in table_sorted])

    return r

def bwt_decode(bwt):
    # """Inverse Burrows-Wheeler Transform
    # Args: bwt, BWT'ed string, which should contain '{' and '}'
    # Returns: reconstructed original string s, must not contains '{' or '}'

    table = [[0 for x in range(len(bwt))] for y in range(len(bwt))]

    #fill in last column
    for b in range(0,len(bwt)):
        table[b][-1] = bwt[b]

    sorted_bwt = ''.join(sorted(bwt))
    #fill in first column
    for f in range(0,len(bwt)):
        table[f][0] = sorted_bwt[f]

    #fill in rest of table
    for x in range(1, len(bwt)):
        k = x+1
        k_mer = [[0 for x in range(k)] for y in range(len(bwt))]
        for r in range(0,len(bwt)): # set first column of k-mer as bwt
            k_mer[r][0] = bwt[r]
        for h in range(1,k):
            for g in range(0,len(bwt)):
                k_mer[g][h] = table[g][h-1]
        #convert each row of list into string
        for l in range(0,len(bwt)):
            k_mer[l] = ''.join(k_mer[l])
        #sort each row in alphabetical order
        k_mer = sorted(k_mer)
        #convert string to list for each row
        for l in range(0,len(bwt)):
            k_mer[l] = list(k_mer[l])
        #fill in k_mer to table
        for y in range(0, k):
            for z in range(0,len(bwt)):
                table[z][y] = k_mer[z][y]

    for i in range(0,len(bwt)):
        if table[i][0] == '{':
            s = ''.join(table[i][1:-1])
    s = ''.join(c for c in s if c not in '{')
    s = ''.join(c for c in s if c not in '}')
    return s


def test_string(s):
    compressed = rle(s)
    bwt = bwt_encode(s)
    compressed_bwt = rle(bwt)
    reconstructed = bwt_decode(bwt)
    template = "{:25} ({:3d}) {}"
    print(template.format("original", len(s), s))
    print(template.format("bwt_enc(orig)", len(bwt), bwt))
    print(template.format("bwt_dec(bwt_enc(orig))", len(reconstructed), reconstructed))
    print(template.format("rle(orig)", len(compressed), compressed))
    print(template.format("rle(bwt_enc(orig))", len(compressed_bwt), compressed_bwt))
    print()
    print()

if __name__ == "__main__":
    # Add more of your own strings to explore for question (i)
    test_strings = ["WOOOOOHOOOOHOOOO!", "scottytartanscottytartanscottytartanscottytartan"]
    for s in test_strings:
        test_string(s)
    #print(len(rle("scottytartanscottytartanscottytartanscottytartan")))



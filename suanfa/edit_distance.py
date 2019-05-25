import difflib

def normal_leven(str1, str2):  #使用一维数组实现
    len_str1 = len(str1) + 1
    len_str2 = len(str2) + 1
    # create matrix
    matrix = [0 for n in range(len_str1 * len_str2)]
    # init x axis
    for i in range(len_str1):
        matrix[i] = i
    # init y axis
    for j in range(0, len(matrix), len_str1): #可以进行跨数的取值，例如：range(0,42,6)为：[0,6,12,18.24.30.36]
        if j % len_str1 == 0:
            matrix[j] = j // len_str1 #整除取商

    for i in range(1, len_str1):
        for j in range(1, len_str2):
            if str1[i - 1] == str2[j - 1]:
                cost = 0
            else:
                cost = 1
            matrix[j * len_str1 + i] = min(matrix[(j - 1) * len_str1 + i] + 1,
                                           matrix[j * len_str1 + (i - 1)] + 1,
                                           matrix[(j - 1) * len_str1 + (i - 1)] + cost)

    return matrix[-1]


def edit(str1, str2):  #与上面算法相同思想，使用二维数组实现，比上面的方法更加直观和简洁
    matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]

    for i in xrange(1, len(str1) + 1):
        for j in xrange(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)

    return matrix[len(str1)][len(str2)]


def difflib_leven(str1, str2):
    leven_cost = 0
    s = difflib.SequenceMatcher(None, str1, str2)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        # print('{:7} a[{}: {}] --> b[{}: {}] {} --> {}'.format(tag, i1, i2, j1, j2, str1[i1: i2], str2[j1: j2]))

        if tag == 'replace':
            leven_cost += max(i2 - i1, j2 - j1)
        elif tag == 'insert':
            leven_cost += (j2 - j1)
        elif tag == 'delete':
            leven_cost += (i2 - i1)
    return leven_cost


if __name__ == '__main__':
    print(normal_leven('a','cba'))
    print(normal_leven('ab','cba'))
    print(normal_leven('11','cba'))
    print(normal_leven('1','cba'))
    print(normal_leven('batyu','beauty'))

    print("~~~~~~~~~~~~~~~~~")
    print(difflib_leven('a','cba'))
    print(difflib_leven('ab', 'cba'))
    print(difflib_leven('11', 'cba'))
    print(difflib_leven('1', 'cba'))

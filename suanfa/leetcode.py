def longestCommonPrefix(strs):
    """
    :type strs: List[str]
    :rtype: str
    """
    if strs:
        minLen = len(strs[0])
        for str in strs:
            if len(str) < minLen:
                minLen = len(str)

        for i in range(minLen):
            # flag = True
            char = strs[0][i]
            for str in strs:
                if str[i] != char:
                    # flag = False
                    return strs[0][:i]
    else:
        return ""


if __name__ == "__main__":
    # print(longestCommonPrefix(["flower","flow","flight"]))
    print(longestCommonPrefix(["a","ab","abc"]))
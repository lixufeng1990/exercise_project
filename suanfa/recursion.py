def recursive(i):
    sum = 0
    if (0 == i):
        return 1
    else:
        sum = i * recursive(i - 1)


    return sum;

def test_recursive():
    while True:
        input_num = input("请输入一个数字：")
        int_input_num = int(input_num)
        if int_input_num != 0:
            print(recursive(int_input_num))
        else:
            break

if __name__ == "__main__":
    test_recursive()
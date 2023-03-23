def func(str):
    print(f"String len is: {len(str)}")
    print(str[0])
    print(str[1])
    print(str[2])

def string_seq(seq):
    for index, value in enumerate(seq):
        print(f"index {index}, value {value}")

def string_seq_reverse(seq):
    for index, value in reversed(list(enumerate(seq))):
        print(f"index {index}, value {value}")

if __name__=="__main__":
    func("xyz")
    string_seq('xyz')
    print("s" == "s")
    string_seq_reverse('zyx')
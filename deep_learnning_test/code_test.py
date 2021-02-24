from functools import reduce

def f(x):
    return x * x

def add(x,y):
    return x + y

def test_map():
    #map返回一个惰性序列，可迭代对象，需要转化为list
    #map()函数接收两个参数，一个是函数，一个是Iterable，map将传入的函数依次作用到序列的每个元素，并把结果作为新的Iterator返回。
    list1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    L = list(map(f, list1))
    print(L)

    #reduce把一个函数作用在一个序列[x1, x2, x3, ...]上，这个函数必须接收两个参数，reduce把结果继续和序列的下一个元素做累积计算。返回的是一个计算的最终结果，函数接收两个参数：
    ll = reduce(add, list(map(f, list1)))
    print(ll)
    lll = reduce(lambda x, y: x*y, list(map(f, list1)))
    print(lll)

if __name__ == '__main__':
    print('HELLO')
    test_map()


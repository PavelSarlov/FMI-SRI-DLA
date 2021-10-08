x=5
x+1
x+=1
x
print("Hello World!")
print(x*10)

example_list = [1, 2, '3', 'four']
example_set = set([1, 2, '3', 'four', 'four'])
example_dictionary = {'1':'one', '2':'two', '3':'three'}

list_of_lists = [[1,2,3],[4,5,6],[7,8,9]]
three = list_of_lists[0][2]
four = list_of_lists[1][0]

my_list = [i for i in range(10)]
my_list2 = [i**2 for i in range(10)]
initialize_2d_list = [[i + j for i in range(5)] for j in range(10)]

random_list = [3,12,5,6]
sorted_list = sorted(random_list)
random_list = [(3,'A'),(12,'D'),(5,'M'),(6,'B')]
sorted_list = sorted(random_list, key = lambda x: x[1])

############### functions, loops, control flow

def myFunction(a,b):
    for num1 in range(a):
        if num1 % 2 == 0 and num1 % 4 == 0:
            print(str(num1) + " is multiple of four!")
        elif num1 % 2 == 0 and num1 % 4 != 0:
            print(str(num1) + " is even, but not a multiple of four!")
        else:
            print(str(num1) + " is odd!")
    for num2 in range(1,b,2):
        print(num2, 2**num2)

def main():
    a = 5
    b = 10
    myFunction(a,b)

if __name__ == '__main__':
    main()

############### classes

class Vehicle:

    def __init__(self, make, name, year, is_electric=False, price=100):
        self.name = name
        self.make = make
        self.year = year
        self.is_electric = is_electric
        self.price = price

        self.odometer = 0

    def drive(self, distance):
        self.odometer += distance

    def compute_price(self):
        if self.odometer == 0:
            price = self.price
        elif self.is_electric:
            price = self.price / (self.odometer * 0.8)
        else:
            price = self.price / self.odometer
        return price

if __name__ == '__main__':
    family_car = Vehicle('Honda', 'Accord', '2019', price=10000)

    print(family_car.compute_price())
    family_car.drive(100)
    print(family_car.compute_price())

############### numpy

import numpy as np

ones = np.ones(10)
randomMatrix = np.random.rand(5,10)
fromPythonList = np.array([[0,1,2],[3,4,5],[6,7,8]])

x = np.array([1,0,0,1])
y = np.array([-1,5,10,-1])
x+y

A = np.array([[1,0],[0,1]])
B = np.array([[0,1],[1,0]])
A
B
A+B

A = np.array([[5,10],[3,4]])
B = np.array([[6,20],[-4,-5]])
A*B

x = np.array([1,2,3,4])
y = np.array([5,10,15,20])
np.dot(x,y)
sum([i * j for (i,j) in zip(x,y)])

A = np.array([[1,10],[2,5],[3,3]])
x = np.array([3,4])
np.dot(A,x)

A = np.array([[1,5],[2,3],[3,10]])
B = np.array([[3,4],[4,5]])
A
B
np.dot(A,B)
np.matmul(A,B)

x = np.array([1,10,15,100])
x+10

A = np.array([[1,10],[15,20],[25,50]])
x = np.array([5,100])
A.shape
x.shape
A+x

x = np.array([0, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
x
np.sin(x)
A = np.array([[1,3],[2,4],[3,5]])
A
np.sum(A)
np.sum(A,axis=0)
np.sum(A,axis=1)

a = np.array([i for i in range(100000)])
b = np.array([i for i in range(100000)])

import time

tic = time.time()
dot = 0.0
for i in range(len(a)):
    dot += a[i] * b[i]
toc = time.time()

print("dot_product = "+str(dot))
print("Computation time = " + str(1000*(toc-tic)) + "ms")

n_tic = time.time()
n_dot_product = np.array(a).dot(np.array(b))
n_toc = time.time()

print("\nn_dot_product = "+str(n_dot_product))
print("Computation time = " + str(1000*(n_toc-n_tic)) + "ms")

myListFor = [i for i in range(100000)]
tic = time.time()
for i in range(len(myListFor)):
    myListFor[i] = np.sin(myListFor[i])
toc = time.time()

myListMap = [i for i in range(100000)]
mtic = time.time()
myListMap = list(map(np.sin,myListMap))
mtoc = time.time()

myListNumpy = [i for i in range(100000)]
numpytic = time.time()
myListNumpy = np.sin(myListNumpy)
numpytoc = time.time()

print("for_loop = " + str(1000*(toc-tic)) + "ms")
print("map = " + str(1000*(mtoc-mtic)) + "ms")
print("numpy = " + str(1000*(numpytoc-numpytic)) + "ms")

########################### plotting

import matplotlib.pyplot as plt

x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine','Cosine'])
plt.show()



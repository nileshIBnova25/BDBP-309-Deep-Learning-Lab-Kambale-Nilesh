import cmath

x1=[1,2,3,4]
x2=[5,6,7,8]
x3=[9,10,11,12]
x=[x1,x2,x3]

def myfun(x,pos):
    final= []
    mylist = []
    lenx= len(x[0])
    print(lenx/2)
    for i in range(pos):
        for j in range(int(lenx/2)):
            exp = i/4
            mylist.append((cmath.sin(i/(10000**(exp)))))
            mylist.append((cmath.cos(i/(10000**(exp)))))
        final.append([mylist])
    return final

result = myfun(x,2)
print(result)









#list = []
#for pos in range(len(1000)):


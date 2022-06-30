# -*- coding: utf-8 -*-
### BeakJun ###

## For loop
# def gugu_class(N):
#     N > 0 and N <= 10
#     for _ in range(1,10):
#         print(N,'*',_,'=',N*_)

# gugu_class(12)

# while loop
def gugu_class2(N):
    while N > 0 and N < 10:
        for i in range(1, 10):
            print(N,'*',i,'=',N*i)
        break
            
gugu_class2(3)
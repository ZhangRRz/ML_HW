import matplotlib.pyplot as plt
import numpy as np

while (True):
    print("please input with format : a,b,c,d and all need to be a number")
    inputStr = input()
    strlist = inputStr.split(',')
    if (strlist.__len__() != 4):
        continue
    try:
        a = int(strlist[0])
        b = int(strlist[1])
        c = int(strlist[2])
        d = int(strlist[3])
    except:
        continue
    else:
        break
print("please input with format : ax,ay,bx,by,cx,cy... and all need to be a number")
fivePtXYarray = []
while (True):
    fivePtStr = input()
    fivePtStrlist = fivePtStr.split(',')
    if (fivePtStrlist.__len__() % 2 != 0):
        print("Error! please check the format")
        continue
    try:
        for i in range(0, fivePtStrlist.__len__()):
            fivePtXYarray.append(int(fivePtStrlist[i]))
    except:
        print("Error! please check the format")
        fivePtXYarray.clear()
        continue
    else:
        break


X = np.linspace(-25, 25, 100)

Y = a * (X ** 3) + b * (X ** 2) + c * (X ** 1) + d
plt.plot(X, Y, color="purple", linewidth=1.0, linestyle="-",
         label="%dx^3+%dx^2+%dx+%d" % (a, b, c, d))

plt.ylim(-5000.0, 5000.0)
plt.xlim(-50.0, 50.0)

redFlag = False
greenFlag = False
blueFlag = False
for i in range(0, int(fivePtStrlist.__len__() / 2.0)):
    pt_x = fivePtXYarray[2 * i]
    pt_y = fivePtXYarray[2 * i + 1]
    eq_y = a * (pt_x ** 3) + b * (pt_x ** 2) + c * (pt_x ** 1) + d

    if pt_y < eq_y:
        if (blueFlag):
            plt.plot(pt_x, pt_y, color="blue", marker="o")
        else:
            plt.plot(pt_x, pt_y, color="blue", marker="o", label="lower")
            blueFlag = True
    elif pt_y > eq_y:
        if (redFlag):
            plt.plot(pt_x, pt_y, color="red", marker="o")
        else:
            plt.plot(pt_x, pt_y, color="red", marker="o", label="higher")
            redFlag = True
    else:
        if (greenFlag):
            plt.plot(pt_x, pt_y, color="green", marker="o")
        else:
            plt.plot(pt_x, pt_y, color="green", marker="o", label="equal")
            greenFlag = True

plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.)
plt.xlabel("X axis")
plt.ylabel("Y axis")

plt.show()

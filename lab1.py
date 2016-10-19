import numpy as np
import numpy.linalg as la

file = open("data.txt")
data = np.genfromtxt(file, delimiter=",")
file.close()

print "Data:\n", data

# Create matrices for M and b
M = np.zeros([len(data)*2,6])
b = np.zeros([len(data)*2,1])

M = np.matrix(M)
b = np.matrix(b)

# Copy values into M
for i in range(0, len(data)):
	M[2 * i, 0] = data[i][2]
	M[2 * i, 1] = data[i][3]
	M[2 * i, 2] = 1
	M[(2 * i) + 1, 3] = data[i][2]
	M[(2 * i) + 1, 4] = data[i][3]
	M[(2 * i) + 1, 5] = 1
print "M:\n", M

# Copy values into b
for i in range(0, len(data)):
	b[2 * i] = data[i][0]
	b[(2 * i) + 1] = data[i][1]
print "b:\n", b

a, e, r, s = la.lstsq(M, b)
print "a:\n", a

print M * a

sum_squared_err = (la.norm(M*a-b))**2
print "Sum-squared error: ", sum_squared_err
print "Residue: ", e

	
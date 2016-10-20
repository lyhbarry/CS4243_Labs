# CS4243 - Lab 6 & 7
# Name: Lau Yun Hui Barry
# Matric no.: A0111682M
# Fri 10am session

import math
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

	
PI = math.pi


def quatmult(q1, q2):
	""" Quaternion multiplication """
	out = [0, 0, 0, 0]

	p0, p1, p2, p3 = q1
	q0, q1, q2, q3 = q2

	out[0] = p0 * q0 - p1 * q1 - p2 * q2 - p3 * q3
	out[1] = p0 * q1 + p1 * q0 + p2 * q3 - p3 * q2
	out[2] = p0 * q2 - p1 * q3 + p2 * q0 + p3 * q1
	out[3] = p0 * q3 + p1 * q2 - p2 * q1 + p3 * q0

	print out

	return out

def quat2rot(q):
	""" Quaternion to rotation matrix """	
	q0, q1, q2, q3 = q

	Rq = [[q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2,
			2 * (q1 * q2 - q0 * q3),
			2 * (q1 * q3 + q0 * q2)],
		   [2 * (q1 * q2 + q0 * q3),
		    q0 ** 2 + q2 ** 2 - q1 ** 2 - q3 ** 2,
		    2 * (q2 * q3 - q0 * q1)],
		   [2 * (q1 * q3 - q0 * q2),
		    2 * (q2 * q3 + q0 * q1),
		    q0 ** 2 + q3 ** 2 - q1 ** 2 - q2 ** 2]]

	return np.matrix(Rq)

if __name__ == "__main__":
	### PART 1 - Defining the scene points and camera position and orientations
	
	## PART 1.1 - Define the Shape
	pts = np.zeros([11, 3])
	pts[0, :] = [-1, -1, -1]
	pts[1, :] = [1, -1, -1]
	pts[2, :] = [1, 1, -1]
	pts[3, :] = [-1, 1, -1]
	pts[4, :] = [-1, -1, 1]
	pts[5, :] = [1, -1, 1]
	pts[6, :] = [1, 1, 1]
	pts[7, :] = [-1, 1, 1]
	pts[8, :] = [-0.5, -0.5, -1]
	pts[9, :] = [0.5, -0.5, -1]
	pts[10, :] = [0, 0.5, -1]

	## PART 1.2 - Define the Camera Translation
	teta = -30 * PI / 180
	w1, w2, w3 = [0, 1, 0]

	ct_0 = [0, 0, 0, -5]									# Camera translation 1
	# Quarternion
	q = [math.cos(teta / 2.0),
		  math.sin(teta / 2.0) * w1,
		  math.sin(teta / 2.0) * w2,
		  math.sin(teta / 2.0) * w3]

	ct_1 = quatmult(quatmult(ct_0, q), np.conjugate(q))		# Camera translation 2			
	ct_2 = quatmult(quatmult(ct_1, q), np.conjugate(q))		# Camera translation 3
	ct_3 = quatmult(quatmult(ct_2, q), np.conjugate(q))		# Camera translation 4
	cts = [ct_0, ct_1, ct_2, ct_3]

	print cts

	## Part 1.3	- Define the Camera Orientation
	teta = 30 * PI / 180
	quatmat_0 = np.asmatrix([[1, 0, 0],			# Camera orientation 1
		  		 			 [0, 1, 0],
		  		 			 [0, 0, 1]])
	q = [math.cos(teta / 2.0),
		  math.sin(teta / 2.0) * w1,
		  math.sin(teta / 2.0) * w2,
		  math.sin(teta / 2.0) * w3]

	Rq = quat2rot(q)		# Rotation matrix
	
	quatmat_1 = Rq * quatmat_0					# Camera orientation 2
	quatmat_2 = Rq * quatmat_1					# Camera orientation 3
	quatmat_3 = Rq * quatmat_2					# Camera orientation 4
	quatmats = [quatmat_0, quatmat_1, quatmat_2, quatmat_3]

	## PART 2 - Projecting 3D shape points onto camera image planes
	# Setting variables
	u_0 = 0		# Image center horizontal offset
	v_0 = 0		# Image center vertical offset
	B_u = 1		# Pixel scaling factor in horizontal direction
	B_v = 1		# Pixel scaling factor in vertical direction
	k_u = 1		# Camera optical x-axis
	k_v = 1		# Camera optical z-axis
 	f = 1		# Focal length

 	# Orthographic projection
 	sp_idx = 1 	
 	fig = plt.figure()
	colors = cm.rainbow(np.linspace(0, 1, 11))
	ctqm = zip(cts, quatmats)
	ptcl = zip(pts, colors)
 	for ct, qm in ctqm:
		fig.add_subplot(2, 2, sp_idx) 		
 		for pt, c in ptcl:
 			u_fp = np.transpose(pt - ct[1:]) * qm[:, 0] * B_u + u_0
 			v_fp = np.transpose(pt - ct[1:]) * qm[:, 1] * B_v + v_0
 			plt.scatter(u_fp, v_fp, color=c)
 		sp_idx += 1
 	plt.savefig('orthographic_proj.png')

 	# Perspective projection
	sp_idx = 1 	
	k_f = np.transpose(np.asmatrix([k_u, 0, k_v]))
 	fig = plt.figure()
	colors = cm.rainbow(np.linspace(0, 1, 11))
	ctqm = zip(cts, quatmats)
	ptcl = zip(pts, colors)
 	for ct, qm in ctqm:
		fig.add_subplot(2, 2, sp_idx) 		
 		for pt, c in ptcl:
 			u_fp = ((f * np.transpose(pt - ct[1:]) * qm[:, 0]) / (np.transpose(pt - ct[1:]) * k_f)) * B_u + u_0
 			v_fp = ((f * np.transpose(pt - ct[1:]) * qm[:, 1]) / (np.transpose(pt - ct[1:]) * k_f)) * B_v + v_0
 			plt.scatter(u_fp, v_fp, color=c)
 		sp_idx += 1
 	plt.savefig('perspective_proj.png')

 	## PART 3 - Homography






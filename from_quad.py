from __future__ import division
import numpy as np
import cv2

import server
import utils

np.set_printoptions(suppress=True)

# camera pose, in meter units
pose = None

window_name = 'res'

@server.data_handler
def update(alpha, beta, gamma):
	# alpha(Y) - 0 side level - increase tilting away from you
	# beta (X) - 0 top level - increase tilting right
	# gamma(Z) - 0 north    - increase rotating counterclockwise
	print int(alpha), int(beta), int(gamma)
	global pose

	# ry = utils.rot_y(np.deg2rad(gamma)) 
	rx = utils.rot_x(np.deg2rad(alpha + 90)) 
	rz = utils.rot_z(np.deg2rad(beta)) 

	pose = utils.translate(y = 5).dot(rz).dot(rx)

def intersection_with_xz_plane(origin, vector):
	# matrix to drop y component
	to_2d = np.array([
		[1, 0, 0, 0],
		[0, 0, 1, 0],
		[0, 0, 0, 1]
	])

	# find intersections with xz plane
	#          0 = (origin + vector * a).y 
	#     =>   a = -origin.y / vector.y
	#     => int = origin + vector * a
	return to_2d.dot(
		origin + utils.scale(all=-origin[1] / vector[1])
					 .dot(vector)
	)

frame_size = np.array([1000, 500, 0])

def reframe(corners):
	"""
	corners: list of homogenous 2-vector
	reframe(corners) -> new_corners, size
	"""



	corners = corners + frame_size / 2

	return corners, frame_size

def m_from_axis_and_center(x, center):
	return np.array([
		[x[0], -x[1], center[0]],
		[x[1],  x[0], center[1]],
		[0,        0, 1]
	])

def rotations_of(axis):
	t = np.array([
		[0, -1, 0],
		[1, 0, 0],
		[0, 0, 1]
	])
	for i in range(4):
		yield axis
		axis = t.dot(axis)

def reframe(corners):
	"""
	corners: list of homogenous 2-vector
	reframe(corners) -> new_corners, size
	"""
	# find diagonal lenghts and units
	diag02 = corners[2] - corners[0]
	diag13 = corners[3] - corners[1]
	diag02_l = np.linalg.norm(diag02)
	diag13_l = np.linalg.norm(diag13)
	diag02 /= diag02_l
	diag13 /= diag13_l


	# find center and distances to center
	dists = np.zeros(4)

	m = np.array([diag02[:2], diag13[:2]]).T


	side = (corners[1] - corners[0])[:2]

	dists[0], md1 = np.linalg.solve(m, side)
	dists[1] = -md1
	dists[2] = diag02_l - dists[0]
	dists[3] = diag13_l - dists[1]

	center = corners[0] + dists[0] * diag02


	x_axis = diag02 - diag13
	x_axis /= np.linalg.norm(x_axis)

	x_axis = max(rotations_of(x_axis), key=np.array([1, 0, 0]).dot)

	from_cropped = m_from_axis_and_center(x_axis, center)
	to_cropped = np.linalg.inv(from_cropped)

	print m

	centered_corners = np.array([to_cropped.dot(corner) for corner in corners])

	diag02_cropped = to_cropped.dot(diag02)
	diag13_cropped = to_cropped.dot(diag13)

	cropped_corners = min(dists) * np.array([
		-diag02_cropped,
		-diag13_cropped,
		diag02_cropped,
		diag13_cropped
	])

	return (
		centered_corners - cropped_corners.min(axis=0),
		cropped_corners.max(axis=0) - cropped_corners.min(axis=0)
	)


with utils.get_camera() as cam, \
	utils.named_window(window_name), \
	utils.named_window("raw"):

	server.go()
	while pose is None:
		import time
		time.sleep(0)
	while True:
		ret, im = cam.read()

		w = im.shape[1]
		h = im.shape[0]
		#  a field of view about 194.5 inches wide at 195 inches
		# f = h / 194.5 * 195
		f = h / (460*2) * 730


		# image corners in pixel dimentions
		image_corners = [
			np.array([0, 0, 1]),
			np.array([w, 0, 1]),
			np.array([w, h, 1]),
			np.array([0, h, 1])
		]

		# converts pixels to directions
		to_directions = np.array([
			[1, 0, -w/2],
			[0, 1, -h/2],
			[0, 0,    f],
			[0, 0,    0]
		])

		# corner vectors pointing out of camera, in pixel units
		view_corners = np.array([to_directions.dot(i) for i in image_corners])


		# rotate camera corners and origin into world space
		world_corners = np.array([pose.dot(v) for v in view_corners])
		world_pos = pose.dot([0, 0, 0, 1])

		plane_corners = np.array([
			intersection_with_xz_plane(world_pos, world_corner)
			for world_corner in world_corners
		])

		# convert to pixels
		plane_corners = plane_corners * [50, 50, 1]

		try:
			plane_corners, size = reframe(plane_corners)
			assert (size[:2] < np.array([1000, 1000])).all()
			perspective_cv = cv2.getPerspectiveTransform(
				src=np.array([image_corner[:2] for image_corner in image_corners], np.float32),
				dst=np.array([plane_corner[:2] for plane_corner in plane_corners], np.float32)
			)

			dst = cv2.warpPerspective(im, perspective_cv, dsize=tuple(np.int32(size[:2])))

			cv2.imshow(window_name, dst)
			cv2.imshow('raw', im)

		except np.linalg.LinAlgError as e:
			print e
		except AssertionError as e:
			print e


		if cv2.waitKey(50) != -1:
			break

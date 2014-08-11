from __future__ import division
import numpy as np
import cv2

import server
import utils

np.set_printoptions(suppress=True)

# camera pose, in meter units
pose = utils.translate(y = 5)

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


with utils.get_camera() as cam, utils.named_window(window_name):
	server.go()
	while True:
		ret, im = cam.read()

		w = im.shape[1]
		h = im.shape[0]
		#  a field of view about 194.5 inches wide at 195 inches
		f = h / 194.5 * 195

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

		plane_corners, size = reframe(plane_corners)

		perspective_cv = cv2.getPerspectiveTransform(
			src=np.array([image_corner[:2] for image_corner in image_corners], np.float32),
			dst=np.array([plane_corner[:2] for plane_corner in plane_corners], np.float32)
		)

		dst = cv2.warpPerspective(im, perspective_cv, dsize=tuple(np.int32(size[:2])))

		cv2.imshow(window_name, dst)
		if cv2.waitKey(50) != -1:
			break

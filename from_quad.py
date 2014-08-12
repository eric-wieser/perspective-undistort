from __future__ import division
import numpy as np
import cv2

import utils

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


def apply_fixes_to(im, camera_pose):
	w = im.shape[1]
	h = im.shape[0]
	#  a field of view about 194.5 inches wide at 195 inches
	# f = h / 194.5 * 195
	f = h / (460*2) * 730


	# image corners in pixel dimentions
	unit_corners = np.array([
		[0, 0, 1],
		[1, 0, 1],
		[1, 1, 1],
		[0, 1, 1]
	])
	image_corners = unit_corners * [w, h, 1]

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
	world_corners = np.array([camera_pose.dot(v) for v in view_corners])
	world_pos = camera_pose.dot([0, 0, 0, 1])

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
		perspective_cv_inv = cv2.getPerspectiveTransform(
			dst=np.array([image_corner[:2] for image_corner in image_corners], np.float32),
			src=np.array([plane_corner[:2] for plane_corner in plane_corners], np.float32)
		)

		crop_corners = cv2.perspectiveTransform(
			np.array([(unit_corners * size)[:,:2]], np.float32),
			perspective_cv_inv
		)[0]

		dst = cv2.warpPerspective(im, perspective_cv, dsize=(tuple(np.int32(size[:2]))))
		cv2.polylines(
			im, np.int32([crop_corners * 8]), True, (0, 0, 255),
			shift=3, thickness=2, lineType=cv2.CV_AA
		)
		return dst

	except np.linalg.LinAlgError as e:
		print e
	except AssertionError as e:
		print e


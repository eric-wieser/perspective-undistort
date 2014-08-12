from __future__ import division
import numpy as np
import cv2
import time

import from_quad
import utils
import server

window_name = 'res'

@server.data_handler
def update(alpha, beta, gamma):
	# alpha(Y) - 0 side level - increase tilting away from you
	# beta (X) - 0 top level - increase tilting right
	# gamma(Z) - 0 north    - increase rotating counterclockwise

	ra = utils.rot_x(np.deg2rad(alpha)) 
	rb = utils.rot_z(np.deg2rad(beta))

	update.pose = (
		utils.translate(y = 5) # vertical offset
		.dot(ra)               # roll
		.dot(rb)               # yaw
		.dot(utils.rot_x(np.deg2rad(90))) # camera -> device
	)

update.pose = None

server.go()
print "Server running... ",

while update.pose is None:
	time.sleep(0)

print "OK"

with utils.get_camera() as cam, \
	utils.named_window(window_name), \
	utils.named_window("raw"):

	while cv2.waitKey(50) == -1:
		ret, im = cam.read()
		dst = from_quad.apply_fixes_to(im, update.pose)

		if dst is not None:
			cv2.imshow(window_name, dst)
			cv2.imshow('raw', im)



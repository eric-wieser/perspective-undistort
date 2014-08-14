import sys
import os
import shutil
import json
import utils

import cv2
from PIL import Image, ExifTags

import from_quad

def mkdir_f(d):
	if os.path.exists(d):
		shutil.rmtree(d)
	os.mkdir(d)
	return d


_, orig_dir = sys.argv
orig_dir = unicode(orig_dir)

# remove trailing slash
if orig_dir.endswith(u'/'):
	orig_dir = orig_dir[:-1]


new_dir = mkdir_f(orig_dir + u'.out')
frame_dir = mkdir_f(orig_dir + u'.frame')

for fname in os.listdir(orig_dir):
	new_fname   = os.path.join(new_dir, fname)
	frame_fname = os.path.join(frame_dir, fname)
	fname       = os.path.join(orig_dir, fname)

	img = Image.open(fname)
	exif_data = img._getexif()

	exif = {
		ExifTags.TAGS[k]: v
		for k, v in exif_data.items()
		if k in ExifTags.TAGS
	}

	geo_data = exif.get('UserComment')
	if not geo_data:
		print "Discarding {!r} - no data".format(fname)
		continue

	encoding, geo_data = geo_data[:8], geo_data[8:]
	geo_data = json.loads(geo_data)

	print geo_data

	# angle in centi degrees
	yaw = geo_data['Yaw']
	pitch = geo_data['Pitch']
	roll = geo_data['Roll']

	pose = (utils.translate(y=5)
		.dot(utils.rot_y(yaw))
		.dot(utils.rot_x(pitch))
		.dot(utils.rot_z(roll))
	)

	#damn you, opencv
	shutil.copy(fname, 'temp.jpg')
	im = cv2.imread('temp.jpg')
	dst = from_quad.apply_fixes_to(im, pose)
	if dst is None:
		print "Skipping {!r} -  could not transform".format(fname)
		continue

	cv2.imwrite('temp.jpg', dst)
	shutil.copy('temp.jpg', new_fname)

	cv2.imwrite('temp.jpg', im)
	shutil.copy('temp.jpg', frame_fname)
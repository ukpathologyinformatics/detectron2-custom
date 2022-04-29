# Download the 56 zip files in Images_png in batches
import urllib.request
import zipfile
import os
import pandas as pd
import cv2
import csv
import numpy as np
import hashlib
import shutil
from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.file import save_json
import json

from detectron2.structures import BoxMode

###########################
# Create root folder 
# Create 'labels' folder inside of root folder
# Create 'images' folder inside of root folder
##	Create 'test', 'val', 'train' folders inside images folder
# Copy DL_info to root folder (if not already there)
# Pip install pandas, cv2, zipfile, numpy
# Change paths below
# Run it, and pray
###########################

SRC_PATH = 'C:\\Users\\sarmstrong\\Desktop\\Images_png_56\\Images_png\\'
OUTPUT_PATH = 'C:\\projects\\detectron2\\datasets\\lesion\\data\\' # keep trailing slashes
LABEL_PATH = 'C:\\projects\\detectron2\\datasets\\lesion\\'
BORDER_BOX_COORDS_PATH = 'C:\\projects\\detectron2\\datasets\\lesion\\DL_info.csv'
coords_df = pd.read_csv(BORDER_BOX_COORDS_PATH)
coco_train = Coco()
coco_val = Coco()
coco_test = Coco()
coco_train.add_category(CocoCategory(id=0, name='lesion'))
coco_val.add_category(CocoCategory(id=0, name='lesion'))
coco_test.add_category(CocoCategory(id=0, name='lesion'))
# coco.add_category(CocoCategory(id=1, name='bone'))
# coco.add_category(CocoCategory(id=2, name='abdomen'))
# coco.add_category(CocoCategory(id=3, name='mediastinum'))
# coco.add_category(CocoCategory(id=4, name='liver'))
# coco.add_category(CocoCategory(id=5, name='lung'))
# coco.add_category(CocoCategory(id=6, name='kidney'))
# coco.add_category(CocoCategory(id=7, name='soft tissue'))
# coco.add_category(CocoCategory(id=8, name='pelvis'))

# label_map = {1: 'bone', 2: 'abdomen', 3: 'mediastinum', 4: 'liver', 5: 'lung', 6: 'kidney', 7: 'soft tissue', 8: 'pelvis'}



# def verify_md5(filename):
# 	with open(OUTPUT_PATH+"MD5_checksums.txt", 'r') as f:
# 		row = f.readline()
# 		if filename in row:
# 			row_s = row.split("  ")
# 			check = row_s[0]
# 			file = row_s[1]
# 			calc = hashlib.md5(open(OUTPUT_PATH+filename,'rb').read()).hexdigest()
# 			if calc == check:
# 				print(filename + ": Checksum OK (" + calc + ":" + check + ")")
# 				return True
# 			else:
# 				print(filename + ": Checksum NOT OK (" + calc + ":" + check + ")")
# 				return False
# 	print(filename + " not in MD5 file... skipping.")
# 	return True


# get all image files in dir
def getListOfFiles(path):
	listOfFile = os.listdir(path)
	allFiles = list()
	# Iterate over all the entries
	for entry in listOfFile:
		# Create full path
		fullPath = os.path.join(path, entry)
		# If entry is a directory then get the list of files in this directory 
		if os.path.isdir(fullPath):
			allFiles = allFiles + getListOfFiles(fullPath)
		else:
			allFiles.append(fullPath)
	return allFiles

# border box coord converter
# def convert(x1, y1, x2, y2, image_width, image_height): #may need to normalize
# 	dw = 1./image_width
# 	dh = 1./image_height
# 	x = (x1 + x2)/2.0
# 	y = (y1 + y2)/2.0
# 	w = x2 - x1
# 	h = y2 - y1
# 	x = x*dw
# 	w = w*dw
# 	y = y*dh
# 	h = h*dh
# 	return x,y,w,h

def read_DL_info():
	"""read spacings and image indices in DeepLesion"""
	spacings = []
	idxs = []
	with open(BORDER_BOX_COORDS_PATH, 'r') as csvfile:
		reader = csv.reader(csvfile)
		rownum = 0
		for row in reader:
			if rownum == 0:
				header = row
				rownum += 1
			else:
				idxs.append([int(d) for d in row[1:4]])
				spacings.append([float(d) for d in row[12].split(',')])

	idxs = np.array(idxs)
	spacings = np.array(spacings)
	return idxs, spacings

def fix_save_images(src_path, out_path):
	images = getListOfFiles(src_path)
	idxs, spacings = read_DL_info()
	# train_images = []
	# val_images = []
	# test_images = []

	print("Moving images and finding bounding box coords...")
	for im in images:
		## move
		image_name_split = im.split('\\')
		image_name = image_name_split[-2] + "_" + image_name_split[-1]
		#curr_image = cv2.imread(im, -1)
		#converted_im = convert_to_png([np.array((curr_image.astype(np.int32) - 32768).astype(np.int16), np.int16)])
		#imageio.imwrite(DST_IMAGE_PATH + image_name, converted_im) # DONT CHANGE THIS IS UINT8
		

		## find coords
		name = coords_df.loc[coords_df['File_name'].str.contains(image_name, case=False)]
		if name.index.size > 0:
			coords = coords_df["Bounding_boxes"][name.index[0]].split(',')
			label = coords_df["Coarse_lesion_type"][name.index[0]]
			train_val_test = coords_df["Train_Val_Test"][name.index[0]]

			if train_val_test == 1:     # -1 labels are training   
				#train_images.append(OUTPUT_PATH + "images\\train\\" + image_name)
				shutil.copy(im, out_path + "train\\" + image_name)
				coco_image = CocoImage(file_name=out_path + "train\\" + image_name, height=512, width=512)
				coco_image.add_annotation(
					CocoAnnotation(
						bbox=[float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])],
						#bbox_mode=BoxMode.XYXY_ABS,
						category_id=0,
						category_name='lesion'
					)
				)
				coco_train.add_image(coco_image)
				# with open(LABEL_PATH + image_name[:-3] + "txt", 'a') as f:
				#     x,y,w,h = convert(float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3]), 512, 512)
				#     f.write(str(label-1) + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + "\n")
			
			#Validation
			elif train_val_test == 2:   
				#val_images.append(OUTPUT_PATH + "images\\val\\" + image_name)
				shutil.copy(im, out_path + "val\\" + image_name)
				coco_image = CocoImage(file_name=out_path + "val\\" + image_name, height=512, width=512)
				coco_image.add_annotation(
					CocoAnnotation(
						bbox=[float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])],
						#bbox_mode=BoxMode.XYXY_ABS,
						category_id=0,
						category_name='lesion'
					)
				)
				coco_val.add_image(coco_image)
				# with open(LABEL_PATH + image_name[:-3] + "txt", 'a') as f:
				# 	x,y,w,h = convert(float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3]), 512, 512)
				# 	f.write(str(label-1) + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + "\n")

			#Test
			elif train_val_test == 3:
				#test_images.append(OUTPUT_PATH + "images\\test\\" + image_name)
				shutil.copy(im, out_path + "test\\" + image_name)
				coco_image = CocoImage(file_name=out_path + "test\\" + image_name, height=512, width=512)
				coco_image.add_annotation(
					CocoAnnotation(
						bbox=[float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])],
						#bbox_mode=BoxMode.XYXY_ABS,
						category_id=0,
						category_name='lesion'
					)
				)
				coco_test.add_image(coco_image)
				# with open(LABEL_PATH + image_name[:-3] + "txt", 'a') as f:
				# 	x,y,w,h = convert(float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3]), 512, 512)
				# 	f.write(str(label-1) + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + "\n")


fix_save_images(SRC_PATH, OUTPUT_PATH)
# coco_json = coco.json

from sahi.utils.file import save_json

json_mod_train = coco_train.json
for ann in json_mod_train['annotations']:
	ann['bbox'] = BoxMode.convert(ann['bbox'], from_mode=BoxMode.XYXY_ABS, to_mode=BoxMode.XYWH_ABS) 

json_mod_val = coco_val.json
for ann in json_mod_val['annotations']:
	ann['bbox'] = BoxMode.convert(ann['bbox'], from_mode=BoxMode.XYXY_ABS, to_mode=BoxMode.XYWH_ABS) 

json_mod_test = coco_test.json
for ann in json_mod_test['annotations']:
	ann['bbox'] = BoxMode.convert(ann['bbox'], from_mode=BoxMode.XYXY_ABS, to_mode=BoxMode.XYWH_ABS) 

save_json(data=json_mod_train, save_path=LABEL_PATH+"train_labels.json")
save_json(data=json_mod_val, save_path=LABEL_PATH+"val_labels.json")
save_json(data=json_mod_test, save_path=LABEL_PATH+"test_labels.json")



# md5_link = 'https://nihcc.box.com/shared/static/q0f8gy79q2spw96hs6o4jjjfsrg17t55.txt'
# urllib.request.urlretrieve(md5_link, OUTPUT_PATH + "MD5_checksums.txt")  # download the MD5 checksum file
# for idx, link in enumerate(links):
# 	fn = OUTPUT_PATH+'Images_png_%02d.zip' % (idx+1)
# 	print('Downloading', fn, '...')
# 	urllib.request.urlretrieve(link, fn)  # download the zip file
# 	while not verify_md5('Images_png_%02d.zip' % (idx+1)):
# 		os.remove(fn)
# 		print('Re-Downloading', fn, '...')
# 		urllib.request.urlretrieve(link, fn)  # download the zip file
		
# 	with zipfile.ZipFile(fn, 'r') as zip_ref:
# 		print('Extracting', fn, '...')
# 		zip_ref.extractall(OUTPUT_PATH)

# 	os.remove(fn)
# 	print('Re-saving images correctly...')
# 	fix_save_images(OUTPUT_PATH + 'Images_png\\')

# print("Done.")




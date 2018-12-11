from keras_resnet import *
import cv2
import matplotlib as plt
import numpy as np
from keras.layers import Input, Dense
from keras.models import *
import sys
import os, errno
import json


class Sketchy(object):

	def __init__(self, train_data, test_data, im_path, op_dir_path, batch_size):

		self.im_size = 1111
		self.ip_size = ( self.im_size, self.im_size)
		self.batch_size = int(batch_size)
		# self.op_size = 1
		self.class_id = 0
		self.dict = {}
		self.files = [[],[]]
		self.test = [[],[]]
		self.train_files = train_data
		self.test_files = test_data
		self.im_path = im_path
		self.op_dir_path = op_dir_path
		
	
	def load_train_data(self):

		with open(self.train_files) as f:
			content = f.readlines()
			content = [x.strip() for x in content]

			for x in content:
				path = x.split("/")
				
				if not (path[0] in self.dict.keys()):
					self.dict[path[0]] = self.class_id
					self.class_id += 1

				self.files[0].append( self.im_path+x )
				self.files[1].append( path[0] )
				# print("Loading train files",self.im_path+x, path[0])

		self.op_size = len(self.dict.keys())
		# self.save_classes()
		print("num classes", self.op_size) 

	def train_model(self):

		self.load_train_data()
		print("Loaded training data file paths")

		self.model = ResNet34( self.ip_size, self.op_size )
		self.model.compile( loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

		print("Model initialized")
		
		im = np.zeros( (self.batch_size , self.im_size, self.im_size, 3) )
		i = 0
		# print("image", len( self.files[0] ))
		while ( i < len( self.files[0] ) ):
			print("Iteration", i)
			op = np.zeros( (self.batch_size, self.op_size ) )

			for b in range( self.batch_size):

				# print("len",len( self.files[0] ), "ind", i+b)
				if (i+b) >= len( self.files[0] ):
					break
				# print("after break", i+b,"reading file", self.files[0][ i + b ])
				
				print("reading", self.files[0][ i + b ])
				
				im [b] = cv2.imread(self.files[0][ i + b ])
				# im[b] = cv2.resize(big_image, (self.im_size, self.im_size), interpolation = cv2.INTER_CUBIC)
				op [b] [self.dict [ self.files[1][ i + b ] ] ] = 1.0
				print(self.files[0][ i + b ], self.files[1][ i + b ] ,self.dict[self.files[1][ i + b ]])
				# cv2.imshow( "shit",im[ b ] )
				# cv2.waitKey()
				

			i += self.batch_size

			self.model.fit(x = im, y = op, verbose = 1, validation_split = 0.1, batch_size = self.batch_size, epochs = 5)

			self.save_models()
		
	def save_models(self):
		try:
		    os.makedirs(self.op_dir_path)
		except OSError as e:
		    if e.errno != errno.EEXIST:
		        raise
		model_json = self.model.to_json()
		with open(self.op_dir_path+"/model.json", "w") as json_file:
			json_file.write(model_json)
		self.model.save_weights(self.op_dir_path+"/model.h5")

	def load_models(self):
		json_file = open(self.op_dir_path+'/model.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		self.model = model_from_json(loaded_model_json)

		# load weights into new model
		self.model.load_weights(self.op_dir_path+"/model.h5")
		print("Loaded model from disk")
		self.model.compile( loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

	def load_test_data(self):

		with open(self.test_files) as f:
			content = f.readlines()
			content = [x.strip() for x in content]

			for x in content:
				path = x.split("/")
				
				self.test[0].append( self.im_path+x )
				self.test[1].append( path[0] )

	def test_model(self):

		self.op_size = len(self.dict.keys())
		self.load_models()
		self.load_test_data()
		self.inv_dict = { v: k for k, v in self.dict.iteritems() }

		im = np.zeros( ( 1, self.im_size, self.im_size, 3) )
		i = 0
		while ( i < len( self.test[0] )):

			op = np.zeros((1, self.op_size))
			big_image = cv2.imread(self.test[0][ i ])
			im[0] = cv2.resize(big_image, (self.im_size, self.im_size))
			op [0][self.dict [ self.test[1][ i ] ] ] = 1.0
							
			p = self.model.predict(x = im)

			max_ind = 0
			max_val = 0

			for ind in range(p.shape[1]):
				# print(ind)
				if p[0][ind] > max_val:
					max_val = p[0][ind]
					max_ind = ind

			print(self.test[0][ i ], " Expected", self.dict[ self.test[1][ i ]], " Predicted ", max_ind)
			i += 1

	def save_classes(self):
		with open(self.op_dir_path+'/classes.json', 'w') as fp:
			json.dump(self.dict, fp)

	def load_classes(self):
		with open(self.op_dir_path+'/classes.json') as f:
			self.dict = json.load(f)


print("Usage <script> <training files> <testing files> <images path> <op_path> <batch_size>")
obj = Sketchy(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])

if sys.argv[6] == "normal":
	obj.train_model()
	obj.save_classes()
	obj.test_model()

if sys.argv[6] == "train":
	obj.train_model()
	# obj.save_classes()

if sys.argv[6] == "test":
	obj.load_classes()
	obj.test_model()
	



from resnet import *
import cv2
import matplotlib as plt
import numpy as np
from keras.layers import Input, Dense
import sys
import os, errno


class Sketchy(object):

	def __init__(self, train_data, test_data, im_path, op_dir_path, batch_size):

		self.im_size = 128
		self.ip_size = (3, self.im_size, self.im_size)
		self.batch_size = int(batch_size)

		self.class_id = 0
		self.dict = {}
		self.files = [[],[]]
		self.test_files = []
		self.train_files = train_data
		self.test_files = test_data
		self.im_path = im_path
		self.op_dir_path = op_dir_path
		self.load_train_data()
		
	
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
		print("num classes", self.op_size) 

	def train_model(self):

		self.model = ResnetBuilder.build_resnet_50( self.ip_size, self.op_size )
		self.model.compile( loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
		
		im = np.zeros( (self.batch_size , self.im_size, self.im_size, 3) )
		i = 0
		# print("image", len( self.files[0] ))
		while ( i < len( self.files[0] ) ):

			op = np.zeros( (self.batch_size, self.op_size ) )

			for b in range( self.batch_size):

				# print("len",len( self.files[0] ), "ind", i+b)
				if (i+b) >= len( self.files[0] ):
					break
				# print("after break", i+b,"reading file", self.files[0][ i + b ])
				

				big_image = cv2.imread(self.files[0][ i + b ])
				im[b] = cv2.resize(big_image, (self.im_size, self.im_size))
				op [b] [self.dict [ self.files[1][ i + b ] ] ] = 1.0
				# print(self.files[0][ i + b ], self.files[1][ i + b ])

			i += self.batch_size

			self.model.fit(x = im, y = op, verbose = 1, validation_split = 0.2, batch_size = self.batch_size, epochs = 1)

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
				
				if not (path[0] in self.dict.keys()):
					self.dict[path[0]] = self.class_id
					self.class_id += 1

				self.test_files[0].append( self.im_path+x )
				self.test_files[1].append( path[0] )

	def test_model(self):

		self.load_models()
		self.load_test_data()
		self.inv_dict = { v: k for k, v in self.dict.iteritems() }

		im = np.zeros( (self.batch_size , self.im_size, self.im_size, 3) )
		i = 0
		while ( i < len( self.files[0] )):

			op = np.zeros((1, self.op_size))
			big_image = cv2.imread(self.files[0][ i ])
			im[b] = cv2.resize(big_image, (self.im_size, self.im_size))
			op [b] [self.dict [ self.files[1][ i ] ] ] = 1.0
				
			i += 1

			p = self.model.predict(x = im)

			max_ind = 0
			max_val = 0

			for ind in range(p.shape[1]):
				# print(ind)
				if p[0][ind] > max_val:
					max_val = p[0][ind]
					max_ind = ind

			print(self.files[0][ i ], " Expected", self.dict[ self.files[1][ i ]], " Predicted ", self.inv_dict[max_ind])


print("Usage <script> <training files> <testing files> <images path> <op_path> <batch_size>")
obj = Sketchy(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
obj.train_model()
obj.test_model()


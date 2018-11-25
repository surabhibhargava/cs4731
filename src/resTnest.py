from resnet import *
import cv2
import matplotlib as plt
import numpy as np
from keras.layers import Input, Dense


class Sketchy(object):

	def __init__(self, train_data, test_data):

		self.im_size = 128
		self.ip_size = (3, self.im_size, self.im_size)
		self.op_size = (251)
		self.batch_size = 8

		self.model = ResnetBuilder.build_resnet_50( self.ip_size, self.op_size )
		self.model.compile( loss = 'binary_crossentropy', optimizer = 'rmsprop')

		self.class_id = 0
		self.dict = {}
		self.files = [[],[]]
		self.test_files = []
		self.load_train_data(train_data)
		
	
	def load_train_data(self, train_data):
		with open("../splits/train.txt") as f:
			content = f.readlines()
			content = [x.strip() for x in content]

			for x in content:
				path = x.split("/")
				
				if not (path[0] in self.dict.keys()):
					self.dict[path[0]] = self.class_id
					self.class_id += 1

				self.files[0].append( "../../png/"+x )
				self.files[1].append( path[0] )

		# print(self.dict)

	def train_model(self):
		
		im = np.zeros( (self.batch_size , self.im_size, self.im_size, 3) )
		i = 0
		while ( i < len( self.files[0] ) ):

			op = np.zeros((self.batch_size, 251))
			for b in range( self.batch_size):

				big_image = cv2.imread(self.files[0][ i + b ])
				im[b] = cv2.resize(big_image, (self.im_size, self.im_size))
				op [b] [self.dict [ self.files[1][ i + b ] ] ] = 1.0

			i += self.batch_size

			self.model.fit(x = im, y = op, batch_size = self.batch_size, epochs = 5)
			# print(self.model.evaluate(batch_size = self.batch_size, steps = 10))

		model_json = self.model.to_json()
		with open("./model.json", "w") as json_file:
			json_file.write(model_json)
		self.model.save_weights("model.h5")


	def test_model(self):
		
		with open("../splits/test.txt") as f:
			content = f.readlines()
			content = [x.strip() for x in content]

			for x in content:
				path = x.split("/")
				
				self.files[0].append( "../../png/"+x )
				self.files[1].append( path[0] )
			



obj = Sketchy("../splits/train.txt", "../splits/test.txt")
obj.train_model()


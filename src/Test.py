from resnet import *
import cv2
import matplotlib as plt
import numpy as np
from keras.models import *
from keras.layers import Input, Dense
import json


class Sketchy(object):

	dict = {'snowman': 48, 'lightbulb': 198, 'fire hydrant': 249, 'eyeglasses': 183, 'camel': 95, 'laptop': 214, 'teddy-bear': 141, 'scissors': 189, 'flying saucer': 197, 'arm': 212, 'crocodile': 104, 'chandelier': 24, 'radio': 45, 'shoe': 90, 'keyboard': 235, 'chair': 106, 'tablelamp': 176, 'flashlight': 182, 'tire': 210, 'cup': 44, 'tv': 122, 'door handle': 215, 'backpack': 209, 'scorpion': 68, 'mug': 161, 'bench': 21, 'bush': 194, 'trousers': 162, 'piano': 22, 'hat': 186, 'pumpkin': 43, 'crane (machine)': 150, 'sponge bob': 76, 'cannon': 201, 'speed-boat': 174, 'barn': 147, 'screwdriver': 34, 'pipe (for smoking)': 113, 'vase': 124, 'spoon': 184, 'giraffe': 73, 'fan': 195, 'snake': 245, 'lighter': 217, 'potted plant': 7, 'foot': 62, 'santa claus': 148, 'bread': 185, 'car (sedan)': 20, 'stapler': 102, 'parrot': 156, 'ashtray': 38, 'sea turtle': 221, 'head-phones': 196, 'human-skeleton': 13, 'wrist-watch': 128, 'microscope': 229, 'saxophone': 134, 'mosquito': 86, 'pickup truck': 137, 'tent': 132, 'wheel': 107, 'bulldozer': 238, 'bicycle': 211, 'house': 175, 'fish': 224, 'church': 93, 'crown': 10, 'spider': 177, 'ice-cream-cone': 145, 'hot air balloon': 36, 'lion': 237, 'satellite dish': 199, 'lobster': 87, 'zebra': 80, 'dragon': 169, 'trombone': 47, 'candle': 71, 'cactus': 26, 'hammer': 31, 'panda': 84, 'hourglass': 52, 'loudspeaker': 119, 'leaf': 11, 'donut': 53, 'sun': 25, 'alarm clock': 67, 'pigeon': 63, 'revolver': 85, 'moon': 91, 'teacup': 69, 'suv': 143, 'pen': 135, 'rooster': 193, 'skyscraper': 155, 'ipod': 168, 'knife': 246, 'nose': 115, 'umbrella': 173, 'bear (animal)': 139, 'kangaroo': 158, 'mailbox': 78, 'guitar': 105, 'wine-bottle': 130, 'key': 118, 'wheelbarrow': 192, 'van': 100, 'comb': 127, 'person walking': 32, 'monkey': 181, 'toothbrush': 222, 'cow': 231, 'baseball bat': 99, 'skateboard': 159, 'rollerblades': 54, 'pretzel': 239, 'blimp': 30, 'paper clip': 72, 'octopus': 204, 'person sitting': 165, 'owl': 97, 'shovel': 51, 'apple': 1, 'flower with stem': 66, 'toilet': 138, 'armchair': 65, 'standing bird': 200, 'socks': 188, 'swan': 203, 'snowboard': 129, 'duck': 88, 'syringe': 170, 'table': 153, 'walkie talkie': 243, 'cloud': 160, 'sheep': 82, 'computer monitor': 216, 'horse': 223, 'eye': 39, 'motorbike': 120, 'bottle opener': 59, 'traffic light': 228, 'frog': 213, 'grapes': 154, 'submarine': 125, 'parachute': 149, 'camera': 81, 'windmill': 108, 'crab': 27, 'streetlight': 101, 'helicopter': 110, 'feather': 35, 'megaphone': 126, 'banana': 202, 'mermaid': 190, 'fork': 15, 'tennis-racket': 232, 'head': 136, 'diamond': 2, 'door': 244, 'squirrel': 226, 'mushroom': 55, 'mouse (animal)': 167, 'bus': 111, 'envelope': 140, 'floor lamp': 83, 'cabinet': 89, 'tiger': 98, 'train': 172, 'brain': 180, 'rabbit': 187, 'rifle': 56, 'bookshelf': 109, 'ear': 133, 'teapot': 234, 'present': 0, 'penguin': 79, 'space shuttle': 103, 'skull': 4, 'tree': 8, 'bed': 40, 'bee': 58, 'beer-mug': 94, 'strawberry': 18, 'pizza': 220, 'suitcase': 171, 'cake': 236, 'boomerang': 12, 'seagull': 247, 'bathtub': 191, 'grenade': 74, 'tractor': 29, 'race car': 96, 'bell': 178, 'bridge': 77, 'shark': 242, 'computer-mouse': 19, 'hamburger': 207, 'angel': 46, 'canoe': 33, 'wineglass': 233, 'calculator': 50, 'dolphin': 64, 'purse': 219, 'telephone': 131, 'pig': 28, 'ant': 6, 'carrot': 241, 'violin': 41, 'airplane': 157, 'tooth': 42, 'trumpet': 92, 'ship': 208, 'flying bird': 70, 'helmet': 3, 'microphone': 49, 'sailboat': 117, 'power outlet': 123, 'bowl': 146, 'couch': 61, 'palm tree': 151, 'book': 163, 'sword': 14, 'elephant': 248, 'harp': 75, 'hot-dog': 112, 'butterfly': 9, 'satellite': 166, 'snail': 37, 'rainbow': 164, 'ladder': 114, 'pineapple': 225, 'pear': 16, 'hand': 17, 'binoculars': 179, 'cigarette': 230, 'mouth': 116, 'cell phone': 240, 'castle': 206, 'tomato': 227, 't-shirt': 142, 'cat': 5, 'frying-pan': 152, 'dog': 57, 'face': 60, 'truck': 144, 'parking meter': 121, 'axe': 205, 'basket': 23, 'hedgehog': 218}
	def __init__(self, train_data, test_data):

		self.im_size = 128
		self.ip_size = (3, self.im_size, self.im_size)
		self.op_size = (251)
		self.batch_size = 1

		self.class_id = 0
		self.inv_dict = { v: k for k, v in self.dict.iteritems() }
		self.files = [[],[]]
		self.load_model()
		self.load_test_data(test_data)
		
	def load_model(self):
		json_file = open('./models/model.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		self.model = model_from_json(loaded_model_json)

		# load weights into new model
		self.model.load_weights("./models/model.h5")
		print("Loaded model from disk")
		self.model.compile( loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])


		# json_file = open('./models/labels.txt', 'r')
		# loaded_model_json = json_file.read()
		# print(loaded_model_json)

		# self.dict = json.loads(loaded_model_json)[0]
		# print(self.dict)
		# print("loaded labels")
	
	def load_test_data(self, test_data):
		with open("../splits/test.txt") as f:
			content = f.readlines()
			content = [x.strip() for x in content]

			for x in content:
				path = x.split("/")
				
				if not (path[0] in self.dict.keys()):
					self.dict[path[0]] = self.class_id
					self.class_id += 1

				self.files[0].append( "../../png/"+x )
				self.files[1].append( path[0] )

	def test_model(self):

		im = np.zeros( (self.batch_size , self.im_size, self.im_size, 3) )
		i = 0
		while ( i < len( self.files[0] ) ):

			op = np.zeros((self.batch_size, 251))
			for b in range( self.batch_size):

				big_image = cv2.imread(self.files[0][ i + b ])
				im[b] = cv2.resize(big_image, (self.im_size, self.im_size))
				op [b] [self.dict [ self.files[1][ i + b ] ] ] = 1.0
				# print("something ",self.files[0][ i + b ], self.dict [ self.files[1][ i + b ] ])

			i += self.batch_size

			p = self.model.predict(x = im)
			# ind = np.where(p == p.max())
			max_ind = 0
			max_val = 0

			for ind in range(p.shape[1]):

				if p[0][ind] > max_val:
					max_val = p[0][ind]
					max_ind = ind

			# print(self.inv_dict)
			print("max ind ", max_ind, " max val ", max_val)
			print("testing",self.files[0][ i + b ], self.inv_dict[max_ind] )
			# print(self.model.evaluate(x = im, y = op))


obj = Sketchy("../splits/train.txt", "../splits/test.txt")
obj.test_model()


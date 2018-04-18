import numpy as np
from keras.layers import Input, Dense, GaussianNoise
from keras.models import Model

"""
Import as -> 
from Stacked_AE import Stacked_AE
deep_AE = Stacked_AE(args and kwargs)
Stacked_AE.fit(training data, bath size, epochs per layer)

After training the last encoder will be in 
	Stacked_AE.encoder
And the decoder in
	Stacked_AE.decoder
All other encoders will be in the lists
	Stacked_AE.encoder_models, as compiled models
	Stacked_AE.encoders as tensors.

Tweak with the noise_stdev parameter if the mean squared error is too high.
Add BatchNormalization or Convolution layers if you need to. 
I've commented where it's safe to do so.

"""

class Stacked_AE(object):

	def __init__(self, 
					layer_sizes=[26, 16, 8, 4], 
					input_shape=32, 
					noise_stdev=0.1,
					batch_size=1024,
					optimizer='rmsprop', activation='sigmoid'):

		self.layer_sizes = [input_shape] + layer_sizes
		self.encoders = []; self.decoders = []; self.models = []
		self.encoder_models = []; self.decoder_memory = {}
		self._deflayers(input_shape, noise_stdev,
						opt=optimizer, act=activation)


	def _deflayers(self, input_shape, 
					noise_stdev, opt, act):
		"""
		Defines and compiles models, 
		this function is called when the class is instantiated.
		The part of this function before the loop defines the Input and adds noise.
			If you need to handle the input differently add it here.
		In the loop an encoder is a
		
		"""

		input_layer = Input(shape=(input_shape,))
		# If you're image processing, add a convolution layer here.
		self.encoders.append(
					GaussianNoise(noise_stdev)(input_layer)
			)
		for i, ls in enumerate(self.layer_sizes[1:], 1):
			
			added_layer = Dense(ls, activation=act, kernel_initializer='random_uniform',
							bias_initializer='zeros', name='encoder_N{}'.format(i))(self.encoders[-1])
			# Add batch normalization/custom layers/activations for the encoder here
			self.encoders.append(added_layer)
			new_decoder = self._define_decoder(added_layer, i, act=act)

			self.decoders.append(
						Dense(input_shape, activation='linear', kernel_initializer='random_uniform',
							bias_initializer='zeros', name='decoder_out')(new_decoder)
							)
			model = Model(inputs=input_layer, outputs=self.decoders[-1])
			# This is where you can change the optimizer/error/regularzer
			model.compile(optimizer=opt,
							loss='mean_squared_error',
							metrics=['accuracy'])
			self.models.append(model)
			self.encoder_models.append(Model(inputs=input_layer, outputs=self.encoders[-1]))
		self.encoder = Model(inputs=input_layer, outputs=self.encoders[-1])
		self.encoder.compile(optimizer=opt,
							loss='mean_squared_error',
							metrics=['accuracy'])

		self.decoder = self.models[-1]

		self.input_layer = input_layer
	
	def _define_decoder(self, decoder, i, act):
		""" 
		Takes in the first decoding layer
		and adds layers to it until the last layer.
		for each layer, save it, then:
			compare the current layer, if it is there already -> use it
			if not, save it in a dictionary
		"""
		for k in range(1, i):
			dcd = self.decoder_memory.get((self.layer_sizes[i-k], i), False)
			
			if dcd is False:
				dcd = Dense(self.layer_sizes[i-k], activation=act, kernel_initializer='random_uniform',
							bias_initializer='zeros', name='decoder_N{}'.format(i-k))
				# Add batch normalization/custom layers/activations for the decoder here
				self.decoder_memory.update({(self.layer_sizes[i-k], i): dcd})
			
			decoder = dcd(decoder)

		return decoder

	def fit(self, X, batch_size=1024, epochs=[60, 60, 60, 100], v=0):
		for model, epoch in zip(self.models, epochs):
			model.fit(X, X, epochs=epoch, shuffle=True,
						verbose=v, batch_size=batch_size)

	def compress(self, X):
		return self.encoder.predict(X)

	def evaluate(self, X):
		return self.encoder.evaluate(X, X)

if __name__ == '__main__':

		
	deep_AE = Stacked_AE()

	for model in deep_AE.models:
		print(model.summary())

	print([type(model) for model in deep_AE.models])

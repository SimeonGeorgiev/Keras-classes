from keras import backend as K
from keras.regularizers import Regularizer
class Maximise_discrepancy(Regularizer):
    """
	Regularizer that maximises the number of values 
	higher than 1/n for a softmax layer of size n
    # Arguments
        R: Float; L1 regularization factor.
        C: Float; scales the mean, a higher value relaxes the constraint.
	Used when a softmax layer is used for something other than classification,
		such as learning a sparse encoding of the data.
    """
    def __init__(self, R=2e-6, C=1):
        self.R = K.cast_to_floatx(R)
        self.C = K.cast_to_floatx(C)

    def __call__(self, A):  
        """
        A is an activity vector, coming from a softmax activation.
        R is the regularization parameter and 
                            must be positive and less than than 1.
        The following function has a high value when values are higher than A.
        The derivative of this function is equal to
            the jacobian of abs(A - A_bar), where A_bar= mean(A)
            or to the following function:
        def jacobian(A):
            derivatives = []
            A_bar = np.mean(A)
            for val in A:
                if val - A_bar > 0:
                    derivatives.append(1)
                elif val - A_bar < 0:
                    derivatives.append(-1)
                elif val - A_ bar == 0:
                    raise: ValueError("derivative undefined for 0")
            return var
        """
        A_bar = K.mean(A) * self.C
        regularization = K.sum(self.R * K.abs(A-A_bar))
        return regularization

    def get_config(self):
        return {'R': float(self.R),
                'C': float(self.C)}

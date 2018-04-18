from keras import backend as K
from keras.constraints import Constraint

class Min_Max_Constraint(Constraint):
    """
        Constrains the inputs to a certain range.
	Calls keras.backend.clip on the bias or weight matrix.
	For example, non-negative and higher than a value.
        Useful in defining constrained optimization problems.
	Example use:
		constraint = Min_Max_Constraint(min_value=0, max_value=9)
		layer = Dense(*other arguments*, bias_constraint=constraint)(previous_layer)
    """
    def __init__(self, min_value=0, max_value=9):
        self.max_value = max_value
	self.min_value = min_value

    def __call__(self, w):
        """Returns the new weight matrix, after clipping."""
        w = K.clip(w, self.min_value, self.max_value)
        return w

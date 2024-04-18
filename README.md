# Hopfield

This Github contains multiple hopfield network implementations

They are all contained in the hopfield.py file.

The imageTest.py file is run by entering 'python imageTest.py', the user will be prompted to run either the standard hopfield or DAM.

The modernImages.py, and colourImages.py both run the continuous hopfield network, run using 'python filename'.


## Documentation:
### Hopfield Class:

Implements a basic Hopfield network.

**__init__(inputs):**

Initializes a Hopfield network with given input patterns.

    Parameters:
        inputs (list of arrays): List of input patterns (bipolar).

    Returns:
        None

**predict(input, iterations, theta=0.0):**

Predicts the output pattern given an input pattern.

    Parameters:
        input (array): Input pattern.
        iterations (int): Maximum number of iterations.
        theta (float, optional): Threshold for energy.

    Returns:
        predicted (list of arrays): Predicted patterns at each iteration.

### DAMDiscreteHopfield Class:

Implements a Dense Associative Memory (DAM) Hopfield network for discrete patterns.

**__init__(inputs):**

Initializes a DAM Hopfield network with given input patterns.

    Parameters:
        inputs (list of arrays): List of input patterns (bipolar).

    Returns:
        None

**predict(input, iterations):**

Predicts the output pattern given an input pattern.

    Parameters:
        input (array): Input pattern.
        iterations (int, optional): Maximum number of iterations.

    Returns:
        predicted (list of arrays): Predicted patterns at each iteration.

### ContinuousHopfield Class:

Implements a Hopfield network for continuous patterns.

**__init__(inputs):**

Initializes a Continuous Hopfield network with given input patterns.

    Parameters:
        inputs (list of arrays): List of input patterns (continuous).

    Returns:
        None

**predict(input, iterations, beta=8):**

Predicts the output pattern given an input pattern.

    Parameters:
        input (array): Input pattern.
        iterations (int, optional): Maximum number of iterations.
        beta (float, optional): Temperature parameter for softmax function.

    Returns:
        predicted (list of arrays): Predicted patterns at each iteration.

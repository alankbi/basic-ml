# BasicML

BasicML is a Python package containing implementations of basic machine learning algorithms. Below is a list of all the ML models that come with this package: 

**Regression:** 
- LinearRegression

**Classification:** 
- DecisionTree
- KNearestNeighbors
- LogisticRegression
- NeuralNetwork

**Clustering**
- KMeans

## Installation: 

Install using pip:

`pip install basic-ml`


## Usage

```python
import numpy as np  
from ml import LinearRegression  

# Replace with your own data
trn_X, trn_y, tst_X, tst_y = np.ones(1), np.ones(1), np.ones(1), np.ones(1)  

lr = LinearRegression()  
lr.fit(trn_X, trn_y)  

predictions = lr.predict(tst_X)  
print('Predicted: {}\nActual: {}'.format(predictions, tst_y))
```

To see example code, open/run any of the 6 main Python files in the `ml` folder.

## Contact

Reach out to me at [alan.bi326@gmail.com](mailto:alan.bi326@gmail.com) for questions and feedback!


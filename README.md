# BasicML
A lightweight package for basic machine learning needs

BasicML is a Python package containing implementations of basic machine learning algorithms. 

Below is a list of all the ML models that come with this package: 

**Regression:** 
- LinearRegression

**Classification:** 
- DecisionTree
- KNearestNeighbors
- LogisticRegression
- NeuralNetwork

**Clustering**
- KMeans

### Installation: 

Install using pip:  
`pip install basic-ml`


### Usage

```python
from ml import LinearRegression

trn_X, trn_y, tst_X, tst_y = ... # load your data

lr = LinearRegression()
lr.fit(trn_X, trn_y)

predictions = lr.predict(tst_X)
print('Predicted: {}\nActual: {}'.format(predictions, tst_y))

```

To see example code, open/run any of the 6 main Python files in the `ml` folder.

### Contact

Reach out to me at [alan.bi326@gmail.com](mailto:alan.bi326@gmail.com) for questions and feedback!


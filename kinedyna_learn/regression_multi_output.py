# example of evaluating chained multioutput regression with an SVM model https://machinelearningmastery.com/multi-output-regression-models-with-python/
from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVR

# define dataset
X, y = make_regression(n_samples=1000, n_features=2, n_informative=5, n_targets=2, random_state=1, noise=0.5)
print("==>> X.shape: \n", X.shape)
print("==>> y.shape: \n", y.shape)

# define base model
model = LinearSVR()

# define the direct multioutput wrapper model
wrapper = MultiOutputRegressor(model)

# fit the model on the whole dataset
wrapper.fit(X, y)

# make a single prediction
# row = [0.21947749, 0.32948997, 0.81560036, 0.440956, -0.0606303, -0.29257894, -0.2820059, -0.00290545, 0.96402263, 0.04992249]
row = [0.21947749, 0.32948997]

# predict
yhat = wrapper.predict([row])
# summarize the prediction
print('Predicted: %s' % yhat[0])


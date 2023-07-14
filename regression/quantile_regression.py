import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# generate a random dataset with two variables using normal distribution

df = pd.DataFrame(np.random.normal(0, 1, (100, 2)))
df.columns = ['x', 'y']

x = df['x']
y = df['y']

# fit the model
fit = np.polyfit(x, y, deg=1)
_x = np.linspace(x.min(), x.max(), num=len(y))

model = smf.quantreg('y ~ x', df)
quantiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.95]
fits = [model.fit(q=q) for q in quantiles]

# the quantile lines
_y_005 = fits[0].params['x'] * _x + fits[0].params['Intercept']
_y_095 = fits[5].params['x'] * _x + fits[5].params['Intercept']

# start and end coordinates of the quantile lines
p = np.column_stack((x, y))
a = np.array([_x[0], _y_005[0]])  # first point of 0.05 quantile fit line
b = np.array([_x[-1], _y_005[-1]])  # last point of 0.05 quantile fit line
a_ = np.array([_x[0], _y_095[0]])
b_ = np.array([_x[-1], _y_095[-1]])

# mask for coordinates above 0.95 or below 0.05 quantile lines
mask = lambda p, a, b, a_, b_: (np.cross(p - a, b - a) > 0) | (np.cross(p - a_, b_ - a_) < 0)
mask = mask(p, a, b, a_, b_)

# generate the plots
figure, axes = plt.subplots()
axes.scatter(x[mask], df['y'][mask], facecolor='r', edgecolor='none', alpha=0.3, label='data point')
axes.scatter(x[~mask], df['y'][~mask], facecolor='g', edgecolor='none', alpha=0.3, label='data point')
axes.plot(x, fit[0] * x + fit[1], label='best fit', c='lightgrey')
axes.plot(_x, _y_095, label=quantiles[5], c='orange')
axes.plot(_x, _y_005, label=quantiles[0], c='lightblue')
axes.legend()
axes.set_xlabel('x')
axes.set_ylabel('y')
plt.show()

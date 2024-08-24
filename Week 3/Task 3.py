import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def take_input():
    sizes = []
    prices = []
    
    n = int(input("Enter the number of data points: "))
    
    for i in range(n):
        size = float(input(f"Enter the size of the house in square feet for data point {i + 1}: "))
        price = float(input(f"Enter the price of the house in thousands of dollars for data point {i + 1}: "))
        sizes.append(size)
        prices.append(price)
    
    return np.array(sizes).reshape(-1, 1), np.array(prices)

sizes, prices = take_input()

X_train, X_test, y_train, y_test = train_test_split(sizes, prices, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred) if len(y_test) > 1 else float('nan')

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}" if not np.isnan(r2) else "R-squared: Not defined (need at least two test samples)")

plt.figure(figsize=(10, 6))

plt.scatter(sizes, prices, color='blue', label='Actual data')

sizes_range = np.linspace(sizes.min(), sizes.max(), 100).reshape(-1, 1)
prices_range = model.predict(sizes_range)
plt.plot(sizes_range, prices_range, color='red', linewidth=2, label='Regression line')

plt.title('House Size vs. Price')
plt.xlabel('Size of the House (in square feet)')
plt.ylabel('Price (in thousands of dollars)')
plt.legend()
plt.grid(True)
plt.show()


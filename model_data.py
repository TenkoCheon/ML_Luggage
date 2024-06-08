import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Step 1: Data Collection
modelData = {
    'length': [55, 60, 45, 50, 70],
    'width': [35, 40, 30, 33, 45],
    'height': [25, 30, 20, 22, 35],
    'type': ['suitcase', 'backpack', 'suitcase', 'duffel', 'suitcase'],
    'weight': [15, 10, 12, 8, 20]
}

df = pd.DataFrame(modelData)

# Step 2: Data Preprocessing
# Convert 'type' to dummy variables
df = pd.get_dummies(df, columns=['type'], drop_first=True)

# Step 3: Feature Engineering
# Create volume feature
df['volume'] = df['length'] * df['width'] * df['height']

# Step 4: Model Selection
X = df.drop('weight', axis=1)
y = df['weight']

# Step 5: Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')

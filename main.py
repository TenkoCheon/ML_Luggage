import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Step 1: Data Collection
data = {
    'length': [55, 60, 45, 50, 70],
    'width': [35, 40, 30, 33, 45],
    'height': [25, 30, 20, 22, 35],
    'type': ['suitcase', 'backpack', 'suitcase', 'duffel', 'suitcase'],
    'weight': [15, 10, 12, 8, 20]
}

df = pd.DataFrame(data)

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

import pandas as pd

jumlah_penumpang = int(input("Masukkan Jumlah barang: "))
data_barang = []

for i in range(jumlah_penumpang):
    length = float(input(f"Masukkan panjang untuk barang {i + 1}: "))
    width = float(input(f"Masukkan lebar untuk barang {i + 1}: "))
    height = float(input(f"Masukkan tinggi untuk barang {i + 1}: "))
    type_backpack = int(input(f"Apakah barang {i + 1} adalah tas ransel? (1 untuk ya, 0 untuk tidak): "))
    type_duffel = int(input(f"Apakah barang {i + 1} adalah tas duffel? (1 untuk ya, 0 untuk tidak): "))
    type_suitcase = int(input(f"Apakah barang {i + 1} adalah koper? (1 untuk ya, 0 untuk tidak): "))
    volume = length * width * height
    
    data_barang.append({
        'length': length,
        'width': width,
        'height': height,
        'type_backpack': type_backpack,
        'type_duffel': type_duffel,
        'type_suitcase': type_suitcase,
        'volume': volume
    })

new_luggage = pd.DataFrame(data_barang)

# Assuming X_train contains the same columns as the features used in training the model
# Replace X_train.columns with the appropriate column names from your training set
new_luggage = new_luggage[X_train.columns]

weight_predictions = model.predict(new_luggage)

for i, weight in enumerate(weight_predictions):
    print(f'Predicted weight for barang {i + 1}: {weight} kg')

total_weight = sum(weight_predictions)

print(f'Total predicted weight for {jumlah_penumpang} passengers: {total_weight} kg')


max_weight_plane = 200 #kg
if total_weight > max_weight_plane:
    print(f"Beban melebihi ketentuan")
else:
    print(f"Beban dibawah ketentuan dan pesawat boleh terbang")
import pandas as pd
import model_data

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
new_luggage = new_luggage[model_data.X_train.columns]

weight_predictions = model_data.model.predict(new_luggage)

for i, weight in enumerate(weight_predictions):
    print(f'Predicted weight for barang {i + 1}: {weight} kg')

total_weight = sum(weight_predictions)

print(f'Total predicted weight for {jumlah_penumpang} passengers: {total_weight} kg')


max_weight_plane = 200 #kg
if total_weight > max_weight_plane:
    print(f"Beban melebihi ketentuan")
else:
    print(f"Beban dibawah ketentuan dan pesawat boleh terbang")
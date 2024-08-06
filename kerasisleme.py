import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Excel dosyasını oku
df = pd.read_excel('output1.xlsx')

# Sayısal olmayan değerleri temizle
df['birinci_deger'] = pd.to_numeric(df['birinci_deger'].str.replace(',', '.'), errors='coerce')
df['ikinci_deger'] = pd.to_numeric(df['ikinci_deger'].str.replace(',', '.'), errors='coerce')

# NaN değerlerini kontrol et ve işleme
df = df.dropna()

# Sütunları sayısal formata dönüştür
df['birinci_deger'] = df['birinci_deger'].astype(float)
df['ikinci_deger'] = df['ikinci_deger'].astype(float)

# Giriş ve çıkış kolonlarını ayır
X = df[['referans', 'birinci_deger', 'ikinci_deger']]
y = df['cevap']

# Özellikleri ölçeklendir
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model oluşturma fonksiyonu
def create_model(layers, activation='relu', loss='mean_squared_error', output_activation=None):
    model = Sequential()
    model.add(Dense(layers[0], activation=activation, input_shape=(X_train.shape[1],)))
    for units in layers[1:]:
        model.add(Dense(units, activation=activation))
    model.add(Dense(1, activation=output_activation))
    model.compile(optimizer=Adam(), loss=loss)
    return model

# Model yapılarını (dört değer içerecek şekilde güncellenmiş)
model_configs = [
    ([64, 32], 'relu', 'mean_squared_error', None),
    ([128, 64], 'relu', 'mean_squared_error', None),
    ([64, 64, 32], 'relu', 'mean_squared_error', None),
    ([32, 16], 'relu', 'binary_crossentropy', 'sigmoid'),
    ([64, 32], 'relu', 'binary_crossentropy', 'sigmoid'),
    ([32, 16, 8], 'relu', 'binary_crossentropy', 'sigmoid'),
    ([128, 64, 32], 'relu', 'binary_crossentropy', 'sigmoid'),
    ([64, 64], 'relu', 'binary_crossentropy', 'sigmoid'),  # Binary classification için düzeltildi
    ([128, 64], 'relu', 'binary_crossentropy', 'sigmoid'),
    ([64, 32, 16], 'relu', 'binary_crossentropy', 'sigmoid'),
    ([32, 16, 8], 'relu', 'binary_crossentropy', 'sigmoid'),
    ([64, 32], 'tanh', 'mean_squared_error', None),
    ([128, 64], 'tanh', 'mean_squared_error', None),
    ([64, 64, 32], 'tanh', 'mean_squared_error', None),
    ([32, 16], 'tanh', 'binary_crossentropy', 'sigmoid')
]

# Modelleri oluştur ve eğit
results = {}
for i, (layers, activation, loss, output_activation) in enumerate(model_configs, start=1):
    print(f"Model {i} oluşturuluyor...")
    model = create_model(layers, activation, loss, output_activation)
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
    loss = model.evaluate(X_test, y_test, verbose=0)
    results[f"Model {i}"] = loss
    print(f"Model {i} kaydedildi: Test loss: {loss}")

# Sonuçları yazdır
for model_name, loss in results.items():
    print(f"{model_name}: Test loss: {loss}")

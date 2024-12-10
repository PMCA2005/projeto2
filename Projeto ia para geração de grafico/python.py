#gpt usado para comentar que eu não tenho costume de fazer
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Passo 1: Carregar os dados
df = pd.read_csv('World Energy Consumption.csv')

# Passo 2: Filtrar o consumo total de energia para 2020
df_2020 = df[df['year'] == 2020]

# Calcular o consumo total de energia para cada país
df_2020['total_energy_consumption'] = df_2020['biofuel_consumption'] + df_2020['coal_consumption'] + \
                                       df_2020['gas_consumption'] + df_2020['hydro_consumption'] + \
                                       df_2020['nuclear_consumption'] + df_2020['oil_consumption'] + \
                                       df_2020['solar_consumption'] + df_2020['wind_consumption'] + \
                                       df_2020['other_renewable_consumption']

# Ordenar os países por consumo de energia e pegar os 10 primeiros
top_10_consumers_2020 = df_2020.nlargest(10, 'total_energy_consumption')

# Exibir os 10 países/regiões que mais consumiram energia
print(top_10_consumers_2020[['country', 'total_energy_consumption']])

# Passo 3: Preprocessamento dos dados
# Preencher valores ausentes com 0 ou com média, dependendo da sua escolha
df.fillna(0, inplace=True)  # Substituindo valores ausentes por 0

# Filtrando os dados para incluir apenas os países das 10 maiores regiões
top_10_countries = top_10_consumers_2020['country'].values
df_top_10 = df[df['country'].isin(top_10_countries)]

# Calcular o consumo total de energia para cada linha
df_top_10['total_energy_consumption'] = df_top_10['biofuel_consumption'] + df_top_10['coal_consumption'] + \
                                         df_top_10['gas_consumption'] + df_top_10['hydro_consumption'] + \
                                         df_top_10['nuclear_consumption'] + df_top_10['oil_consumption'] + \
                                         df_top_10['solar_consumption'] + df_top_10['wind_consumption'] + \
                                         df_top_10['other_renewable_consumption']

# Passo 4: Treinando o modelo de previsão
model = LinearRegression()

# Dicionário para armazenar as previsões
predictions = {}

for country in top_10_countries:
    # Filtrar os dados para cada país
    country_data = df_top_10[df_top_10['country'] == country]
    
    # Usar o ano como variável independente (X) e o consumo de energia como variável dependente (y)
    X = country_data[['year']]  # Ano
    y = country_data['total_energy_consumption']  # Consumo total de energia
    
    # Treinar o modelo
    model.fit(X, y)
    
    # Prever o consumo para os próximos 10 anos (2021 a 2030)
    future_years = np.array([[i] for i in range(2021, 2031)])  # Anos de 2021 a 2030
    future_predictions = model.predict(future_years)
    
    # Salvar as previsões
    predictions[country] = future_predictions

# Exibir as previsões
for country, future_predictions in predictions.items():
    print(f"\nPrevisões para {country}:")
    for year, prediction in zip(range(2021, 2031), future_predictions):
        print(f"Ano {year}: {prediction:.2f} TWh")

# Passo 5: Visualizar os resultados
for country, future_predictions in predictions.items():
    plt.plot(range(2021, 2031), future_predictions, label=country)

plt.xlabel('Ano')
plt.ylabel('Consumo de Energia (TWh)')
plt.title('Previsão de Consumo de Energia para os Próximos 10 Anos')
plt.legend()
plt.show()

# Passo 6: Avaliar o modelo
mae = {}

for country in top_10_countries:
    # Filtrar os dados de treinamento para cada país
    country_data = df_top_10[df_top_10['country'] == country]
    
    # Usar o ano como variável independente (X) e o consumo de energia como variável dependente (y)
    X = country_data[['year']]  # Ano
    y = country_data['total_energy_consumption']  # Consumo total de energia
    
    # Prever o consumo com o modelo treinado
    y_pred = model.predict(X)
    
    # Calcular o erro absoluto médio (MAE)
    mae[country] = mean_absolute_error(y, y_pred)

print("\nErro Médio Absoluto (MAE) para cada país:")
for country, error in mae.items():
    print(f"{country}: {error:.2f} TWh")

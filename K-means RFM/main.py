"""
Zadanie
Otrzymałeś zadanie pogrupowania klientów dużego sklepu online. Użyj metody RFM (recency, frequency,
monetary value) oraz metody kwantyli (K-means) aby przekazać pracownikom działu marketingu kilka wersji segmentacji
behawioralnej. W końcowych rozważaniach scharakteryzuj powstałe grupy i wybierz model twoim zdaniem najlepszy.

Metoda RFM
RFM to narzędzie analizy marketingowej służące do identyfikacji najlepszych klientów
firmy lub organizacji za pomocą określonych miar.

Model RFM opiera się na trzech czynnikach:

    recency: jak niedawno klient dokonał zakupu
    frequency: jak często klient dokonuje zakupu
    monetary value: ile pieniędzy klient wydaje na zakupy

Analiza RFM pomaga firmom rozsądnie przewidzieć, którzy klienci są bardziej skłonni do ponownych
zakupów w przyszłości, ile przychodów pochodzi od nowych (w porównaniu z powracającymi klientami) i
jak zmienić okazjonalnych kupujących w stałych klientów.

Zbiór danych - atrybuty
    - Invoice - numer faktury, 6-cyfrowy numer jednoznacznie przypisany do każdej transakcji.
    Jeśli kod rozpoczyna się od litery 'c', oznacza to fakturę anulowaną.
    - StockCode - kod produktu, 5-cyfrowy numer jednoznacznie przypisany do każdego odrębnego produktu
    - Description - opis
    - Quantity - ilość każdego produktu na transakcję
    - InvoiceDate - data i godzina wystawienia faktury
    - Price - cena jednostkowa
    - Customer ID - numer klienta
    - Country - nazwa kraju, z którego pochodzi klient
"""

# region Biblioteki
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# endregion

# region Pandas options
pd.set_option('float_format', '{:.3f}'.format)
pd.set_option("display.max_columns", 999)
# endregion

# region Dataset
df = pd.read_csv('online_retail_II.csv')
df.columns = ['Invoice', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'Price', 'CustomerID', 'Country']
# endregion

# region Typy zmiennych
# Jak można zauważyć przy konwersji pliku danych wystąpiły pewne komplikacje. Sprawdzam jakie typy mają atrybuty.
print(df.dtypes)
# endregion

# region Brakujące wartości
summary = pd.DataFrame(df.dtypes, columns=['Dtype'])
summary['Nulls'] = pd.DataFrame(df.isnull().any())
summary['Sum'] = pd.DataFrame(df.isnull().sum())
summary.Dtype = summary.Dtype.astype(str)
print(summary)
# endregion

# region Ilość obserwacji
print('Pozostało ' + str(df.shape[0]) + ' obserwacji.')
# endregion

# region Braki danych jako procenty
print(str(round(df.isnull().any(axis=1).sum() / df.shape[0] * 100, 2)) + '% obserwacji zawiera braki w danych.')
# endregion

# region Usuwanie wierszy
df = df[~df.CustomerID.isnull()]
df = df[df.Invoice.astype('str').str.isdigit()]
# endregion

# region Usuwanie zbędnych atrybutów
df.drop(['Description', 'StockCode', 'Country'], axis=1, inplace=True)
# endregion

# region Zmiana typów
df['CustomerID'] = df.CustomerID.astype(int)
df['Invoice'] = df.Invoice.astype(int)
df['Quantity'] = df.Quantity.astype(int)
df['Price'] = df.Price.astype(float)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
print(df.dtypes)
# endregion

# region Stare transakcje
lastInvoice = df.InvoiceDate.max()
print(lastInvoice)
df = df[df.InvoiceDate >= lastInvoice - pd.to_timedelta(365 * 2, "D")]
print('Pozostało ' + str(df.shape[0]) + ' obserwacji.')
# endregion

# region Monetary value
df['MonetaryValue'] = df.Quantity * df.Price
print(df.MonetaryValue.describe())
# endregion

# region Recency
df['Recency'] = (lastInvoice - df.InvoiceDate) / np.timedelta64(1, 'D')
# endregion

# region Frequency
rfm = df.groupby(['CustomerID']).agg({'Recency': 'min', 'MonetaryValue': 'sum', 'Invoice': 'count'})
rfm.rename(columns={'Invoice': 'Frequency'}, inplace=True)
rfm = rfm[['Recency', 'Frequency', 'MonetaryValue']]
print(rfm)
# endregion

# region Transformacja zmiennych
r = pd.qcut(rfm.Recency, 4, labels=list(range(0, 4)))
f = pd.qcut(rfm.Frequency, 4, labels=list(range(0, 4)))
m = pd.qcut(rfm.MonetaryValue, 4, labels=list(range(0, 4)))
rfm_cutted = pd.DataFrame({'Recency': r, 'Frequency': f, 'MonetaryValue': m})
rfm_raw = rfm_cutted.values
print(rfm_raw)
# endregion

# region Wybór odpowiedniej liczby grup
res = []
for n in range(2, 20):
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(rfm_raw)
    res.append([n, kmeans.inertia_])
res = pd.DataFrame(res, columns=['liczbaGrup', 'inercja'])
print(res)

plt.figure(figsize=(10, 7))
sns.set(font_scale=1.4, style="whitegrid")
sns.lineplot(data=res, x='liczbaGrup', y='inercja')
plt.title("Miara odmienności grup vs liczba grup")
plt.axvline(x=4, linestyle='--')
plt.axvline(x=6, linestyle='--')
plt.axvline(x=8, linestyle='--')
plt.xticks(np.arange(2.0, 20, 1.0))
plt.show()
# endregion

# region Model dla n = 4
model = KMeans(n_clusters=4)
groups = model.fit_predict(rfm_raw)
rfm['groups'] = groups

print((rfm.groups.value_counts(normalize=True, sort=True) * 100).to_string())

print(rfm.groupby('groups').agg(['mean']))

'''
Interpretacja powstałych grup

Grupa 0 - klienci, którzy
    * robią zakupy za niewielkie kwoty
    * kupują rzadko
    * niedawno robili zakupy
    
Grupa 1 - klienci, którzy
    * robią zakupy za niskie kwoty
    * kupują rzadko
    * od bardzo dawna nie robili zakupów
    
Grupa 2 - klienci, którzy
    * robią zakupy za wysokie kwoty
    * kupują często
    * niedawno robili zakupy
    
Grupa 3 - klienci, którzy
    * robią zakupy za duże kwoty
    * kupują umiarkowanie często
    * od dawna nie robili zakupów
'''
# endregion

# region Model dla n = 6
model = KMeans(n_clusters=6)
groups = model.fit_predict(rfm_raw)
rfm['groups'] = groups

print((rfm.groups.value_counts(normalize=True, sort=True) * 100).to_string())

print(rfm.groupby('groups').agg(['mean']))

'''
Interpretacja powstałych grup

Grupa 0 - klienci, którzy
    * robią zakupy za bardzo niskie kwoty
    * kupują bardzo rzadko
    * od dawna nie robili zakupów
    
Grupa 1 - klienci, którzy
    * robią zakupy za bardzo wysokie kwoty
    * kupują bardzo często
    * niedawno robili zakupy
    
Grupa 2 - klienci, którzy
    * robią zakupy za duże kwoty
    * kupują umiarkowanie często
    * niedawno robili zakupy
    
Grupa 3 - klienci, którzy
    * robią zakupy za niskie kwoty
    * kupują  rzadko
    * niedawno robili zakupy
    
Grupa 4 - klienci, którzy
    * robią zakupy za wysokie kwoty
    * kupują często
    * od dłuższego czasu nie robili zakupów
    
Grupa 5 - klienci, którzy
    * robią zakupy za umiarkowane kwoty
    * kupują umiarkowanie często
    * od długiego czasu nie robili zakupów
'''
# endregion

# region Model dla n = 8
model = KMeans(n_clusters=8)
groups = model.fit_predict(rfm_raw)
rfm['groups'] = groups

print((rfm.groups.value_counts(normalize = True, sort = True) * 100).to_string())

print(rfm.groupby('groups').agg(['mean']))

'''
Interpretacja powstałych grup

Grupa 0 - klienci, którzy
    * robią zakupy za małe kwoty
    * kupują bardzo rzadko
    * od dawna nie robili zakupów
    
Grupa 1 - klienci, którzy
    * robią zakupy za wysokie kwoty
    * kupują często
    * od dłuższego czasu nie robili zakupów
    
Grupa 2 - klienci, którzy
    * robią zakupy za umiarkowane kwoty
    * kupują rzadko
    * jakiś czas temu robili zakupy
    
Grupa 3 - klienci, którzy
    * robią zakupy za umiarkowane kwoty
    * kupują rzadko
    * od bardzo dawna nie robili zakupów
    
Grupa 4 - klienci, którzy
    * robią zakupy za niemałe kwoty
    * kupują umiarkowanie często
    * jakiś czas temu robili zakupy
    
Grupa 5 - klienci, którzy
    * robią zakupy za ogromne pieniądze
    * kupują bardzo często
    * niedawno robili zakupy
    
Grupa 6 - klienci, którzy
    * robią zakupy za niemałe kwoty
    * kupują dosyć często
    * od dawna nie robili zakupów
    
Grupa 7 - klienci, którzy
    * robią zakupy za bardzo małe kwoty
    * kupują prawie nigdy
    * od bardzo dawna nie robili zakupów
'''
# endregion

import requests
import datetime
import csv
import time

# Binance API endpoint
base_url = "https://api.binance.com/api/v3/klines"

# Bugün
today = datetime.date.today()

# Son x gün için veri çekilecek
total_days = 730

# Sonuçları tutacak liste
results = []

# API isteğini yapacak fonksiyon
def fetch_data(start_time, end_time):
    symbol = "USDTTRY"
    interval = "1d"  # 1 günlük veriler
    limit = 1000 # Max veri sayısı

    url = f"{base_url}?symbol={symbol}&interval={interval}&startTime={start_time}&endTime={end_time}&limit={limit}"

    try:
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            if data:
                for entry in data:
                    timestamp = entry[0] / 1000  # ms -> s çevirme
                    date = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')  # UTC timestamp dönüştürme
                    close_price = entry[4]  # Kapanış fiyatı
                    results.append([date, close_price])
            else:
                print(f"No data found for {start_time}.")
        else:
            print(f"Error fetching data for {start_time}: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {e}")

# Başlangıç ve bitiş zamanları
start_date = today - datetime.timedelta(days=total_days)
start_time = int(time.mktime(start_date.timetuple()) * 1000)  # unix timestamp (ms)
end_time = int(time.mktime(today.timetuple()) * 1000)  # bugün unix timestamp (ms)

# Veriyi çek
fetch_data(start_time, end_time)

# Sonuçları tarihe göre sırala
results.sort(key=lambda x: datetime.datetime.strptime(x[0], "%Y-%m-%d"))

# Verileri CSV dosyasına yaz
with open('usd.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Date", "TRY"])
    writer.writerows(results)

print(f"Son {total_days} günün verileri 'usd.csv' dosyasına kaydedildi.")
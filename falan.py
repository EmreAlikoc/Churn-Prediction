import joblib

factorize_map = joblib.load("log_factorize_map.pkl")
print(f"Toplam sütun: {len(factorize_map)}")
print(f"İlk 5 mapping örneği:\n")
for k, v in list(factorize_map.items())[:5]:
    print(f"{k}: {list(v.items())[:3]}")
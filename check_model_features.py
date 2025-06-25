import joblib

# Modeli yükle
model_path = "final_log_model.pkl"  # Dosya adını burada değiştir
loaded_model = joblib.load(model_path)

# Eğer model bir dictionary içinde değilse doğrudan model nesnesi olarak kontrol et
if hasattr(loaded_model, "feature_names_in_"):
    print("✅ Bu model .feature_names_in_ desteğine sahip.")
    print("Eğitim sırasında kullanılan özellikler:")
    print(loaded_model.feature_names_in_)
else:
    print("❌ Bu modelde .feature_names_in_ özelliği yok.")

    # Eğer model bir dict ise (örneğin {'model': ..., 'features': ...} gibi)
    if isinstance(loaded_model, dict):
        if "features" in loaded_model:
            print("📝 Model .feature_names_in_ özelliği içermiyor ama 'features' anahtarında kayıtlı.")
            print("Eğitimde kullanılan sütunlar:")
            print(loaded_model["features"])
        else:
            print("❗ Model dict formatında ama içinde 'features' bilgisi yok.")

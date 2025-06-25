import streamlit as st
import pandas as pd
import numpy as np
import joblib


import warnings
warnings.filterwarnings("ignore")


st.set_page_config(page_title="Churn Prediction", layout="wide")
st.title("📉 Churn Prediction Uygulaması")
st.markdown("Müşteri verilerini içeren bir `.xlsx` dosyası yükleyin ve kullanmak istediğiniz modeli seçin.")

model_options = {
    "Random Forest - Dengeli genel performans": "Random Forest",
    "Logistic Regression - Churn olan müşteriler için daha yüksek recall": "Logistic Regression"
}

selected_label = st.selectbox("Kullanmak istediğiniz modeli seçin:", list(model_options.keys()))
model_name = model_options[selected_label]

@st.cache_resource
def load_model(name):
    if name == "Random Forest":
        return joblib.load("final_random_forest_model.pkl")
    elif name == "Logistic Regression":
        return joblib.load("final_log_model.pkl")

# ✅ Modeli yükle
model = load_model(model_name)


# Excel dosyasını yükleme
uploaded_file = st.file_uploader("📂 Dosya Seç (.xlsx)", type=["xlsx"])


if uploaded_file is not None:
    try:
        # Veriyi oku
        df = pd.read_excel(uploaded_file)

        st.subheader("🔍 Yüklenen Veri")
        st.dataframe(df.head())

        model_features = [
        'CustomerID', 'Gender', 'Senior Citizen', 'Partner', 'Dependents',
        'Tenure Months', 'Phone Service', 'Multiple Lines',
        'Internet Service', 'Online Security', 'Online Backup',
        'Device Protection', 'Tech Support', 'Streaming TV',
        'Streaming Movies', 'Contract', 'Paperless Billing',
        'Payment Method', 'Monthly Charges', 'Total Charges','CLTV'
        ]

        if set(model_features) - set(df.columns):
            st.error("❌ Modelin beklediği bazı sütunlar dosyada yok. Lütfen kontrol edin.")
            st.stop()

        uploaded_cols = df.columns.tolist()
        extra_cols = [col for col in uploaded_cols if col not in model_features]

        if extra_cols:
            st.warning(f"⚠️ Modelin kullanmadığı şu {len(extra_cols)} sütun(lar) bulundu ve kaldırıldı: {extra_cols}")
            df = df.drop(columns=extra_cols)


        missing_summary = df.isnull().sum()
        total_missing = missing_summary[missing_summary > 0]

        if not total_missing.empty:
            # 💡 Eksik verileri doldurma stratejisi
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    if df[col].dtype in ['float64', 'int64']:
                        # Sayısal sütunlar: ortalama ile doldur
                        df[col].fillna(df[col].mean(), inplace=True)
                    else:
                        # Kategorik sütunlar: en sık görülen (mod) ile doldur
                        df[col].fillna(df[col].mode()[0], inplace=True)


        customer_ids = df['CustomerID']
        df = df.drop(columns=['CustomerID'])

        if model_name=='Logistic Regression':
            
            @st.cache_resource
            def load_log_features():
                return joblib.load("log_features.pkl")

            FEATURES = load_log_features()
            CATS = []
            for c in FEATURES:
                if df[c].dtype == "object":
                    CATS.append(c)
                    df[c] = df[c].fillna("NAN")

            combo_data = {}
            new_features = []

            for i in range(len(CATS)):
                for j in range(i + 1, len(CATS)):
                    col1, col2 = CATS[i], CATS[j]
                    new_col = f"{col1}_{col2}"
                    combo_data[new_col] = df[col1].astype(str) + "_" + df[col2].astype(str)
                    new_features.append(new_col)

            df = pd.concat([df, pd.DataFrame(combo_data, index=df.index)], axis=1)

            ENGINEERED_CATS = CATS + new_features
            ENGINEERED_FEATURES = FEATURES + new_features

            @st.cache_resource
            def load_factorize_mappings():
                return joblib.load("log_factorize_map.pkl")

            factorize_mappings = load_factorize_mappings()
            print(len(factorize_mappings))

            for c in ENGINEERED_FEATURES:
                if c in ENGINEERED_CATS:
                    if c in factorize_mappings:
                        mapping = factorize_mappings[c]
                        df[c] = df[c].map(mapping).fillna(-1).astype("int32")
                    else:
                        st.warning(f"⚠️ '{c}' için factorize mapping bulunamadı!")
                else:
                    if df[c].dtype == "float64":
                        df[c] = df[c].astype("float32")
                    if df[c].dtype == "int64":
                        df[c] = df[c].astype("int32")
            '''
            df = df.apply(lambda col: col.cat.codes if col.dtypes.name == 'category' else col)

            for col in df.columns:
                if np.issubdtype(df[col].dtype,np.integer):
                    df[col] = df[col].astype('category')
            '''
            @st.cache_resource
            def load_expected_log_features():
                return joblib.load("log_expected_features.pkl")
            
            expected_log_features = load_expected_log_features()

            # Eksik sütunları 0 ile tamamla
            for col in expected_log_features:
                if col not in df.columns:
                    df[col] = 0

            # Fazla sütunları kaldır
            df = df[expected_log_features]


        else:
            df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
            df["AvgChargePerMonth"] = df["Total Charges"] / (df["Tenure Months"] + 1)

            # Yüksek fiyatla yeni başlayan kullanıcıları yakalamak için
            df["HighRiskNew"] = ((df["Monthly Charges"] > 80) & (df["Tenure Months"] < 6)).astype(int)

            @st.cache_resource
            def load_encoders():
                return joblib.load("label_encoders.pkl")

            label_encoders = load_encoders()


            for col in df.select_dtypes(include='object').columns:
                if col in label_encoders:
                    df[col] = label_encoders[col].transform(df[col])
            '''
            def encode_data(dataframe_series):
                if dataframe_series.dtype=='object':
                    dataframe_series = LabelEncoder().fit_transform(dataframe_series)
                return dataframe_series

            df = df.apply(lambda x: encode_data(x))
            '''
    
    except Exception as e:
        st.error(f"{e}")  
    

    # Model tahmini
    predictions = model.predict(df)

    results_df = pd.DataFrame({
        'CustomerID': customer_ids,
        'Churn Prediction': predictions
    })

    # Daha okunabilir hale getir
    results_df['Churn Prediction'] = results_df['Churn Prediction'].map({0: "Not Churn", 1: "Churn"})

    st.subheader("📊 Tahmin Sonuçları")
    st.dataframe(results_df)

    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    csv_data = convert_df_to_csv(results_df)

    st.download_button(
        label="📥 Sonuçları İndir (CSV)",
        data=csv_data,
        file_name="tahmin_sonuclari.csv",
        mime='text/csv',
    )

        


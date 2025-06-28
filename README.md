# Churn Prediction 📈

Bu proje, makine öğrenmesi teknikleri kullanarak müşterilerin hizmetten ayrılma (churn) ihtimalini tahmin etmeyi amaçlamaktadır. Özellikle churn oranının dengesiz olduğu veri setlerinde, doğru öznitelik mühendisliği, veri dengesizliği yönetimi ve uygun model seçimiyle başarılı bir tahmin modeli geliştirilmiştir.

## İçindekiler

- [Proje Amacı](#proje-amacı)
- [Veri Analizi ve Temizliği](#veri-analizi-ve-temizliği)
- [Öznitelik Mühendisliği (Feature Engineering)](#öznitelik-mühendisliği-feature-engineering)
- [Veri Dengesizliği Problemi ve SMOTE](#veri-dengesizliği-problemi-ve-smote)
- [Modelleme ve Performans Karşılaştırması](#modelleme-ve-performans-karşılaştırması)
- [Model Encoding ve Haritalama](#model-encoding-ve-haritalama)
- [Streamlit Dashboard](#streamlit-dashboard)

## Proje Amacı

Telekomünikasyon sektöründeki müşteri kaybını (churn) önceden tahmin edebilmek işletmeler için oldukça kritik bir konudur. Bu proje, veri analizi, öznitelik mühendisliği, veri dengesizliği ile başa çıkma ve çeşitli makine öğrenmesi modelleri kullanarak bu tahmini en doğru şekilde gerçekleştirmeyi hedeflemektedir.

## Veri Analizi ve Temizliği

- *Eksik Veri Kontrolleri:*  
  Veri seti ilk olarak eksik değerler açısından detaylı şekilde incelendi. Eksik veriler uygun yöntemlerle temizlendi.

- *Veri Görselleştirmeleri:*  
  Özellik dağılımları, korelasyon analizleri ve özellikle churn değişkeninin dağılımı matplotlib ve seaborn kullanılarak görselleştirildi.

- *Churn Dağılımı:*  
  Veri setinde churn oranının yalnızca %26 olduğu tespit edildi. Bu dengesizlik, model performansı üzerinde önemli bir dezavantaj oluşturabileceğinden sonraki adımlarda dikkate alındı.

## Öznitelik Mühendisliği (Feature Engineering)

Veri setinin daha anlamlı hale gelmesi için aşağıdaki işlemler uygulandı:

- *Özellik İlişkilendirme (Engineered Features):*  
  Her bir özelliğin diğer özelliklerle olan ilişkisi analiz edilerek yeni anlamlı öznitelikler üretildi.

- *Farklı Encoding Yöntemleri:*  
  - *Label Encoding:* Bazı modeller için daha uygun olduğu düşünülen kategorik değişkenlerde kullanıldı.  
  - *Feature Relationship Encoding:* Özellikler arasındaki ilişkiler özel olarak encode edilerek model girdileri daha anlamlı hale getirildi.

- *Encoding Map:*  
  Daha sonra production ortamında veya Streamlit dashboard üzerinden veri gönderilirken encoding karmaşasını önlemek için kullanılan tüm encoding işlemleri ve mapping yapıları kaydedildi.

## Veri Dengesizliği Problemi ve SMOTE

- *Dengesizlik Problemi:*  
  Churn=1 sınıfının düşük sayıda olması, modelin bu sınıfı doğru tahmin etmesini zorlaştırmaktaydı.

- *SMOTE (Synthetic Minority Over-sampling Technique):*  
  Eğitim setinde churn=1 ve churn=0 sınıfları SMOTE kullanılarak %50-%50 dengelendi. Böylece model, churn olan müşterileri daha iyi öğrenebildi.

## Modelleme ve Performans Karşılaştırması

- *Çeşitli Makine Öğrenmesi Modelleri* denendi:

  - Logistic Regression
  - Decision Tree
  - Random Forest
  - XGBoost
  - Diğer temel modeller

- *Seçim Kriterleri:*  
  - Genel doğruluk (Accuracy)  
  - Churn=1 sınıfını yakalama performansı (Recall, Precision for churn=1)  

- *Sonuç:*  
  İki farklı model öne çıktı:  
  - Biri genel performansı yüksek olan model  
  - Diğeri churn=1 sınıfındaki hataları minimuma indiren model  

> Kullanıcılar, Streamlit arayüzü üzerinden bu iki model arasında tercih yapabilir.

## Model Encoding ve Haritalama

Model eğitim sürecinde kullanılan tüm encoding işlemleri ve mapping yapıları kaydedildi. Böylece dashboard tarafında son kullanıcı veri gönderdiğinde encoding hatalarının önüne geçildi.

## Streamlit Dashboard

- Kullanıcı dostu bir arayüz ile *Streamlit* kullanılarak basit bir dashboard tasarlandı.
- Kullanıcılar:
  - İstedikleri modeli seçebilir
  - Yeni müşteri verisi girerek churn tahmini alabilir




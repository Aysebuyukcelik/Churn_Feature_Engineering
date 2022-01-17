# Churn_Feature_Engineering

![image](https://user-images.githubusercontent.com/84872652/149845261-70ce7a79-6c63-4531-9e19-2a39a051750f.png)

Bu çalışmada, üçüncü çeyrekte Kaliforniya'daki 7043 müşteriye ev telefonu ve İnternet hizmetleri sağlayan hayali bir telekom şirketi hakkındaki müşteri kaybı verileri kullanılmıştır. Uçtan uca modellenerek müşteri kayıp tahmini yapılmıştır.

Veri setini yakından tanıayacak olursak 21 değişkenden oluşan bu verisetinde değişkenler:

CustomerId - Müşteri İd’si

Gender - Cinsiyet

SeniorCitizen - Müşterinin yaşlı olup olmadığı (1, 0)

Partner - Müşterinin bir ortağı olup olmadığı (Evet, Hayır)

Dependents - Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı (Evet, Hayır

tenure - Müşterinin şirkette kaldığı ay sayısı

PhoneService - Müşterinin telefon hizmeti olup olmadığı (Evet, Hayır)

MultipleLines - Müşterinin birden fazla hattı olup olmadığı (Evet, Hayır, Telefon hizmeti yok)

InternetService - Müşterinin internet servis sağlayıcısı (DSL, Fiber optik, Hayır)

OnlineSecurity - Müşterinin çevrimiçi güvenliğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)

OnlineBackup - Müşterinin online yedeğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)

DeviceProtection - Müşterinin cihaz korumasına sahip olup olmadığı (Evet, Hayır, İnternet hizmeti yok)

TechSupport - Müşterinin teknik destek alıp almadığı (Evet, Hayır, İnternet hizmeti yok)

StreamingTV - Müşterinin TV yayını olup olmadığı (Evet, Hayır, İnternet hizmeti yok)

StreamingMovies - Müşterinin film akışı olup olmadığı (Evet, Hayır, İnternet hizmeti yok)

Contract - Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl)

PaperlessBilling - Müşterinin kağıtsız faturası olup olmadığı (Evet, Hayır)

PaymentMethod - Müşterinin ödeme yöntemi (Elektronik çek, Posta çeki, Banka havalesi (otomatik), Kredi kartı (otomatik))

MonthlyCharges - Müşteriden aylık olarak tahsil edilen tutar

TotalCharges - Müşteriden tahsil edilen toplam tutar

Churn - Müşterinin kullanıp kullanmadığı (Evet veya Hayır)

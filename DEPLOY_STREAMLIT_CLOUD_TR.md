# STREAMLIT COMMUNITY CLOUD KURULUM REHBERİ
## MK Trend Following Analytics Engine v0.01
### By Murat Konuklar

## 1. ZIP'i aç
`MK_Trend_Following_Streamlit_Cloud_v001.zip` dosyasını bilgisayarında aç.

ZIP içindeki dosyaları tek bir GitHub repository köküne yükleyeceğiz.

## 2. GitHub repository oluştur
Önerilen repository adı:

`mk-trend-following-engine`

Repository içinde `app.py` kök dizinde kalmalı.

Özellikle şu gizli klasörlerin de GitHub'a yüklendiğini kontrol et:

- `.streamlit/`
- `.github/`

## 3. Repository yapısını kontrol et

Kök dizinde en az şunlar görünmeli:

- `app.py`
- `MK_Trend_Following_Engine_v001.py`
- `MK_Trend_Following_HTML_Report_v001.py`
- `requirements.txt`
- `smoke_test.py`
- `README.md`

Ayrıca:

- `.streamlit/config.toml`
- `.github/workflows/validate.yml`

bulunmalı.

## 4. GitHub validation
Dosyaları `main` branch'ine commit ettiğinde GitHub Actions içindeki
`Validate Trend Engine` workflow'u otomatik çalışır.

Bu workflow canlı Yahoo verisi istemeden Golden Master'ı test eder.

Başarılı sonuç:

`PASS — Golden Master replication`

olmalıdır.

## 5. Streamlit Community Cloud
Streamlit Community Cloud workspace'ine giriş yap.

- `Create app`
- `Yup, I have an app`
- Repository: GitHub'daki repository
- Branch: `main`
- Main file path: `app.py`

## 6. Advanced settings
`Advanced settings` ekranını aç.

Python sürümü:

`3.12`

olarak seç.

Bu v0.01 sürümünde secret/API key yoktur; Secrets alanını boş bırak.

## 7. Deploy
Deploy işlemini başlat.

Cloud `requirements.txt` içindeki Python bağımlılıklarını kuracak ve
repository kökünden `app.py` dosyasını çalıştıracak.

## 8. Uygulama açıldığında test
İlk basit kontrol için:

- Ticker: `AAPL`
- Frequency: `Daily`
- Strategy: `ATR Trailing Stop`
- Legacy Fidelity: `On`

Uygun tarih aralığını seç ve `RUN ANALYSIS` butonuna bas.

Yahoo Finance geçerli veri döndürürse dashboard oluşur.

Yahoo erişimi başarısız olursa sistem:

`STRICT DATA STOP`

göstermelidir.

Bu davranış bilinçlidir.

Başka veri sağlayıcıya geçiş yapılmayacaktır.

## 9. BIST kullanımı
Yahoo ticker formatı kullanılmalı.

Örnek:

- `THYAO.IS`
- `GARAN.IS`
- `ASELS.IS`
- `ASTOR.IS`
- `XU100.IS`

Ticker Yahoo üzerinde bulunmuyorsa sistem fallback kullanmaz.

## 10. Güncelleme
Kodda değişiklik yaptıktan sonra GitHub'a commit/push et.

Streamlit Community Cloud repository değişikliklerini algılar ve uygulamayı yeniden çalıştırır.
Dependency değişirse `requirements.txt` güncellenmelidir.

## 11. Python sürümünü değiştirmek gerekirse
Python interpreter sürümü normal dependency update gibi değiştirilmez.
Community Cloud tarafında yeni Python sürümüne geçiş gerekirse uygulamayı uygun Advanced Settings ile
yeniden deploy etmek gerekir.

## 12. Kurumsal tema
`.streamlit/config.toml` içerisinde:

- Light institutional theme
- 300 base font weight
- neutral colors
- restrained accent
- minimal toolbar
- small radius

sabitlenmiştir.

`app.py` ayrıca ince font ve hedge-fund dashboard CSS katmanını uygular.

## 13. Veri yönetişimi
Değiştirilemez proje prensipleri:

- Synthetic data: YOK
- Fallback market-data source: YOK
- Forward fill: YOK
- Backfill: YOK
- Yahoo validation failure: HARD STOP
- Silent substitution: YOK

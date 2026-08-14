# Streamlit Cloud — v0.06 Nadaraya-Watson Güncellemesi

GitHub repository'nizi silmeyin. Mevcut doğrulanmış dosyaları koruyun.

## Yeni eklenecek dosyalar

- `MK_Nadaraya_Watson_Trend_v006.py`
- `MK_Nadaraya_Watson_HTML_Report_v006.py`
- `nadaraya_watson_validation_v006.py`
- `NADARAYA_WATSON_INTEGRATION_v006.md`

## Değiştirilecek dosyalar

- `app.py`
- `.github/workflows/validate.yml`
- `cold_start_regression_test_v0051.py` (state-schema kontrolü v3)

## Korunacak çekirdek dosyalar

- `MK_Trend_Following_Engine_v001.py`
- `MK_Trend_Following_Decision_Engine_v002.py`
- `MK_Trend_Following_Universe_v002.py`
- `MK_Trend_Following_Risk_Analytics_v004.py`
- `MK_Trend_Following_Entry_Gate_v005.py`
- Golden Master CSV ve test dosyaları
- `.streamlit/config.toml`

## Deploy sonrası

Streamlit uygulamasında sol panelde `Nadaraya-Watson Trend Module` bölümü görünmelidir.

Varsayılan araştırma ayarı:

- Enable NW Research Layer: ON
- Preset: MK Institutional Balanced
- Strategy Logic: MK Confirmed Trend
- Price Source: Adjusted Close
- Avoid Entry Above Upper Residual Band: ON

`RUN ANALYSIS` sonrasında yeni ana tab:

`Nadaraya-Watson Trend`

oluşmalıdır.

GitHub Actions içinde hem eski Golden Master hem de `Nadaraya-Watson Trend v0.06 validation` testi PASS vermelidir.

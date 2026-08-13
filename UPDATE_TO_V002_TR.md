# MK Trend Following Analytics Engine — v0.02 Güncelleme Rehberi
## By Murat Konuklar

Mevcut Streamlit Cloud uygulamasını silmeye veya yeniden repository kurmaya gerek yoktur.

## GitHub'da güncellenecek / eklenecek dosyalar

Repository root'unda aşağıdaki dosyaları yükleyin:

1. `app.py` — mevcut dosyanın yerine v0.02 sürümü.
2. `MK_Trend_Following_Decision_Engine_v002.py` — yeni dosya.
3. `MK_Trend_Following_Universe_v002.py` — yeni dosya.
4. `MK_Trend_Following_HTML_Report_v002.py` — yeni dosya.
5. `.github/workflows/validate.yml` — mevcut validation workflow'unu v0.02 testleriyle güncelleyin.

Çekirdek `MK_Trend_Following_Engine_v001.py` değiştirilmemiştir. Golden Master matematiği korunmuştur.

## Streamlit Cloud tarafında

GitHub commit sonrasında Community Cloud normalde repository değişikliğini algılar ve uygulamayı yeniden çalıştırır.
Gerekirse Streamlit uygulama menüsünden **Reboot app** kullanın.

## v0.02'de gelen ana değişiklikler

- BIST instrument universe ve sektör seçimi.
- US Stocks universe ve sektör seçimi.
- Precious Metals altında Futures ve Exchange-Traded Products listeleri.
- Manual Yahoo ticker seçeneği korunur.
- BUY / HOLD / SELL / WAIT-CASH portfolio-level decision engine.
- Raw legacy signal ile gerçek portfolio action ayrıştırılır.
- Decision Causality Matrix: breakout, stop, position ve execution gate'leri.
- Candlestick + volume + rolling max + active stop + executed BUY/SELL markers.
- 1M / 3M / 6M / YTD / 1Y / 3Y / ALL range selector'ları.
- Mouse zoom, hover ve legend trace toggle.
- Strategy vs Buy & Hold equity chart.
- Drawdown chart.
- Rolling return / volatility chart.
- All-stop Trend Diagnostics chart.
- Trade Ledger ve closed-trade statistics.
- Filterable Instrument Universe tab.
- Standalone HTML v0.02 içinde Decision & Causality ve interaktif grafikler.

## Karar mantığı

`Signal` ile `Decision` aynı şey değildir.

- `Signal` legacy worksheet'in ham trigger alanıdır.
- `Decision` pozisyon durumunu da dikkate alan portfolio-level aksiyondur.

### BUY
Önceki tamamlanmış barın adjusted close değeri önceki rolling maximum'a ulaşır/aşar ve portföy cash'tedir.
İşlem mevcut barın adjusted open fiyatından yürütülür.

### HOLD
Portföy zaten long'dur ve executable SELL oluşmamıştır.
Legacy raw signal tekrar BUY olsa bile pyramiding yapılmadığı için portfolio action HOLD olur.

### SELL
Önceki tamamlanmış barın adjusted close değeri aktif stop threshold'a eşit veya altındadır ve portföy long'dur.
Satış mevcut barın adjusted open fiyatından yürütülür.

### WAIT / CASH
Portföy cash'tedir ve executable BUY yoktur. Raw SELL gelse dahi satılabilecek pozisyon olmadığı için cash'te beklenir.

### REDUCE neden yok?
Orijinal workbook all-in/all-out çalışır. Partial position reduction doğrulanmış legacy matematiğinin parçası değildir.
Bu nedenle v0.02 Legacy Fidelity içinde REDUCE uydurulmamıştır.

## Veri yönetişimi

Değişmedi:

- Yahoo Finance only.
- Synthetic data yok.
- Alternate vendor fallback yok.
- Forward fill yok.
- Backfill yok.
- Yahoo validation failure = HARD STOP.

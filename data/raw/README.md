# Ham veri

Bu klasöre şu dosyalar konur (değiştirilmez, sadece okunur):

| Dosya | İçerik | Kaynak |
|---|---|---|
| `49_sektor_gunluk_getiri_temiz.csv` | 49 sektörün günlük getirisi (%), **value-weighted**, 1926-07 – 2026-09 | Kenneth R. French Data Library |
| `ff3_daily_clean.csv` | Mkt-RF, SMB, HML, RF (%) | Kenneth R. French Data Library |
| `Siccodes49.txt` (repoda yok) | SIC kodu → sektör eşlemesi | Kenneth R. French Data Library |

Notlar:
- Birim yüzde: `0.45` = %0,45. Bileşik hesapta `(1 + r/100)`.
- Sektör dosyasının ilk sütun adında gizli bir BOM karakteri var; `data/io.py` bunu temizliyor.
- Erken dönemde bazı sektörler boş (henüz yoklar). 1979 sonrası eksik değer yok.
- Sektör getirileri value-weighted (teyit edildi); piyasa faktörü (Mkt-RF) de value-weighted olduğu için tutarlı.

# Ham veri

Bu klasöre şu dosyalar konur (değiştirilmez, sadece okunur):

| Dosya | İçerik | Kaynak |
|---|---|---|
| `49_sektor_gunluk_getiri_temiz.csv` | 49 sektörün günlük getirisi (%), 1926-07 – 2026-09 | Kenneth R. French Data Library |
| `ff3_daily_clean.csv` | Mkt-RF, SMB, HML, RF (%) | Kenneth R. French Data Library |
| `Siccodes49.txt` | SIC kodu → sektör eşlemesi | Kenneth R. French Data Library |

Notlar:
- Birim yüzde: `0.45` = %0,45. Bileşik hesapta `(1 + r/100)`.
- Sektör dosyasının ilk sütun adında gizli bir BOM karakteri var; `data/io.py` bunu temizliyor.
- Erken dönemde bazı sektörler boş (henüz yoklar). 2000 sonrası eksik değer yok.
- **Açık soru:** Sektör dosyası value-weighted mı, equal-weighted mı? İndiren kişiye sorulacak, rapora yazılacak.

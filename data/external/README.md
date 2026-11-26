# Dış parametreler

| Dosya (planlanan) | İçerik | Durum |
|---|---|---|
| `events_vienna.csv` | Viyana'daki üniversitenin dünya olayları veri seti | Formatı netleştirilecek (tarih, olay tipi, ülke, şiddet?) |
| `ads_index.csv` | Philadelphia Fed Aruoba–Diebold–Scotti günlük iş koşulları endeksi (1960+) | İndirilecek |

Hizalama kuralları (`data/` altında bir `external.py` yazılırken):
- Olay tarihi = **piyasanın olayı öğrendiği gün.** ABD kapanışından sonra olan olay t+1'e kayar.
- Sonradan tarihlenen olaylar (ör. "krizin başlangıcı") look-ahead riski taşır, işaretlenmeli.
- Modeller dış parametreyi t−1'den alır (`models/base.lagged_design`).

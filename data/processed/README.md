# İşlenmiş veri (`data/processed/`)

> Yapay zekâ araçları ve ayrıntılı veri sözlüğü için: [`AI_CONTEXT.md`](AI_CONTEXT.md)

Bu klasördeki `*_1983.csv` ve `*_1985.csv` (artık / beta / oynaklık) dosyalarının hepsi tek komutla üretilir:

```bash
python scripts/00_build_processed_data.py      # ~15 saniye
```

Kaynak: `data/raw/49_sektor_gunluk_getiri_temiz.csv` (value-weighted) ve `data/raw/ff3_daily_clean.csv`.
Birim her yerde **yüzde**: `0.45` = %0,45.

## Hangi dosyayı kullanmalıyım?

| Amaç | Dosya |
|---|---|
| **Ana analiz girdisi** (sektörler arası etki analizi) | `artik_ff3_haftalik_1985.csv` |
| Sektöre özgü oynaklık | `oynaklik_rv_ff3_haftalik_1985.csv` |
| Günlük sağlamlık kontrolü | `artik_ff3_gunluk_1985.csv` |
| “FF3 olmadan ne olurdu?” karşılaştırması | `49_sektor_haftalik_1983.csv` (1985+ kısmı) |

## Dosyalar

Her biri dört frekansta var: `gunluk`, `haftalik`, `2haftalik`, `aylik`.

| Dosya | Başlangıç | İçerik |
|---|---|---|
| `49_sektor_<frekans>_1983.csv` | 1983 | Ham sektör getirileri. Haftalık/2 haftalık/aylıkta her sektör için üç sütun: `<sektör>` (dönem getirisi), `<sektör>_min` ve `<sektör>_max` (dönem içindeki en düşük / en yüksek **günlük** getiri) |
| `ff3_<frekans>_1983.csv` | 1983 | `Mkt-RF`, `SMB`, `HML`, `RF` aynı dönemlere birleştirilmiş + `n_gun` (dönemdeki işlem günü sayısı) |
| `artik_ff3_<frekans>_1985.csv` | 1985 | **FF3 artıkları:** (sektör − RF) − (α + β₁·Mkt-RF + β₂·SMB + β₃·HML). 49 sütun |
| `beta_ff3_<frekans>_1985.csv` | 1985 | Artığı hesaplarken kullanılan betalar: `<sektör>_mkt`, `<sektör>_smb`, `<sektör>_hml` |
| `oynaklik_rv_ff3_<frekans>_1985.csv` | 1985 | **Gerçekleşen varyans (%²):** dönem içindeki günlük FF3 artıklarının kareleri toplamı + `n_gun`. Sadece haftalık, 2 haftalık, aylık |

CAPM artıkları varsayılan olarak üretilmez. Sağlamlık kontrolü gerekirse:
`python scripts/00_build_processed_data.py --capm` → `artik_capm_<frekans>_1985.csv` ve `beta_capm_<frekans>_1985.csv`.

## Kurallar

**Birleştirme** (`src/sektorgraf/data/aggregate.py`):
- Getiriler **bileşik** birleştirilir: `[(1+r₁/100)·…·(1+rₙ/100) − 1]·100`.
- Haftalar **Cuma** biter. 2 haftalık dönemler 11 Ocak 1985'te biten çifte göre hizalıdır.
- Dönemin etiketi (tarih), dönemin **son işlem günüdür** (Cuma tatilse Perşembe; aylıkta ayın son işlem günü).
- `Mkt-RF` bir fazla getiri olduğu için doğrudan bileşiklenmez: `Mkt = Mkt-RF + RF` ve `RF` ayrı ayrı bileşiklenip farkı alınır. (Doğrudan bileşiklemeyle fark haftada ortalama 0,0008 puan.)
- `SMB` ve `HML` günlük değerlerinden bileşiklenir. Bu, uzun ve kısa bacakların ayrı bileşiklenmesine çok yakın bir yaklaşımdır.

**Artıklar ve betalar** (`src/sektorgraf/data/residuals.py`, `rolling_factor_model`):
- Betalar **kayan pencereyle** her dönem yeniden tahmin edilir. t döneminin betası **sadece [t−pencere, t−1]** dönemlerinden gelir (look-ahead yok, `tests/test_aggregate.py` kontrol ediyor).
- Regresyon dönem verisi üzerinde yapılır (haftalık artık = haftalık getirilerle tahmin edilmiş haftalık betalar).

| Frekans | Beta penceresi | Pencerede en az gözlem | Isınma için kullanılan ham veri |
|---|---|---|---|
| Günlük | 252 gün (~1 yıl) | 200 | 1984 |
| Haftalık | 104 hafta (2 yıl) | 84 | 1983–1984 |
| 2 haftalık | 52 dönem (2 yıl) | 42 | 1983–1984 |
| Aylık | 60 ay (5 yıl) | 48 | 1980–1984 |

Aylıkta 5 yıllık pencere standarttır (2 yılda sadece 24 gözlem olurdu). Bu yüzden aylık artıklar, 1983 dosyasından daha eski (1980–1982) ham veriyle ısınır. Bütün artık dosyaları 1985'te başlar ve eksik değer içermez.

**Oynaklık:** Günlük FF3 artıkları (252 günlük pencere) üzerinden hesaplanır. 4 günlük tatil haftalarında değer mekanik olarak düşüktür. Gerekirse `5 / n_gun` ile ölçekleyin.

## Dikkat edilecekler

- **Son dönem kısmi:** Veri 30 Eylül 2026 Çarşamba bitiyor. Son hafta 3 gün, son 2 haftalık dönem ve son ay da eksik. Modellemede son satırı atmak veya işaretlemek gerekir.
- **Kısa haftalar:** Haftalıkta 382 hafta 4 gün, 3 hafta 3 gün, 1 hafta 1 gün (11 Eylül 2001 sonrası borsa kapanışı).
- **Eski `49_sektor_<frekans>_1985.csv` dosyaları kaldırıldı.** Yerlerini `_1983` dosyaları aldı (1985+ kısmı aynı; eskilerde 4 Ocak 1985'te biten ilk hafta ve ilk 2 haftalık dönem, 31 Aralık 1984 eksik hesaplanmıştı).

## Özet (1985–2026, sektörler arası ortalama korelasyon)

| Frekans | Ham | CAPM artığı | FF3 artığı | Medyan piyasa betası (FF3) |
|---|---|---|---|---|
| Günlük | 0,543 | 0,046 | 0,021 | 0,99 |
| Haftalık | 0,536 | 0,059 | 0,029 | 1,00 |
| 2 haftalık | 0,533 | 0,059 | 0,030 | 1,02 |
| Aylık | 0,522 | 0,075 | 0,041 | 1,04 |

Haftalık FF3 betalarının tüm dönem medyanından örnekler: Gold mkt 0,39 (piyasaya en az duyarlı), Util 0,65, Food 0,66; Banks hml +0,76 ve Oil hml +0,70 (value); Softw hml −0,67 ve Chips hml −0,47 (growth).

## Doğrulama

- Haftalık FF3 artıkları ve betaları rastgele haftalarda scikit-learn ile bağımsız olarak yeniden hesaplandı, aynı çıktı.
- Gerçekleşen varyans, günlük artıklardan elle hesaplanan değerle aynı.
- `pytest`: birleştirme kuralları, 2 haftalık hizalama, Mkt-RF yöntemi ve look-ahead olmaması test ediliyor.

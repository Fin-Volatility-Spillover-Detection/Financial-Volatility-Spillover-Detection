# AI_CONTEXT: `data/processed/` veri sözlüğü

> Bu dosya, bu klasördeki CSV'leri kullanacak bir yapay zekâ (veya yeni bir ekip üyesi) için yazıldı.
> Amaç: her dosyanın **ne olduğunu, neyin sonucu olduğunu, neden üretildiğini ve nasıl kullanılması gerektiğini**
> tahmin gerektirmeden anlatmak. Bilgiler, dosyalar programla incelenerek yazıldı (28 Kasım 2026).

## 0. Önce bunları bil (özet)

1. **Ana girdi `artik_ff3_haftalik_1985.csv`.** Sektörler arası etki analizlerinin girdisi bu dosya. Ham getirilerle analiz yapma.
2. **Birim yüzde.** `0.45` = %0,45, `−2.1` = −%2,1. Oynaklık dosyaları **yüzde kare (%²)**.
3. **Satır = dönem, sütun = sektör.** İndeks `Date`. Etiket, o dönemin **son işlem günü**.
4. **Hiçbir dosyada eksik değer (NaN) yok.** Aynı frekanstaki 1985 dosyalarının tarihleri birebir hizalı.
5. **Dosya adındaki yıl, dosyanın başlangıç yılıdır.** `_1983` = ham veri (2 yıllık ısınma dahil), `_1985` = işlenmiş veri (analiz dönemi).
6. **Son satır kısmi bir dönemdir** (veri 30 Eylül 2026 Çarşamba bitiyor). Analizde at veya işaretle.
7. **CSV'leri elle düzenleme.** Hepsi `python scripts/00_build_processed_data.py` ile yeniden üretilir.

## 1. Proje bağlamı (bu veriler ne için var?)

GTU bilgisayar mühendisliği bitirme projesi. Soru: **"Hangi sektör, hangi sektördeki artışı veya düşüşü ne kadar etkiliyor?"**
Proje şu an **veri hazırlığı aşamasında**: bu klasördeki CSV'ler, sektörler arası etki analizlerine girdi olmak üzere hazırlandı.

- **Neden ham getiri değil de artık?** Piyasa bütün sektörleri birlikte hareket ettirir. Ham veride sektörler birbirini etkiliyormuş gibi görünür (ortalama korelasyon 0,54). FF3 artıkları bu ortak hareketi çıkarır (ortalama korelasyon 0,03). Geriye **sektöre özgü** hareket kalır; sektörler arası etkiler bunun üzerinden incelenir.

## 2. Veri kökeni ve akış

```
data/raw/49_sektor_gunluk_getiri_temiz.csv   (49 sektör, günlük, %, value-weighted, 1926–2026, Kenneth R. French)
data/raw/ff3_daily_clean.csv                 (Mkt-RF, SMB, HML, RF, günlük, %, Kenneth R. French)
        │
        │  scripts/00_build_processed_data.py
        │  (src/sektorgraf/data/aggregate.py, src/sektorgraf/data/residuals.py)
        ▼
 [1] Dönemlere birleştir (bileşik)  ──►  49_sektor_<frekans>_1983.csv,  ff3_<frekans>_1983.csv
        │
 [2] Kayan pencereli FF3 regresyonu  ──►  artik_ff3_<frekans>_1985.csv,  beta_ff3_<frekans>_1985.csv
        │
 [3] Günlük artıkların dönem içi kare toplamı  ──►  oynaklik_rv_ff3_<frekans>_1985.csv
```

`<frekans>` ∈ {`gunluk`, `haftalik`, `2haftalik`, `aylik`}. Oynaklık dosyası günlük için yok.

## 3. Ortak format

| Özellik | Değer |
|---|---|
| Ayraç / kodlama | Virgül, UTF-8 (BOM yok) |
| İlk sütun | `Date`, `YYYY-MM-DD`. `pd.read_csv(f, index_col="Date", parse_dates=True)` |
| Ondalık | 4 basamak |
| Sektör sütun sırası | Her dosyada aynı, aşağıdaki 49'luk liste sırasıyla |
| Eksik değer | Yok |
| Haftalık dönem | Pazartesi–Cuma. Etiket son işlem günü (Cuma tatilse Perşembe) |
| 2 haftalık dönem | İki ardışık hafta. 11 Ocak 1985'te biten çifte göre hizalı (…, 1984-12-28, 1985-01-11, 1985-01-25, …) |
| Aylık dönem | Takvim ayı. Etiket ayın son işlem günü |

### 49 sektör (sütun adları)

| # | Kod | Sektör (EN, French) | Türkçe |
|---|---|---|---|
| 1 | Agric | Agriculture | Tarım |
| 2 | Food | Food Products | Gıda ürünleri |
| 3 | Soda | Candy & Soda | Şekerleme ve meşrubat |
| 4 | Beer | Beer & Liquor | Bira ve alkollü içecek |
| 5 | Smoke | Tobacco Products | Tütün |
| 6 | Toys | Recreation | Eğlence ürünleri (oyuncak, tekne, müzik aleti) |
| 7 | Fun | Entertainment | Eğlence hizmetleri (film, spor) |
| 8 | Books | Printing and Publishing | Basım-yayın |
| 9 | Hshld | Consumer Goods | Tüketim malları |
| 10 | Clths | Apparel | Giyim |
| 11 | Hlth | Healthcare | Sağlık hizmetleri |
| 12 | MedEq | Medical Equipment | Tıbbi cihaz |
| 13 | Drugs | Pharmaceutical Products | İlaç |
| 14 | Chems | Chemicals | Kimya |
| 15 | Rubbr | Rubber and Plastic Products | Kauçuk ve plastik |
| 16 | Txtls | Textiles | Tekstil |
| 17 | BldMt | Construction Materials | İnşaat malzemeleri |
| 18 | Cnstr | Construction | İnşaat |
| 19 | Steel | Steel Works Etc | Çelik ve metal |
| 20 | FabPr | Fabricated Products | Metal ürünler |
| 21 | Mach | Machinery | Makine |
| 22 | ElcEq | Electrical Equipment | Elektrikli ekipman |
| 23 | Autos | Automobiles and Trucks | Otomotiv |
| 24 | Aero | Aircraft | Havacılık |
| 25 | Ships | Shipbuilding, Railroad Equipment | Gemi ve demiryolu ekipmanı |
| 26 | Guns | Defense | Savunma |
| 27 | Gold | Precious Metals | Değerli metaller |
| 28 | Mines | Non-Metallic and Industrial Metal Mining | Madencilik |
| 29 | Coal | Coal | Kömür |
| 30 | Oil | Petroleum and Natural Gas | Petrol ve doğalgaz |
| 31 | Util | Utilities | Elektrik, gaz, su |
| 32 | Telcm | Communication | Telekomünikasyon |
| 33 | PerSv | Personal Services | Kişisel hizmetler |
| 34 | BusSv | Business Services | İş hizmetleri |
| 35 | Hardw | Computers | Bilgisayar donanımı |
| 36 | Softw | Computer Software | Yazılım |
| 37 | Chips | Electronic Equipment | Elektronik / yarı iletken |
| 38 | LabEq | Measuring and Control Equipment | Ölçüm ve kontrol ekipmanı |
| 39 | Paper | Business Supplies | Kâğıt ve ofis malzemeleri |
| 40 | Boxes | Shipping Containers | Ambalaj |
| 41 | Trans | Transportation | Ulaştırma |
| 42 | Whlsl | Wholesale | Toptan ticaret |
| 43 | Rtail | Retail | Perakende |
| 44 | Meals | Restaurants, Hotels, Motels | Restoran ve otel |
| 45 | Banks | Banking | Bankacılık |
| 46 | Insur | Insurance | Sigorta |
| 47 | RlEst | Real Estate | Gayrimenkul |
| 48 | Fin | Trading | Finansal aracılık / yatırım |
| 49 | Other | Almost Nothing | Diğer |

SIC kodu eşlemesi: `Siccodes49.txt` (French Data Library).

## 4. Dosyalar

### 4.1 `49_sektor_<frekans>_1983.csv`: ham sektör getirileri

- **Ne:** 49 sektörün dönem getirisi (%). Formül uygulanmamış, sadece günlükten döneme **bileşik** olarak birleştirilmiş.
- **Neden 1983:** Artıklar için beta penceresinin ısınması (bkz. §5). Analiz dönemi 1985+.
- **Sütunlar:**
  - `gunluk`: 49 sütun (`Agric` … `Other`).
  - `haftalik`, `2haftalik`, `aylik`: 147 sütun. Her sektör için sırayla `<S>` (dönem getirisi), `<S>_min` (dönem içindeki en düşük **günlük** getiri), `<S>_max` (en yüksek günlük getiri).
- **Kullanım:** "FF3 olmadan ne olurdu?" karşılaştırması (`.loc["1985":]` ile kes). Min/max sütunları oynaklık veya uç hareket analizi için.
- **Kullanma:** Sektörler arası etki analizlerinin girdisi olarak. Ortak piyasa hareketi sahte bağlantılar üretir.

### 4.2 `ff3_<frekans>_1983.csv`: Fama–French faktörleri

- **Ne:** Formülün **malzemesi**, sektör verisi değil. Sütunlar: `Mkt-RF` (piyasanın risksiz faizi aşan getirisi), `SMB` (küçük − büyük şirket), `HML` (value − growth), `RF` (risksiz faiz, 1 aylık hazine bonosu), `n_gun` (dönemdeki işlem günü sayısı; günlükte hep 1).
- **Kullanım:** Artıkları yeniden hesaplamak, piyasa getirisini veya RF'yi dış değişken / kontrol olarak kullanmak.
- **Not:** Adında "ff3" geçmesi FF3 formülünün sonucu olduğu anlamına gelmez. Formülün sonucu `artik_ff3_*`.

### 4.3 `artik_ff3_<frekans>_1985.csv`: FF3 artıkları (ANA GİRDİ)

- **Ne:** Her sektörün, piyasa + küçük şirket + value/growth etkileri çıkarıldıktan sonra kalan **sektöre özgü** getirisi (%).
- **Formül:** `artik[t, S] = (R[t, S] − RF[t]) − (α[t, S] + β_mkt[t, S]·MktRF[t] + β_smb[t, S]·SMB[t] + β_hml[t, S]·HML[t])`
- **Sütunlar:** 49 sektör.
- **Kullanım:** Sektörler arası etki analizlerinin ana girdisi. Haftalık ana analiz, günlük sağlamlık kontrolü.
- **Ölçek (sektör standart sapmalarının medyanı):** günlük 0,84 · haftalık 1,89 · 2 haftalık 2,67 · aylık 4,07.

### 4.4 `beta_ff3_<frekans>_1985.csv`: kullanılan betalar

- **Ne:** `artik_ff3` dosyasının aynı satırını hesaplarken kullanılan katsayılar. 147 sütun: her sektör için `<S>_mkt`, `<S>_smb`, `<S>_hml`. (α kaydedilmez.)
- **Kullanım:** Beta zamanla nasıl değişiyor (ör. otomotivin piyasa duyarlılığı), sektör profili, artıkların denetimi. **Model girdisi değildir.**
- **Örnek medyanlar (haftalık):** Gold mkt 0,39 · Util 0,65 · Food 0,66 · Banks hml +0,76 · Oil hml +0,70 · Softw hml −0,67 · Chips hml −0,47.

### 4.5 `oynaklik_rv_ff3_<frekans>_1985.csv`: gerçekleşen varyans

- **Ne:** Dönem içindeki **günlük** FF3 artıklarının kareleri toplamı (%²). Sektöre özgü oynaklığın ölçüsü. 49 sektör + `n_gun`.
- **Kaynak:** `artik_ff3_gunluk` ile aynı yöntemle (252 günlük pencere) hesaplanan günlük artıklar.
- **Frekanslar:** haftalık, 2 haftalık, aylık (günlük yok).
- **Kullanım:** Sektöre özgü oynaklığın (yönden bağımsız "çalkantı") incelenmesi. Log dönüşümü (`np.log(rv)`) genelde önerilir.
- **Tipik değer (medyan):** haftalık 1,87 · 2 haftalık 4,24 · aylık 10,17.

## 5. Yöntem ayrıntıları

**Birleştirme:**
- Getiri: `R_dönem = [Π(1 + r_gün/100) − 1] × 100`.
- `Mkt-RF` doğrudan bileşiklenmez: `Mkt = Mkt-RF + RF` (günlük) bileşiklenir, `RF` bileşiklenir, `Mkt-RF_dönem = Mkt_dönem − RF_dönem`.
- `SMB`, `HML`: günlük değerleri bileşiklenir (uzun-kısa bacakları ayrı bileşiklemeye çok yakın bir yaklaşım).

**Kayan pencere regresyonu** (`rolling_factor_model`):
- Her sektör ve her dönem t için OLS: `(R − RF) ~ 1 + Mkt-RF + SMB + HML`. Tahmin penceresi **[t − pencere, t − 1]**: t dönemi pencereye dahil değil.
- Elde edilen α ve β'lar t dönemine uygulanır → `artik[t]`. Pencere her dönem bir adım kayar, betalar her dönem yeniden tahmin edilir.

| Frekans | Pencere | Pencerede en az gözlem | Isınma |
|---|---|---|---|
| gunluk | 252 gün | 200 | 1984 |
| haftalik | 104 hafta | 84 | 1983–1984 |
| 2haftalik | 52 dönem | 42 | 1983–1984 |
| aylik | 60 ay | 48 | 1980–1984 (1983 dosyasından eski ham veri kullanılır) |

## 6. Zaman kuralları (look-ahead)

- `beta[t]`: Sadece t'den **önceki** dönemlerden tahmin edilir (pencere [t − pencere, t − 1]).
- `artik[t]`: t döneminin kendi getirisi ve faktörleriyle hesaplanır, yani **t'nin sonunda bilinir.** Bu veriyle geleceğe dönük bir şey yapılacaksa, t+1 için sadece `≤ t` satırları kullanılmalıdır.
- Kriz tarihleri (2008, 2020 vb.) bu dosyalarda yok.
- **Ortak zaman ayrımı** (`configs/config.yaml → split`, bütün ekip için tek kaynak): eğitim 1985–2012 · doğrulama 2013–2018 · test 2019–2026. Zaman sırasına göre, karıştırmadan. Bir dönem, **bitiş tarihine (etiketine)** göre atanır; ör. 4 Ocak 2013'te biten hafta (31 Aralık 2012'yi de içerir) doğrulamaya düşer. Kod: `sektorgraf.data.splits.split`.

  | Frekans | Eğitim | Doğrulama | Test |
  |---|---|---|---|
  | gunluk | 7060 | 1510 | 1947 |
  | haftalik | 1461 | 313 | 404 |
  | 2haftalik | 730 | 157 | 202 |
  | aylik | 336 | 72 | 92 |

  (Kısmi son dönem atıldıktan sonra.)

## 7. Kullanım tarifleri

```python
import numpy as np
from sektorgraf.config import load_config
from sektorgraf.data.splits import load_processed, split

cfg = load_config()

artik = load_processed(cfg, "artik_ff3_haftalik_1985.csv")   # ana girdi; son kısmi hafta atılmış (2178 × 49)
parts = split(artik, cfg)                                     # ortak ayrım
train, val, test = parts["train"], parts["val"], parts["test"]

ham = load_processed(cfg, "49_sektor_haftalik_1983.csv").loc["1985":]
ham = ham[[c for c in ham.columns if not c.endswith(("_min", "_max"))]]   # sadece getiriler
assert ham.index.equals(artik.index)                                     # hizalı

rv = load_processed(cfg, "oynaklik_rv_ff3_haftalik_1985.csv")
n = rv.pop("n_gun")
log_rv = np.log(rv.mul(5 / n, axis=0).clip(lower=1e-4))     # kısa haftaları 5 güne ölçekle, log al

beta = load_processed(cfg, "beta_ff3_haftalik_1985.csv")
beta_mkt = beta.filter(like="_mkt").rename(columns=lambda c: c[:-4])     # dönem × 49
```

| Görev | Dosya |
|---|---|
| Sektörler arası etki analizi (ana girdi) | `artik_ff3_haftalik_1985.csv` |
| Günlük sağlamlık / hızlı yayılma (1–3 gün) | `artik_ff3_gunluk_1985.csv` |
| Frekans sağlamlığı | `artik_ff3_2haftalik_1985.csv`, `artik_ff3_aylik_1985.csv` (aylıkta sadece ~500 gözlem var) |
| Sektöre özgü oynaklık | `oynaklik_rv_ff3_haftalik_1985.csv` |
| Ham vs FF3 karşılaştırması | `49_sektor_haftalik_1983.csv` (1985+) |
| Piyasa getirisi / RF kontrolü | `ff3_haftalik_1983.csv` |
| Beta analizi | `beta_ff3_haftalik_1985.csv` |

## 8. Bilinen tuzaklar

- **Kısmi son dönem:** 2026-09-30 satırı her frekansta eksik bir dönem (haftalıkta 3 gün).
- **Kısa haftalar (haftalık, n_gun):** 1897 hafta 5 gün, 382 hafta 4 gün; 3 günlük: 2007-01-05, 2012-11-02 (Sandy kasırgası), 2026-09-30 (kısmi); 1 günlük: 2001-09-10 (11 Eylül sonrası borsa kapanışı). Bu haftalarda RV mekanik olarak küçük, 2001-09-10'da sıfıra çok yakın.
- **FF3 gerçek etkinin bir kısmını silebilir:** Piyasa endeksi sektörlerden oluşur. Büyük bir sektörün (ör. 2008'de bankalar) şoku piyasayı da hareket ettirir ve bir kısmı "piyasa" olarak çıkarılır. Bulunan yayılma etkileri **alt sınır** olarak yorumlanmalı.
- **Faiz faktörü yok:** FF3'te faiz faktörü bulunmadığı için faize duyarlı sektörler (Banks, RlEst, Util) arasında ortak faiz şokundan kalan bir birlikte hareket olabilir.
- **CAPM dosyaları üretilmiyor.** Gerekirse `python scripts/00_build_processed_data.py --capm` → `artik_capm_*`, `beta_capm_*`.

## 9. Yeniden üretim ve doğrulama

```bash
pip install -r requirements.txt
python scripts/00_build_processed_data.py         # ~15 sn, bütün dosyalar
python scripts/00_build_processed_data.py --capm  # + CAPM dosyaları
PYTHONPATH=src pytest                             # look-ahead, birleştirme, hizalama testleri
```

Doğrulamalar: rastgele haftalarda artık ve betalar scikit-learn ile bağımsız olarak yeniden hesaplandı (aynı); RV elle hesaplananla aynı; 1985+ ham haftalık veri ekibin önceki bağımsız birleştirmesiyle birebir aynı (tek fark: önceki versiyonda 4 Ocak 1985 haftası 31 Aralık 1984'ü içermiyordu).

## 10. Özet istatistikler (1985–2026)

| Frekans | Satır | Ham ort. korelasyon | FF3 artık ort. korelasyon | Medyan piyasa betası |
|---|---|---|---|---|
| gunluk | 10517 | 0,543 | 0,021 | 0,99 |
| haftalik | 2179 | 0,536 | 0,029 | 1,00 |
| 2haftalik | 1090 | 0,533 | 0,030 | 1,02 |
| aylik | 501 | 0,522 | 0,041 | 1,04 |

(Karşılaştırma için CAPM artıklarında ortalama korelasyon: günlük 0,046, haftalık 0,059, 2 haftalık 0,059, aylık 0,075.)

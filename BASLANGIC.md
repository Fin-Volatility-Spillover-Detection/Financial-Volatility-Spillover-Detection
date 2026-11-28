# BAŞLANGIÇ: Ortak proje bağlamı

> Bu dosya, ekibin (3 kişi) ortak hafızasıdır. Projeye katılan biri ya da bir yapay zekâ işe buradan başlar.
> **Sadece kesinleşmiş kararlar ve mevcut durum** yazılır. Tahmin, beklenti veya plan yazılmaz.
> Son güncelleme: 28 Kasım 2026.

---

## 1. Proje

- **Soru:** Hangi sektör, hangi sektördeki artışı veya düşüşü ne kadar etkiliyor?
- **Kapsam:** ABD'deki 49 sektör (Kenneth R. French sınıflandırması).
- **Şu anki durum:** Veri hazırlığı tamamlandı. Analiz aşaması başlamadı.

## 2. Temel kavramlar

- **Getiri:** Bir sektörün bir dönemdeki yüzde değişimi (temettüler dahil). Birim yüzde: `0.45` = %0,45.
- **FF3 (Fama–French 3 faktör):** Sektör getirisini üç ortak etkene ayıran model: piyasanın genel hareketi (`Mkt-RF`), küçük/büyük şirket farkı (`SMB`), value/growth farkı (`HML`).
- **Artık (FF3 artığı):** Sektör getirisinden bu üç ortak etkenin payı çıkarıldıktan sonra kalan, **sektöre özgü** hareket:
  `artık = (sektör getirisi − RF) − (α + β₁·Mkt-RF + β₂·SMB + β₃·HML)`
- **Beta (β):** Sektörün bir etkene duyarlılığı. Örneğin piyasa betası 1,2 ise piyasa %1 oynadığında sektör ortalama %1,2 oynar.
- **Neden artık?** Ham getirilerde sektörler piyasayla birlikte hareket ettiği için birbirini etkiliyormuş gibi görünür (haftalık ortalama korelasyon 0,54). FF3 artıklarında bu ortak hareket çıkarılır (ortalama korelasyon 0,03). Sektörler arası etki, sektöre özgü hareketler üzerinden incelenir.

## 3. Kesinleşmiş kararlar

| Konu | Karar |
|---|---|
| Veri kaynağı | Kenneth R. French Data Library. 49 sektör günlük getirisi (**value-weighted**, teyit edildi) ve FF3 faktörleri |
| Ana frekans | **Haftalık.** Günlük veri sağlamlık kontrolü içindir |
| Dönem | Ham veri **1983'ten**, işlenmiş veri (artık, beta, oynaklık) **1985'ten** başlar. Veri 30 Eylül 2026'da biter |
| Ön işleme | Bütün analizlerin girdisi **FF3 artıklarıdır**, ham getiri değil |
| Betalar | **Kayan pencereyle, sadece geçmişten** hesaplanır: haftalık 104 hafta, 2 haftalık 52 dönem, aylık 60 ay, günlük 252 gün |
| CAPM | Varsayılan olarak üretilmez. Gerekirse `python scripts/00_build_processed_data.py --capm` |
| Ortak zaman ayrımı | **Eğitim 1985–2012 · Doğrulama 2013–2018 · Test 2019–2026.** Tek kaynak `configs/config.yaml → split`. Herkes `sektorgraf.data.splits` üzerinden kullanır, tarih elle yazılmaz (Tarihler daha sonra değiştirilebilir) |
| Ayar seçimi | Ayarlar (gecikme sayısı, ceza katsayısı vb.) **sadece doğrulama döneminde** seçilir. Test dönemine ayar seçerken bakılmaz |
| Kullanılacak yöntemler | **Zero, AR, VAR** (bkz. bölüm 6) ileride deep learning modellerine baseline olması için kullanılacaklar. |
| Kısmi son dönem | Haftalık, 2 haftalık ve aylık dosyalarda son satır (2026-09-30) eksik bir dönemdir, atılır (`load_processed` otomatik yapar) |

## 4. Veriler

Bütün CSV'ler `data/` altındadır. İlk sütun `Date` (YYYY-AA-GG). Sektör sütunlarının sırası her dosyada aynıdır (49 sektör: `Agric`, `Food`, … `Fin`, `Other`; Türkçe karşılıkları `data/processed/AI_CONTEXT.md` içinde).

### 4.1 Ham veri (`data/raw/`): kaynaktan geldiği gibi, değiştirilmez

| Dosya | İçerik | Neden var | Nasıl kullanılır |
|---|---|---|---|
| `49_sektor_gunluk_getiri_temiz.csv` | 49 sektörün günlük getirisi (%), 1926–2026 | Bütün işlenmiş verinin kaynağı | Doğrudan analizde kullanılmaz. `00_build_processed_data.py` bunu okuyup işlenmiş dosyaları üretir |
| `ff3_daily_clean.csv` | Günlük `Mkt-RF`, `SMB`, `HML`, `RF` (%), 1926–2026 | FF3 artıklarını hesaplamak için gereken faktörler | Doğrudan analizde kullanılmaz. İşlenmiş dosyaların kaynağı |

### 4.2 İşlenmiş veri (`data/processed/`): `python scripts/00_build_processed_data.py` ile üretilir

Her dosya grubu dört frekansta vardır (`gunluk`, `haftalik`, `2haftalik`, `aylik`), oynaklık hariç (günlük yok). Toplam 19 CSV.

| Dosya | Ne | Neden üretildi | Nasıl kullanılır |
|---|---|---|---|
| **`artik_ff3_<frekans>_1985.csv`** | 49 sektörün FF3 artıkları (%) | Sektörler arası etkiyi piyasanın ortak hareketinden ayırmak için | **Ana girdi.** Zero, AR ve VAR bu dosyalarla çalışır. Ana analiz `artik_ff3_haftalik_1985.csv`, günlük dosya sağlamlık kontrolü içindir |
| `49_sektor_<frekans>_1983.csv` | Ham sektör getirileri, günlükten bileşik olarak birleştirilmiş. Haftalık/2 haftalık/aylıkta her sektör için `<sektör>`, `<sektör>_min`, `<sektör>_max` (dönem içindeki en düşük / en yüksek günlük getiri) | Artıkların hesaplandığı ara veri; ayrıca ham–artık karşılaştırması için | Model girdisi değildir. "FF3 olmadan ne olurdu?" karşılaştırmasında `.loc["1985":]` ile kullanılır |
| `ff3_<frekans>_1983.csv` | `Mkt-RF`, `SMB`, `HML`, `RF` aynı dönemlere birleştirilmiş + `n_gun` (dönemdeki işlem günü sayısı) | Artıkların hesaplandığı faktörler | Model girdisi değildir. Faktör değerlerine veya RF'ye ihtiyaç olduğunda kullanılır |
| `beta_ff3_<frekans>_1985.csv` | Her artığı hesaplarken kullanılan betalar: `<sektör>_mkt`, `<sektör>_smb`, `<sektör>_hml` | Artık hesabının şeffaflığı ve denetimi için | Model girdisi değildir. Betaların zaman içindeki değişimini incelemek için |
| `oynaklik_rv_ff3_<frekans>_1985.csv` | Sektöre özgü gerçekleşen varyans (%²): dönem içindeki günlük FF3 artıklarının kareleri toplamı + `n_gun` | Sektöre özgü oynaklığın (yönden bağımsız "çalkantı") ölçüsü | Zero/AR/VAR getiri modellerinde kullanılmaz. Oynaklık incelemesi için mevcuttur |

**Satır sayıları (haftalık, kısmi son hafta atıldıktan sonra):** eğitim 1.461 · doğrulama 313 · test 404 hafta.

## 5. Ortak kurallar

1. **Look-ahead yok.** t döneminin betası sadece [t − pencere, t − 1] dönemlerinden gelir. Bir modelin t+1 tahmini sadece t ve öncesinin **gerçek** değerlerini kullanır.
2. **Dönem etiketi:** Haftalar Cuma kapanışında biter. Dönemin tarihi, son işlem günüdür (Cuma tatilse Perşembe). Haftalık getiri, önceki Cuma kapanışından bu Cuma kapanışına olan değişimdir.
3. **Birleştirme bileşiktir:** `[(1 + r₁/100) × … × (1 + rₙ/100) − 1] × 100`.
4. **Bir dönem, bitiş tarihine göre** eğitim/doğrulama/test bölümüne atanır.
5. **CSV'ler elle düzenlenmez.** Değişiklik gerekirse kod güncellenip betik yeniden çalıştırılır.
6. **Ayrım dosyalara bölünmez.** Okurken bellekte yapılır:

```python
from sektorgraf.config import load_config
from sektorgraf.data.splits import load_processed, split

cfg = load_config()
artik = load_processed(cfg, "artik_ff3_haftalik_1985.csv")   # son kısmi hafta atılmış
train, val, test = split(artik, cfg).values()
```

## 6. Yöntemler: Zero, AR, VAR

### 6.1 Ortak düzen (üç yöntem için de aynı)

- **Girdi:** `artik_ff3_haftalik_1985.csv` (ana). Aynı uygulama günlük dosyayla sağlamlık kontrolü olarak tekrarlanır.
- **Hedef:** Her sektörün bir sonraki dönemdeki FF3 artığı, `r[i, t+1]`.
- **Bir adım ileri tahmin:** Her t için t+1 tahmin edilir, sonra bir dönem ilerlenir. Tahmin her zaman o ana kadar gerçekleşmiş **gerçek** değerlerle yapılır; model kendi tahminini girdi olarak kullanmaz.
- **Eğitim:** Katsayılar eğitim döneminde tahmin edilir ve tahmin boyunca sabit kalır.
- **Ayar seçimi:** Gecikme sayısı (p) ve VAR'ın ceza katsayısı doğrulama döneminde  en iyi sonucu veren değere göre seçilir.
- **Değerlendirme:** Seçilen ayarlarla model test döneminde bir kez değerlendirilir.

### 6.2 Zero

- **Ne:** Her sektör için her dönem `r̂[i, t+1] = 0` tahmini.
- **Neden:** En alt çıta. Artıkların ortalaması sıfıra çok yakın olduğu için "hep sıfır" tahmini anlamlı bir referanstır. Bir modelin bu veriden bir şey öğrenip öğrenmediği, Zero'yu geçip geçmediğiyle anlaşılır.
- **Uygulama:** Eğitim gerektirmez. Her tabloda bir satır olarak yer alır.

### 6.3 AR (otoregresif model)

- **Ne:** Her sektör **sadece kendi geçmişiyle** tahmin edilir:
  `r[i, t+1] = c_i + φ_i1 · r[i, t] + … + φ_ip · r[i, t−p+1] + ε`
- **Neden:** Sektörün kendi geçmişinin ne kadar bilgi taşıdığını ölçer. Diğer sektörlere bakmayan, "sektörler arası etki yok" referansıdır.
- **Uygulama:** 49 sektörün her biri için ayrı bir doğrusal regresyon (en küçük kareler). Katsayılar eğitim döneminden.

### 6.4 VAR (vektör otoregresif model)

- **Ne:** Her sektör **bütün sektörlerin geçmişiyle** tahmin edilir. 49 sektör birlikte modellenir:
  `r[t+1] = c + A₁ · r[t] + … + A_p · r[t−p+1] + ε`
  Burada `r[t]` 49 sektörün vektörü, `A_k` 49×49'luk katsayı matrisidir. `A_k[i, j]`, j sektörünün k dönem önceki değerinin i sektörüne etkisidir.
- **Neden:** Diğer sektörlere bakmanın tahmini iyileştirip iyileştirmediğini ölçer. AR ile tek farkı, diğer sektörlerin geçmişini de kullanmasıdır. Katsayı matrisi, sektörler arası doğrusal etkileri doğrudan gösterir.
- **Uygulama:** Her sektör için ayrı bir denklem. Her gecikmede 49 × 49 = 2.401 katsayı olduğundan katsayılar **Lasso** (düzenlileştirilmiş regresyon) ile tahmin edilir: önemsiz katsayılar sıfıra çekilir. Ceza katsayısı doğrulama döneminde seçilir.

### 6.5 Karşılaştırmalar ve metrikler

| Karşılaştırma | Cevapladığı soru |
|---|---|
| AR ile Zero | Sektörün kendi geçmişi tahmine yardım ediyor mu? |
| VAR ile AR | Diğer sektörlere bakmak, sadece kendi geçmişine bakmaktan daha iyi mi? |
| VAR ile Zero | VAR veriden işe yarar bir şey öğreniyor mu? |

- **Metrikler:** RMSE (ortalama hata), örneklem dışı R² (Zero'ya göre: `1 − SSE_model / SSE_Zero`), yön isabeti (artı/eksi işaretini doğru bilme oranı; %50 yazı tura).
- **Farkın anlamlılığı:** Diebold–Mariano testi (iki modelin dönemlik hata farkı şansla açıklanabilir mi?).
- Bütün ekip aynı ayrımı, aynı hedefi ve aynı metrikleri kullanır.

## 7. Bilinen sınırlılıklar

- FF3, büyük sektörlerden yayılan etkinin bir kısmını "piyasa" olarak çıkarabilir (ör. 2008'de bankalar). Bu veriden bulunan sektörler arası etkiler alt sınır olarak yorumlanır.
- FF3'te faiz faktörü yoktur. Faize duyarlı sektörler (Banks, RlEst, Util) arasında ortak faiz şokundan kalan birlikte hareket olabilir.
- Artık mutlak bir hareket değil, göreli bir sapmadır: "sektör X yükseldi" değil, "sektör X'e özgü bir şok" olarak yorumlanır.

## 8. Repo ve kurulum

```
configs/config.yaml                 Veri yolları, başlangıç tarihleri, beta pencereleri, ortak ayrım
data/raw/                           Kaynak veri (sadece okunur)
data/processed/                     İşlenmiş CSV'ler + README.md + AI_CONTEXT.md (ayrıntılı veri sözlüğü)
scripts/00_build_processed_data.py  Bütün işlenmiş veriyi üreten betik
src/sektorgraf/                     Veri hazırlığı kodu (birleştirme, FF3 artıkları, ortak ayrım)
tests/                              Veri ve ayrım testleri
docs/kararlar.md                    Veri kararları ve gerekçeleri
```

```bash
pip install -r requirements.txt
pip install -e .
pytest                                       # 10 test
python scripts/00_build_processed_data.py    # veriyi yeniden üretmek gerekirse
```

**Ayrıntı için:** `data/processed/AI_CONTEXT.md` (her dosyanın şeması, formüller, tuzaklar), `docs/kararlar.md` (kararların gerekçeleri).

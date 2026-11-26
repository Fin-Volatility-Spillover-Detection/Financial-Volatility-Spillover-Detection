# Sektörler Arası Etki Grafı

GTU bitirme projesi. **Soru:** Hangi sektör, hangi sektördeki artışı veya düşüşü ne kadar etkiliyor?

**Çıktılar:**
1. **Etki tablosu (ana sonuç):** Diebold–Yılmaz ile 49×49 tablo. "j sektörü, i sektörünün belirsizliğinin %X'ini açıklıyor."
2. **Doğrulama:** Bu tablo, Graph WaveNet'e graf olarak verildiğinde tahmin, grafsız ve rastgele graflı versiyonlardan daha iyi oluyor mu?

## Kurulum

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .          # src/sektorgraf paketini her yerden import edilebilir yapar
pytest                    # 8 test: look-ahead yok, metrikler ve graf yöntemleri doğru
```

## Çalıştırma sırası

```bash
python scripts/01_make_residuals.py        # ham getiri → FF3 ve CAPM artıkları  (~3 sn)
python scripts/02_build_graphs.py          # Granger + DY etki tabloları          (~2 sn)
python scripts/02_build_graphs.py --rolling   # + kayan pencereli DY (kriz analizi, yavaş)
python scripts/03_run_baselines.py         # Zero, AR, Lasso-VAR + değerlendirme
python scripts/04_run_gwn.py               # Aşama 2: Graph WaveNet (henüz yok)
```

## Klasör yapısı

```
configs/config.yaml        Bütün tarihler, pencereler, parametreler. Kodda sabit sayı yok.
data/raw/                  French verileri (sadece okunur)
data/external/             Viyana olay verisi, ADS endeksi (planlanan)
data/processed/            Üretilen artık tabloları (git'e girmez)
src/sektorgraf/
  data/      io.py         Veri yükleme
             residuals.py  Kayan pencereli CAPM / FF3 artıkları (look-ahead yok)
             splits.py     Eğitim / doğrulama / test ayrımı
  graphs/    granger.py    İkili Granger + FDR → öncülük grafı
             dy.py         Lasso-VAR + genelleştirilmiş FEVD → etki tablosu, bağlantılılık
             utils.py      top-k seyreltme, rastgele kontrol grafı, graf benzerliği
  models/    base.py       Ortak model arayüzü (fit / predict)
             baselines.py  Zero, AR, Lasso-VAR(X)
             windows.py    Derin öğrenme için kayan pencere verisi
             gwn.py        Graph WaveNet (Aşama 2, taslak)
  evaluation/evaluate.py   ORTAK DEĞERLENDİRME: herkes bunu kullanır
             metrics.py    RMSE, R²_oos, yön isabeti, Diebold–Mariano
             periods.py    Sakin / kriz ayrımı
scripts/                   Numaralı adımlar
notebooks/                 Keşif ve grafikler (sonuçlar scripts/ üzerinden üretilir)
results/                   Tablolar, graflar, tahminler, figürler (git'e girmez)
docs/kararlar.md           Tasarım kararları ve gerekçeleri
tests/                     Kritik garantilerin testleri
```

## Model merdiveni

| # | Model | Neye bakıyor | Durum |
|---|---|---|---|
| 0 | Zero | — | ✓ |
| 1 | AR | Kendi geçmişi, doğrusal | ✓ |
| 2 | Lasso-VAR(X) | Bütün sektörler, doğrusal | ✓ |
| 3 | Graph WaveNet `none` | Kendi geçmişi, derin öğrenme | Aşama 2 |
| 4 | Graph WaveNet `random` | Rastgele graf (kontrol) | Aşama 2 |
| 5 | Graph WaveNet `dy` | DY etki tablosu | Aşama 2 |
| 6 | Graph WaveNet `adaptive` | Kendi öğrendiği graf | Aşama 2 |

Dış parametreli versiyonlar (Viyana olayları + ADS): Lasso-VARX, GWN `dy`, GWN `adaptive`.

## Değişmez kurallar

1. **Graf kuralı:** `W[i, j]` = j'nin i'ye etkisi (**j → i**). Satır = etkilenen, sütun = etkileyen.
2. **Look-ahead yok:** d günü için tahmin sadece d'den önceki gerçek değerleri kullanır. Betalar sadece geçmişten.
3. **Test dönemine dokunulmaz:** Ayar seçimi `val` döneminde. Modele verilen DY tablosu `dy_train.csv` (sadece eğitim dönemi). `dy_full.csv` sadece tarif/rapor içindir.
4. **Herkes `evaluate()` kullanır:** Kendi metrik kodunu yazmak yok, böylece sonuçlar karşılaştırılabilir.
5. **Birim yüzde:** `0.45` = %0,45. Bileşik getiri hesaplarken `(1 + r/100)`.
6. **Terminoloji:** "Nedensellik" değil, "öncülük" / "yönlü bağlantı".

## Yeni bir model eklemek

```python
from sektorgraf.models.base import Model

class BenimModelim(Model):
    name = "BenimModelim"
    def fit(self, train, exog=None): ...
    def predict(self, data, dates, exog=None): ...   # index=dates, sütunlar=sektörler

model = BenimModelim().fit(train)
evaluate(model.predict(resid, test_dates), resid, cfg, name=model.name)
compare("BenimModelim", "AR(1)", resid, cfg)    # Diebold–Mariano
```

## İlk sonuçlar (test: 2020-01 – 2026-09)

| Model | RMSE | R²_oos | Yön isabeti |
|---|---|---|---|
| Zero | 1.2154 | 0 | — |
| AR(1) | 1.2159 | −0.0008 | 0.499 |
| Lasso-VAR(1), α=0.01 | 1.2198 | −0.0072 | 0.499 |

Getiride doğrusal modeller sıfır tahminini geçemiyor. Beklenen bir sonuç: günlük artık getiri neredeyse tahmin edilemez. Lasso alfası henüz ayarlanmadı. Bkz. `docs/kararlar.md`.

# Sektörler Arası Etki Analizi: Veri Hazırlığı

> Projeye yeni başlıyorsanız önce [`BASLANGIC.md`](BASLANGIC.md) dosyasını okuyun: ortak bağlam, veriler, kesinleşmiş kararlar.

GTU bitirme projesi. **Soru:** Hangi sektör, hangi sektördeki artışı veya düşüşü ne kadar etkiliyor?

**Şu anki durum:** Veri hazırlığı tamamlandı. 49 ABD sektörünün getirileri günlük, haftalık, 2 haftalık ve aylık dönemlere birleştirildi. Piyasa etkisi Fama–French 3 faktör (FF3) modeliyle çıkarıldı. Analiz aşaması henüz başlamadı.

## Kurulum

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pytest                                              # 10 test: birleştirme, look-ahead yok, beta doğruluğu, ayrım
```

## Veriyi üretmek

```bash
python scripts/00_build_processed_data.py           # ~15 saniye, data/processed/ altındaki bütün CSV'ler
python scripts/00_build_processed_data.py --capm    # + CAPM artıkları (isteğe bağlı)
```

## Klasör yapısı

```
configs/config.yaml                 Veri yolları, başlangıç tarihleri, beta pencereleri
data/raw/                           Kaynak veri: French Data Library (sadece okunur)
data/processed/                     İşlenmiş CSV'ler
  README.md                         Dosyaların kısa açıklaması
  AI_CONTEXT.md                     Ayrıntılı veri sözlüğü (yapay zekâ araçları ve yeni ekip üyeleri için)
scripts/00_build_processed_data.py  Bütün işlenmiş veriyi üreten betik
src/sektorgraf/
  config.py                         config.yaml okuyucu
  data/io.py                        Ham veri yükleme
  data/aggregate.py                 Günlük → haftalık / 2 haftalık / aylık (bileşik)
  data/residuals.py                 Kayan pencereli FF3 / CAPM regresyonu ve artıklar
  data/splits.py                    Ortak veri okuma ve eğitim / doğrulama / test ayrımı
tests/                              Veri hazırlığı ve ayrım testleri
docs/kararlar.md                    Veriyle ilgili kararlar ve gerekçeleri
```

## Ana veri

`data/processed/artik_ff3_haftalik_1985.csv`: 49 sektörün haftalık, piyasa etkisi çıkarılmış (FF3) getirileri, 1985–2026. Bütün dosyalar ve kurallar için `data/processed/AI_CONTEXT.md`.

## Ortak zaman ayrımı

Bütün ekip aynı ayrımı kullanır (`configs/config.yaml → split`): **eğitim 1985–2012 · doğrulama 2013–2018 · test 2019–2026.**

```python
from sektorgraf.config import load_config
from sektorgraf.data.splits import load_processed, split

cfg = load_config()
artik = load_processed(cfg, "artik_ff3_haftalik_1985.csv")   # son kısmi hafta atılmış
train, val, test = split(artik, cfg).values()
```

## Kurallar

- **Birim yüzde:** `0.45` = %0,45. Bileşik getiri hesaplarken `(1 + r/100)`.
- **Look-ahead yok:** Her dönemin betası sadece önceki dönemlerden tahmin edilir.
- **CSV'leri elle düzenlemeyin.** Değişiklik gerekiyorsa kodu güncelleyip betiği yeniden çalıştırın.

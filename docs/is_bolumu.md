# İş bölümü: Zero, AR, VAR, LSTM

> 3 kişilik ekip için. Her parça diğerlerinden **bağımsız** yürür: herkes aynı veriyi aynı şekilde okur,
> tahminlerini aynı formatta bir dosyaya yazar, karşılaştırmayı tek bir ortak değerlendirme kodu yapar.
> Ortak bağlam için önce `BASLANGIC.md`.

| Kişi | İş | Sorumlu |
|---|---|---|
| **A** | Zero + AR + **ortak değerlendirme kodu** | ________ |
| **B** | VAR (Lasso) | ________ |
| **C** | LSTM | ________ |

---

## 1. Ortak sözleşme (herkes buna uyar)

Bu bölüm, işleri birbirinden bağımsız yapan kurallardır. Değiştirmek gerekirse önce ekipte konuşulur.

### 1.1 Girdi

```python
from sektorgraf.config import load_config
from sektorgraf.data.splits import load_processed, split

cfg = load_config()
artik = load_processed(cfg, "artik_ff3_haftalik_1985.csv")   # 2178 hafta × 49 sektör, son kısmi hafta atılmış
train, val, test = split(artik, cfg).values()                 # 1461 / 313 / 404 hafta
```

- Ana girdi **haftalık FF3 artıkları**. Başka dosya veya tarih elle yazılmaz.
- Sağlamlık için aynı kod `artik_ff3_gunluk_1985.csv` ile tekrar çalıştırılır (sadece dosya adı değişir).

### 1.2 Hedef ve tahmin yöntemi

- **Hedef:** Her sektörün bir sonraki haftaki artığı `r[i, t+1]`.
- **Bir adım ileri tahmin:** t+1 tahmini sadece t ve öncesinin **gerçek** değerleriyle yapılır. Model kendi tahminini girdi olarak kullanmaz.
- **Eğitim:** Model sadece eğitim döneminde (1985–2012) eğitilir, katsayılar/ağırlıklar sonra sabit kalır.
- **Ayar seçimi:** Ayarlar doğrulama döneminde (2013–2018) seçilir. Test dönemine (2019–2026) ayar seçerken **bakılmaz**.
- **Test:** Seçilen ayarlarla model test döneminde bir kez çalıştırılır.
- Doğrulama döneminin ilk haftasını tahmin ederken eğitim döneminin son haftaları girdi olarak kullanılabilir (geçmiş veri, look-ahead değil). Test için de aynı şekilde doğrulama dönemi geçmiş sayılır.
- Ölçekleme (standartlaştırma) yapılacaksa ortalama ve standart sapma **sadece eğitim döneminden** hesaplanır.

### 1.3 Çıktı formatı (en önemli kural)

Her model iki tahmin dosyası yazar:

```
results/haftalik/predictions/<model>_val.csv     doğrulama dönemi tahminleri (313 satır)
results/haftalik/predictions/<model>_test.csv    test dönemi tahminleri (404 satır)
results/haftalik/params/<model>.json             seçilen ayarlar ve doğrulama skoru
```

- `<model>`: `zero`, `ar`, `var`, `lstm`.
- **Format:** İlk sütun `Date` (tahmin edilen haftanın tarihi, artık dosyasındaki etiketle aynı), sonra 49 sektör sütunu **artık dosyasıyla aynı sırada**. Değerler yüzde. Eksik değer yok.
- Satır `t`'deki değer, `t` haftasının tahminidir (`t−1` ve öncesinin verisiyle yapılmış).
- Günlük sağlamlık versiyonu aynı yapıyla `results/gunluk/...` altına.
- Tahmin dosyaları küçük olduğu için **repoya commit edilir.** Böylece değerlendirme, başkasının kodunu çalıştırmadan yapılabilir.

### 1.4 Kod düzeni ve git

```
src/sektorgraf/models/<model>.py        model kodu (her kişi sadece kendi dosyası)
scripts/run_<model>.py                  eğitim + ayar seçimi + tahmin dosyalarını yazma
src/sektorgraf/evaluation/metrics.py    ortak metrikler (Kişi A)
scripts/evaluate.py                     bütün tahmin dosyalarını okuyup karşılaştırma tablosu (Kişi A)
```

- Herkes **kendi dalında** çalışır, sadece kendi dosyalarına dokunur, `main`'e pull request ile birleştirir.
- `config.yaml`'a ayar eklenecekse kendi bölümüne eklenir (ör. `models: ar: ...`), başkasının bölümü değiştirilmez.
- `data/processed/` altındaki CSV'lere dokunulmaz.

---

## 2. Kişi A: Zero + AR + ortak değerlendirme

### 2.1 Ortak değerlendirme kodu (ilk iş)

Diğer iki kişi tahmin dosyalarını yazmaya başlamadan önce hazır olması iyi olur, ama onları bekletmez: format 1.3'te sabit.

- `metrics.py`:
  - **RMSE:** `sqrt(ortalama((tahmin − gerçek)²))`, bütün haftalar ve sektörler üzerinden.
  - **Örneklem dışı R² (Zero'ya göre):** `1 − SSE_model / SSE_zero`.
  - **Yön isabeti:** İşareti doğru bilme oranı (gerçek değeri 0 olanlar hariç; hep 0 tahmin eden model için tanımsız).
  - **Diebold–Mariano testi:** Her hafta için iki modelin hata kareleri 49 sektör üzerinden ortalanır, haftalık fark serisi üzerinde test (Harvey–Leybourne–Newbold düzeltmesiyle). Çıktı: istatistik ve p-değeri.
  - Sektör bazında RMSE.
- `evaluate.py`: `results/haftalik/predictions/*_test.csv` dosyalarını okur, gerçek değerlerle karşılaştırır, tek bir tablo yazar (`results/haftalik/tables/karsilastirma.csv`): her model için RMSE, R², yön isabeti. İkili DM testleri: AR–Zero, VAR–AR, VAR–Zero, LSTM–AR, LSTM–Zero.
- Testler: bilinen değerlerle küçük örneklerde metriklerin doğru çıktığı (ör. tahmin = gerçek ise RMSE 0, R² 1).

### 2.2 Zero

- Her hafta, her sektör için `0` tahmini. Eğitim yok, ayar yok.
- `zero_val.csv` ve `zero_test.csv` yazılır.

### 2.3 AR

- Her sektör için ayrı doğrusal regresyon (en küçük kareler): `r[i, t+1] = c_i + φ_i1·r[i, t] + … + φ_ip·r[i, t−p+1]`.
- **Ayar:** gecikme sayısı `p ∈ {1, 2, 3, 4}`. Her p için eğitimde fit, doğrulamada RMSE; en düşük olan seçilir. (Tüm sektörler için tek p.)
- Seçilen p ile test tahminleri. `params/ar.json`: seçilen p, her p'nin doğrulama RMSE'si.

---

## 3. Kişi B: VAR (Lasso)

- 49 sektörün hepsinin geçmişiyle: `r[t+1] = c + A₁·r[t] + … + A_p·r[t−p+1]`. Her sektör için ayrı bir denklem, girdi 49·p gecikmeli değer.
- **Kestirim:** Lasso (scikit-learn `Lasso`), her denklem ayrı. Önemsiz katsayılar sıfıra çekilir.
- **Ayarlar:**
  - Gecikme `p ∈ {1, 2}`.
  - Ceza katsayısı `α`: logaritmik ızgara, ör. `{0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3}`.
  - Her (p, α) için eğitimde fit, doğrulamada RMSE; en düşük olan seçilir. Bütün denklemlerde aynı α (ilk sürüm).
  - Izgaranın kenarında bir değer seçilirse ızgara o yöne genişletilir.
- Seçilen ayarlarla test tahminleri.
- **Ek çıktı:** Seçilen modelin 1. gecikme katsayı matrisi `results/haftalik/params/var_A1.csv` (49×49, satır = etkilenen sektör i, sütun = etkileyen sektör j). Kaç katsayının sıfırdan farklı kaldığı `params/var.json`'a yazılır.
- `params/var.json`: seçilen p ve α, ızgaradaki bütün doğrulama RMSE'leri, sıfırdan farklı katsayı sayısı.

---

## 4. Kişi C: LSTM

- **Yapı (ilk sürüm):** Bütün sektörler için **ortak tek bir LSTM** (ağırlıklar paylaşılır). Her sektör için girdi, o sektörün **kendi** son `L` haftalık artığı; çıktı, bir sonraki haftanın artığı. Yani AR'nin doğrusal olmayan karşılığı: her sektör sadece kendi geçmişine bakar, ama model 49 sektörün kalıplarından birlikte öğrenir.
- **Veri:** Eğitim örnekleri `(sektör, hafta)` çiftleri: girdi `r[i, t−L+1 … t]`, hedef `r[i, t+1]`. Eğitim dönemi içinden (~1461 × 49 örnek).
- **Ölçekleme:** Eğitim döneminin ortalaması ve standart sapmasıyla. Tahminler yüzdeye geri çevrilerek kaydedilir.
- **Eğitim:** PyTorch. Kayıp fonksiyonu MSE. **Erken durdurma doğrulama kaybına göre.** Rastgele tohum sabitlenir.
- **Ayarlar** (doğrulama RMSE'sine göre):
  - Pencere uzunluğu `L ∈ {4, 8, 12}`
  - Gizli katman boyutu `∈ {16, 32, 64}`
  - Öğrenme hızı ve dropout makul varsayılanlarla başlar, gerekirse ayarlanır.
- **Tekrarlanabilirlik:** Seçilen ayarlarla model 5 farklı tohumla eğitilir; tahminlerin ortalaması kaydedilir, tohumlar arası RMSE yayılımı `params/lstm.json`'a yazılır.
- Seçilen ayarlarla test tahminleri.
- `params/lstm.json`: seçilen ayarlar, ızgaradaki doğrulama RMSE'leri, tohum sonuçları, eğitim süresi.

---

## 5. Teslim kontrol listesi (her kişi için)

- [ ] `results/haftalik/predictions/<model>_val.csv` ve `<model>_test.csv`, format 1.3'e uygun (Date + 49 sektör, sıra aynı, eksik değer yok)
- [ ] `results/haftalik/params/<model>.json`
- [ ] Ayar seçiminde test dönemine bakılmadı; ölçekleme sadece eğitim döneminden
- [ ] Kod kendi dosyalarında, `main`'e pull request ile
- [ ] Aynı kod günlük dosyayla çalıştırıldı (`results/gunluk/...`)
- [ ] Kısa not (pull request açıklamasında): ne yapıldı, seçilen ayarlar, doğrulama RMSE'si

## 6. Birleştirme

Bütün tahmin dosyaları `main`'e girdiğinde Kişi A `python scripts/evaluate.py` ile karşılaştırma tablosunu üretir:

| Model | RMSE | R² (Zero'ya göre) | Yön isabeti |
|---|---|---|---|
| Zero | | 0 | — |
| AR | | | |
| VAR | | | |
| LSTM | | | |

ve ikili Diebold–Mariano testleri (AR–Zero, VAR–AR, VAR–Zero, LSTM–AR, LSTM–Zero).

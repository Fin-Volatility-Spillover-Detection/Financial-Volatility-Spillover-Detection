# Karar kayıtları

Tasarım kararları ve gerekçeleri. Jüri "neden?" diye sorduğunda cevap burada.

1. **Girdi FF3 artıkları, ham getiri değil.** Ham veride kenarların ~%93'ü piyasa hareketi ve eş zamansız işlemden kaynaklı sahte bağlantı (868 → 58 kenar, CAPM simülasyonu). CAPM sağlamlık kontrolü.
2. **Betalar kayan pencereyle (252 gün), sadece geçmişten.** Look-ahead yok; testi `tests/test_core.py`.
3. **Ana çıktı: Diebold–Yılmaz etki tablosu.** "Ne kadar etkiliyor" sorusunun yüzdeli cevabı; aynı gün yayılmayı da içeriyor. Granger sadece gecikmeli "öncülük" alt kümesi.
4. **Önce getiri grafı, sonra karar noktası (Ocak ortası).** Graf zayıfsa oynaklık grafına geçilir. Kriterler: sentetik veride geri bulma, bootstrap kararlılığı, tahmine katkı.
5. **Tahmin testi doğrulama aracıdır.** Etki tablosunun gerçek ve kullanışlı olduğunu gösterir.
6. **Modele verilen DY tablosu SADECE eğitim döneminden.** `dy_full` tarif içindir, tahminde kullanılmaz.
7. **Tek derin öğrenme mimarisi: Graph WaveNet**, dört graf ayarı (none / random / dy / adaptive). Grafın etkisi aynı model içinde izole edilir.
8. **Krizler eğitimden çıkarılmaz**, sonuçlar sakin / kriz diye ayrı raporlanır. Kriz tarihleri hiçbir modelin girdisi değil.
9. **Terminoloji:** "nedensellik" yerine "öncülük" / "yönlü bağlantı".
11. **(Yerini #15 aldı) Eğitim dönemi 2000–2017.** FF3 artıklarıyla 1 gecikmeli Granger + FDR %5: 2010–2017'de 0 kenar, 2000–2017'de 126, sadece 2020'de 122. Gecikmeli yayılma krizlerde yoğunlaşıyor; sakin bir eğitim dönemi modele öğrenecek ilişki bırakmıyor. (CAPM ile 2010–2026: 78 kenar, notlardaki 58'e yakın.)
12. **(Günlük, eski ayrım) İlk baseline sonucu (test 2020–2026):** AR(1) ve Lasso-VAR(1) getiride sıfır tahminini geçemiyor (R²_oos ≤ 0). Beklenen bir sonuç; Lasso alfası doğrulama döneminde ayarlanmalı, oynaklık katmanı ciddi bir seçenek.
10. **Input-Output grafı model satırı olarak kullanılmıyor** (SIC ↔ NAICS eşlemesi maliyetli). Opsiyonel elle kenar kontrolü.
13. **İşlenmiş veri 1983'ten üretiliyor, analiz 1985'ten başlıyor.** Beta penceresi haftalıkta 104 hafta, 2 haftalıkta 52, aylıkta 60 ay, günlükte 252 gün; ilk pencere ısınmaya gidiyor. Böylece 1987 krizinin öncesi de artıklarda var. Ana girdi: `data/processed/artik_ff3_haftalik_1985.csv`. Ayrıntılar `data/processed/README.md`.
14. **CAPM dosyaları varsayılan olarak üretilmiyor.** Analizlerin tamamı FF3 artıklarıyla yapılıyor. CAPM sağlamlık kontrolü gerekirse `00_build_processed_data.py --capm` ile tek komutla üretilir.
15. **Analiz akışı haftalık veriye geçti; ayrım 1985–2012 / 2013–2018 / 2019–2026.** `scripts/02–03` işlenmiş dosyaları `load_processed` ile okuyor, frekans config'ten (`data.frequency`, varsayılan haftalık) veya `--frequency` ile seçiliyor; sonuçlar `results/<frekans>/`. Eski `01_make_residuals.py` kaldırıldı (yerini `00` aldı). İlk sonuçlar (FF3, Granger 1 gecikme, FDR %5, eğitim dönemi): haftalık 32 kenar, günlük 191 kenar. Haftalık test (2019–2026): AR(1) ve Lasso-VAR(1, α=0.01) sıfır tahminini geçemiyor (R²_oos −0,001 / −0,055); Lasso alfası val döneminde ayarlanmalı.

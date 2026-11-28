# Karar kayıtları

Veri hazırlığıyla ilgili kararlar ve gerekçeleri. Son güncelleme: 28 Kasım 2026.
Ayrıntılı veri sözlüğü: `data/processed/AI_CONTEXT.md`.

## Veri

1. **Kaynak:** Kenneth R. French Data Library. 49 sektörün günlük getirisi (**value-weighted**, teyit edildi; piyasa faktörüyle tutarlı) ve FF3 faktörleri (Mkt-RF, SMB, HML, RF). Birim yüzde.
2. **Dört frekans üretiliyor: günlük, haftalık, 2 haftalık, aylık.** Ana frekans haftalık: eş zamansız işlemden (az işlem gören sektörlerin habere geç tepki vermesi) kaynaklanan sahte bağlantılar ve gürültü azalıyor, yeterli gözlem var (~2.180 hafta). Günlük, sağlamlık kontrolü için. Aylık sadece ~500 gözlem içeriyor.
3. **Birleştirme kuralları:** Getiriler bileşik birleştiriliyor. Haftalar Cuma'da bitiyor, 2 haftalık dönemler 11 Ocak 1985'te biten çifte göre hizalı, dönem etiketi son işlem günü. `Mkt-RF` doğrudan bileşiklenmiyor (Mkt ve RF ayrı bileşiklenip farkı alınıyor). Kod: `src/sektorgraf/data/aggregate.py`.
4. **Dönem: ham veri 1983'ten, işlenmiş veri 1985'ten.** İlk iki yıl beta penceresinin ısınmasına gidiyor. Böylece 1987 krizinin öncesi de artıklarda var. Aylıkta 60 aylık pencere için ısınma 1980'den başlıyor.
5. **Ham birleştirilmiş dosyalar tutuluyor** (`49_sektor_*_1983`, `ff3_*_1983`): ham–artık karşılaştırması, faktörlerin kullanımı ve şeffaflık için.
6. **Ekibin önceki `49_sektor_*_1985.csv` dosyaları kaldırıldı.** 1985+ kısmı yenileriyle aynıydı, ama 4 Ocak 1985 haftası (ve ilk 2 haftalık dönem) 31 Aralık 1984'ü içermiyordu.
7. **Son dönem kısmi:** Veri 30 Eylül 2026 Çarşamba bitiyor. Haftalık, 2 haftalık ve aylık dosyalarda son satır eksik bir dönem. Kullanırken atılmalı veya işaretlenmeli.

8. **Ortak zaman ayrımı: eğitim 1985–2012, doğrulama 2013–2018, test 2019–2026.** Tek kaynak `configs/config.yaml → split`, herkes `sektorgraf.data.splits` üzerinden okur; ayrı ayrı tarih yazılmaz. Zaman sırasına göre, karıştırmadan. Eğitimde 1987, dot-com ve 2008 krizleri, testte 2020, 2022 ve 2025 var. Bir dönem bitiş tarihine göre atanır. Ayrım dosyalara bölünmüyor (aynı verinin kopyalarını önlemek için).

## Ön işleme

9. **Ana veri FF3 artıkları, ham getiri değil.** Ham veride sektörler arası ortalama korelasyon 0,54, FF3 artıklarında 0,03 (haftalık). Ortak piyasa hareketi çıkarılmazsa sektörler birbirini etkiliyormuş gibi görünür. (Günlük ön denemede ham Granger grafında 868, CAPM sonrası 58 kenar vardı.)
10. **Betalar kayan pencereyle, sadece geçmişten.** Haftalık 104 hafta, 2 haftalık 52 dönem, aylık 60 ay, günlük 252 gün. t döneminin betası [t − pencere, t − 1] aralığından tahmin ediliyor. Look-ahead yok (`tests/test_aggregate.py`).
11. **FF3 seçildi, CAPM varsayılan olarak üretilmiyor.** FF3, CAPM'nin piyasa etkisini zaten içeriyor ve ek olarak küçük şirket (SMB) ve value/growth (HML) etkilerini çıkarıyor. CAPM artıkları gerekirse: `00_build_processed_data.py --capm`.
12. **Oynaklık serisi:** Günlük FF3 artıklarının dönem içi kareleri toplamı (gerçekleşen varyans, %²). Haftalık, 2 haftalık ve aylık.
13. **Bilinen sınırlılıklar:**
    - FF3, büyük sektörlerden yayılan gerçek etkinin bir kısmını "piyasa" olarak silebilir (ör. 2008'de bankalar). Bu veriden bulunacak etkiler alt sınır olarak yorumlanmalı.
    - FF3'te faiz faktörü yok. Faize duyarlı sektörler (Banks, RlEst, Util) arasında ortak faiz şokundan kalan birlikte hareket olabilir.
    - SMB ve HML günlük değerlerinden bileşikleniyor (uzun ve kısa bacakları ayrı bileşiklemeye çok yakın bir yaklaşım).

## Doğrulama ve gözlemler (28 Kasım 2026)

- Haftalık FF3 artıkları ve betaları, rastgele haftalarda bağımsız bir regresyonla (scikit-learn) yeniden hesaplandı. Sonuçlar aynı. Gerçekleşen varyans elle hesaplananla aynı.
- Sektörler arası ortalama korelasyon (1985–2026, haftalık): ham 0,536 → CAPM artığı 0,059 → FF3 artığı 0,029. Günlük 0,543 → 0,021, aylık 0,522 → 0,041.
- Medyan FF3 piyasa betası 1,00 (haftalık). Gold piyasaya en az duyarlı (0,39). Banks ve Oil value tarafında (hml +0,76 / +0,70), Softw ve Chips growth tarafında (hml −0,67 / −0,47).
- Kısa haftalar: 382 hafta 4 gün, 3 hafta 3 gün (2007-01-05, 2012-11-02, kısmi 2026-09-30), 1 hafta 1 gün (2001-09-10, 11 Eylül sonrası borsa kapanışı).

## Değişiklik geçmişi

- Beta penceresi önce sadece günlük (252 gün, 1999'dan başlayan veri) için tanımlıydı. Frekansa göre ayrıldı ve başlangıç 1983'e çekildi.
- CAPM önce her seferinde üretiliyordu, isteğe bağlı oldu.
- Eski `scripts/01_make_residuals.py` kaldırıldı, yerini `00_build_processed_data.py` aldı.
- Henüz başlanmamış analiz aşamasına ait taslak kod ve kararlar repodan çıkarıldı; repo şu an sadece veri hazırlığını içeriyor.

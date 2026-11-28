"""Ortak metrikler (docs/is_bolumu.md, bölüm 2.1).

Bütün fonksiyonlar tahmin ve gerçek değerleri aynı biçimde alır: satırlar hafta (Date indeksi),
sütunlar 49 sektör, değerler yüzde. İki tablonun tarihleri ve sütun sırası birebir aynı olmalıdır;
değilse hata verilir (sessizce hizalama yapılmaz).
"""

import numpy as np
import pandas as pd
from scipy import stats


def check_aligned(pred: pd.DataFrame, actual: pd.DataFrame) -> None:
    """Tahmin ve gerçek tablolarının aynı tarih ve sütunlara sahip olduğunu, eksik değer olmadığını doğrular."""
    if list(pred.columns) != list(actual.columns):
        raise ValueError("Sütunlar (sektörler veya sıraları) gerçek değerlerle aynı değil.")
    if not pred.index.equals(actual.index):
        raise ValueError("Tarihler gerçek değerlerle aynı değil.")
    if pred.isna().any().any() or actual.isna().any().any():
        raise ValueError("Eksik değer var.")


def rmse(pred: pd.DataFrame, actual: pd.DataFrame) -> float:
    """Bütün haftalar ve sektörler üzerinden sqrt(ortalama((tahmin − gerçek)²))."""
    check_aligned(pred, actual)
    return float(np.sqrt(np.mean((pred.values - actual.values) ** 2)))


def rmse_by_sector(pred: pd.DataFrame, actual: pd.DataFrame) -> pd.Series:
    """Her sektör için haftalar üzerinden RMSE."""
    check_aligned(pred, actual)
    return np.sqrt(((pred - actual) ** 2).mean(axis=0))


def oos_r2(pred: pd.DataFrame, actual: pd.DataFrame) -> float:
    """Zero'ya göre örneklem dışı R²: 1 − SSE_model / SSE_zero. SSE_zero = Σ gerçek².

    0'dan büyükse model "hep sıfır" tahmininden iyidir; negatifse kötüdür.
    """
    check_aligned(pred, actual)
    sse_model = np.sum((pred.values - actual.values) ** 2)
    sse_zero = np.sum(actual.values ** 2)
    return float(1 - sse_model / sse_zero)


def direction_accuracy(pred: pd.DataFrame, actual: pd.DataFrame) -> float:
    """İşaretin doğru bilinme oranı. Gerçek değeri tam 0 olan gözlemler hariç tutulur.

    Tahmini 0 olan tek bir gözlem yanlış sayılır (yön belirtmemiştir). Bütün tahminleri 0 olan
    model (Zero) için tanımsızdır, NaN döner.
    """
    check_aligned(pred, actual)
    p, a = pred.values, actual.values
    if np.all(p == 0):
        return float("nan")
    mask = a != 0
    return float(np.mean(np.sign(p[mask]) == np.sign(a[mask])))


def diebold_mariano(pred_a: pd.DataFrame, pred_b: pd.DataFrame, actual: pd.DataFrame, h: int = 1) -> dict:
    """A ve B modellerinin tahmin isabetini karşılaştıran Diebold–Mariano testi (HLN düzeltmeli).

    Her hafta için iki modelin hata kareleri 49 sektör üzerinden ortalanır; test bu haftalık
    kayıp farkı serisi d_t = kayıp_A − kayıp_B üzerinde yapılır. h: tahmin ufku (bir adım ileri için 1).
    Uzun dönem varyans h−1 gecikmeye kadar otokovaryanslarla hesaplanır.
    p-değeri iki taraflıdır, t(T−1) dağılımından (Harvey–Leybourne–Newbold, 1997).

    Dönüş: {"stat", "pvalue", "mean_diff", "n"}. stat < 0 ise A'nın hatası daha küçüktür.
    """
    check_aligned(pred_a, actual)
    check_aligned(pred_b, actual)
    loss_a = ((pred_a.values - actual.values) ** 2).mean(axis=1)
    loss_b = ((pred_b.values - actual.values) ** 2).mean(axis=1)
    d = loss_a - loss_b
    T = len(d)
    d_bar = d.mean()
    dc = d - d_bar
    gamma = [np.sum(dc[k:] * dc[: T - k]) / T for k in range(h)]
    long_run_var = gamma[0] + 2 * sum(gamma[1:])
    if long_run_var <= 0:
        return {"stat": float("nan"), "pvalue": float("nan"), "mean_diff": float(d_bar), "n": T}
    dm = d_bar / np.sqrt(long_run_var / T)
    hln = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    stat = dm * hln
    pvalue = 2 * stats.t.sf(abs(stat), df=T - 1)
    return {"stat": float(stat), "pvalue": float(pvalue), "mean_diff": float(d_bar), "n": T}

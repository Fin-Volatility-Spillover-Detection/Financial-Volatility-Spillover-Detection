"""Graph WaveNet — tek mimari, dört graf ayarı (henüz yazılmadı, Aşama 2).

Ayarlar (graph_mode):
  "none"     : grafsız. Her sektör sadece kendi geçmişine bakar (graf konvolüsyonu kapalı).
  "random"   : graphs/utils.random_like(DY) ile üretilen kontrol grafı.
  "dy"       : DY etki tablosu (SADECE eğitim döneminden hesaplanmış!), top_k_per_row ile seyreltilmiş.
  "adaptive" : model grafı kendisi öğrenir: A = softmax(ReLU(E1 @ E2.T)).
               Eğitimden sonra learned_adjacency() ile okunur, graphs/utils.similarity ile DY'ye kıyaslanır.

Uygulama notları:
  - Referans: Wu et al. (2019), "Graph WaveNet for Deep Spatial-Temporal Graph Modeling", IJCAI.
    PyTorch Geometric Temporal veya yazarların resmi kodu temel alınabilir.
  - Girdi: models/windows.make_windows çıktısı (örnek, history, N, F).
  - Arayüz base.Model ile aynı olmalı ki evaluation/evaluate.py doğrudan kullanılabilsin.
  - Graf kuralı: W[i, j] = j → i. Kütüphanelerin çoğu A[kaynak, hedef] bekler; gerekirse W.T verin.
  - Erken durdurma doğrulama (val) dönemine göre. Test dönemine eğitim sırasında asla bakılmaz.
"""

from .base import Model


class GraphWaveNet(Model):
    def __init__(self, graph_mode: str = "none", adjacency=None, history: int = 20):
        assert graph_mode in {"none", "random", "dy", "adaptive"}
        self.graph_mode, self.adjacency, self.history = graph_mode, adjacency, history
        self.name = f"GWN[{graph_mode}]"

    def fit(self, train, exog=None):
        raise NotImplementedError("Aşama 2: Graph WaveNet implementasyonu")

    def predict(self, data, dates, exog=None):
        raise NotImplementedError

    def learned_adjacency(self):
        raise NotImplementedError

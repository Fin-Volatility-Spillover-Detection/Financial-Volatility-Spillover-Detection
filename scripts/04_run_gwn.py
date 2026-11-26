"""Adım 4 (Aşama 2): Graph WaveNet — grafsız / rastgele / DY / öğrenilen.

Taslak akış:
    resid    = residuals_ff3.csv
    W_dy     = top_k_per_row(results/graphs/dy_train.csv, k=5)   # SADECE eğitim döneminden!
    W_random = random_like(W_dy, seed=0)
    for mode, W in [("none", None), ("random", W_random), ("dy", W_dy), ("adaptive", None)]:
        model = GraphWaveNet(mode, W).fit(train)           # erken durdurma: val dönemi
        evaluate(model.predict(resid, test_dates), resid, cfg, name=model.name)
    compare("GWN[dy]", "GWN[none]", resid, cfg)
    compare("GWN[dy]", "GWN[random]", resid, cfg)
    similarity(W_dy_full, model_adaptive.learned_adjacency())
"""

raise SystemExit("Henüz uygulanmadı — bkz. src/sektorgraf/models/gwn.py")

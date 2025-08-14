# core/region_timer.py
import time
from collections import defaultdict

class RegionStayTimer:
    """
    Belirli bir koşulu (ör. yol dışında olma) sağlayan süreyi kişi bazında (track_id) tutar.
    """
    def __init__(self):
        self.state = defaultdict(lambda: {"cond": False, "t_start": None, "acc": 0.0})

    def update(self, track_id: int, condition: bool) -> float:
        """
        condition=True => süre birikir; False olursa sıfırlanır.
        Dönüş: birikmiş süre (saniye)
        """
        now = time.time()
        st = self.state[track_id]
        if condition and not st["cond"]:
            st["cond"] = True
            st["t_start"] = now
        elif condition and st["cond"]:
            st["acc"] = now - st["t_start"]
        else:
            st["cond"] = False
            st["t_start"] = None
            st["acc"] = 0.0
        return st["acc"]

    def get(self, track_id: int) -> float:
        return self.state[track_id]["acc"]

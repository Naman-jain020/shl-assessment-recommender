import time, logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class trace_span:
    def __init__(self, name): self.name = name
    def __enter__(self):
        self.t0 = time.time(); logging.info(f"[SPAN] {self.name} START"); return self
    def __exit__(self, exc_type, exc, tb):
        dt = time.time()-self.t0; logging.info(f"[SPAN] {self.name} END ({dt:.3f}s)")
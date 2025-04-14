#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
from collections import deque
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import torch, torch.nn as nn

MEAN = [1.0330383038436266, 0.9921870873230328, -5.986730383311507e-05, 0.999709199079134, 0.00018871763767504268, 1.0001191663871114, 0.00010791785095278968, -6.889660576601636e-06, 0.9860516957259265, 0.9729566646191812, -3.6380720025332668e-06, 1.0248069517773497, 0.9910081398957061, 1.0001973061100071, -0.0001358640079276542, -0.00018716975807537948, -0.00023357994728190492, 1.050537061310052, 0.00041538498954872777, 0.00012647459659548423, 0.99868456479531, 0.9927704092292291, 0.9914302523915063, 0.9999569777123373, 1.0097155053653102, 0.9598040141656471, 1.1414477781874402e-05, 1.0344419401524945]
STD  = [0.3652252280428611, 0.48739150905466955, 1.0076352657638126, 1.0493326559525138, 1.006322919804772, 1.4002176241388296, 1.0094259855862298, 1.0063325457549883, 0.505650927798952, 0.5253252364315946, 1.0060853076783716, 0.3809724692806453, 0.47504535280103805, 1.027789063737055, 1.0091309884792374, 1.0059060292420945, 1.0062963827082931, 0.16452628281899337, 1.0087259731914986, 1.008708139108654, 0.6002081989185843, 0.4999109708349692, 0.5654256631807738, 1.1936360604839695, 0.3974849745384516, 0.31342224983603495, 1.0063514133534008, 0.6753538036610093]
NUM_FEATS = len(MEAN)

class SmallMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

base = SmallMLP(NUM_FEATS)
model = torch.quantization.quantize_dynamic(base, {nn.Linear}, dtype=torch.qint8)
model.load_state_dict(torch.load("./save/STUDENT/sml_mlp_pruned_int8.pt", map_location="cpu"))
model.eval()

class CsvTailHandler(FileSystemEventHandler):
    TARGET = "EVERY_PASS_FAIL.csv"
    def on_modified(self, event):
        if event.is_directory or not event.src_path.endswith(self.TARGET):
            return
        try:
            with open(event.src_path, "r", encoding="utf-8") as f:
                last = deque(f, 1)[0] # 마지막 한줄만 read.
            feats = [float(x) for x in last.strip().split(",")[:NUM_FEATS]]
            norm  = [(v - m) / s for v, m, s in zip(feats, MEAN, STD)]
            x     = torch.tensor([norm], dtype=torch.float32)
            prob  = model(x).item()
            pred  = int(prob > 0.5)
            ts    = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"{ts}  prob={prob:.4f}  pred={pred}")
        except Exception as e:
            print("Inference error:", e)

if __name__ == "__main__":
    WATCH_DIR = "!REAL_WAVE_FORM"
    handler   = CsvTailHandler()
    observer  = Observer()
    observer.schedule(handler, path=WATCH_DIR, recursive=False)
    observer.start()
    print(f"Watching {WATCH_DIR}/{handler.TARGET}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

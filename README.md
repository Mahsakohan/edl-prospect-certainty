After cloning this repo you need to install the requirements:
This has been tested with Python `v3.6.8`, Torch `v1.3.1` and Torchvision `v0.4.2`.

```shell
pip install -r requirements.txt
```

```

for training the network:
python main.py --train --dropout --uncertainty --mse --epochs 50

```
```
for testing the network on IID without Prospect certainty:
python test.py

```
```
for testing the network on IID with Prospect certainty:
python iid_test.py

```
```
for testing the network on OOD without Prospect certainty:
python ood_without_pc_test.py

```
```
for testing the network on OOD with Prospect certainty:
python ood_test.py

```


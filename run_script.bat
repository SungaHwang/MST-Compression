@echo On

call C:\Users\sunga\anaconda3\Scripts\activate.bat C:\Users\sunga\anaconda3\envs\sunga2

python C:\MST-Compression\ResNet18\main.py
python C:\MST-Compression\ResNet18\main_layer.py
python C:\MST-Compression\ResNet50\main_layer.py

call deactivate

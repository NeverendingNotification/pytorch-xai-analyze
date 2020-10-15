# pytorch-xai-analyze

pytorch xai scripts to interpret Deep Learning models.

Current XAI Algorithms

- Anchors [^1]
- SHAP [^2]
- Grad-CAM[^3]



# Instalation
```
git clone https://github.com/NeverendingNotification/pytorch-xai-analyze.git
cd pytorch-xai-analyze
pip install -r requirements.txt
```


# Trainining
default setting file is setting.yml.
```
 python main.py --mode train
```


# Visualize
analyze and visualize trained model.
```
 python main.py --mode analyze
```
![sample](https://neverendingnotification.github.io/images/xai_sample.jpg)



[^1]: https://homes.cs.washington.edu/~marcotcr/aaai18.pdf
[^2]: https://github.com/slundberg/shap
[^3]: https://arxiv.org/abs/1610.02391



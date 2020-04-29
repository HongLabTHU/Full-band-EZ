# Full-band-EZ
Code for a tool for the localization of epileptogenic zone based on full band features of stereo-EEG signal in patients with epilepsy, as described in the original paper: *Localization of Epileptogenic Zone Using Full-band Features of SEEG*.
### Quickstart
This repository requires python3.6. Install the dependencies from PyPI:
```python
pip3 install -r requirements.txt
```
To reproduce the major results reported in the paper, download the data from this [link](https://github.com/TongZhh/BrainQuake/raw/master/data). put it under the data directory. Then simply run the notebook *main_fig.ipynb*.<br>
If you want to see the original signal and time-frequency map of any channels by pressing on these three figures, you can also download and run another version with interaction:
```python
python3 main.py
```
### Support
If you have a question or feedback, or find any problems, you can contact us by [email](mailto:zhaotongztzt@gmail.com) or open an issue on GitHub.

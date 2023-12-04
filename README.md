# Explaining the Predictions of the MOSNet Classifier via the LIME Framework
Implementation of paper: ["'Why Should I Trust You?' Explaining the Predictions of any Classifier"](https://arxiv.org/pdf/1602.04938.pdf). By modifying the proposed framework to work for [MOSNet](https://arxiv.org/pdf/1904.08352.pdf), we attempt to draw conclusions about which features are particularly relevant during audio classification.

### Author
Ada Lamba.39

### Version
CSE 5539 - Khaliligarekani</br>
December 4, 2023

## Project Deliverables
- [Code](https://github.com/abarach/mosnet-lime/tree/main)
- [Presentation slides](https://github.com/abarach/mosnet-lime/tree/main/docs/2023.12.04_FinalProjectPresentation_Lamba.pdf)
- [Paper](https://github.com/abarach/mosnet-lime/tree/main/docs/FinalReport_Lamba.pdf)

## File Structure
The file structure is modified from the [original MOSNet implementation](https://github.com/lochenchou/MOSNet).
- `data` contains scripts and tables delineating the VCC Voice Conversion Dataset used to train MOSNet. 
- `lime` contains a copy of the LIME framework implementation.
- `pre_trained` contains pre-trained `.h5` files for three different MOSNet versions: CNN, CNN + BLSTM, and BLSTM.
- `model.py` defines the MOSNet model object, and **was modified for this project**.
- `mosnet_lime.py` **is the code written for this project**.
- `original_MOSNET_README.md` is the original README for the MOSNet implementation. 
- `requirements.txt` lists the library dependencies for the project. 
- `test.py`, `train.py`, `custom_test.py`, and `utils.py` are training, testing, and utility scripts for MOSNet. 

## MOSNet Framework
MOSNet is a convolution and recurrent neural network proposed by Lo et al. in 2021 [2]. It takes the magnitude spectrogram of an audio signal as input and predicts the [mean opinion score (MOS)](https://en.wikipedia.org/wiki/Mean_opinion_score) of the signal. 

## LIME Explanations 
The LIME framework attempts to explain a classifier's prediction by using an already-interpretable model which locally approximates the target classifier [1]. 

## Project Status
`mosnet_lime.py` initializes the LIME explainer and MOSNet pre-trained model. 

## To Do
*Final Presentation*: Monday, December 4, 2023 </br>
*Slides, Code, and Paper due*: Tuesday, December 5, 2023

- Predict MOS score for a sample input.
- Generate a LIME explanation for that sample input.
- Generalize the script to predict and explain several data inputs. 
- Write paper. 
- Write presentation slides. 

## References
[1] M. T. Ribeiro, S. Singh, and C. Guestrin, [“‘Why Should I Trust You?” Explaining the Predictions of Any Classifier,”](https://arxiv.org/pdf/1602.04938.pdf) in *KDD ’16: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2016, pp. 1135–1144.</br>
[2] C.-C. Lo, S.-W. Fu, W.-C. Huang, X. Wang, J. Yamagishi, Y. Tsao, and H.-M. Wang, [“MOSNet: Deep learning based objective assessment for voice conversion,”](https://arxiv.org/pdf/1904.08352.pdf) in *Proc. Interspeech 2019*, 2019.

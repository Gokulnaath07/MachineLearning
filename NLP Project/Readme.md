Dataset: Twitter US Airline Sentiment Dataset
if you are trying to run .ipynb file you can open it straight in the jupyter or in colab


if you don't have CUDA installed in your pc, which can only be there if you are using NVIDIA GPUs then use thios model 'prajjwal1/bert-mini' in the place where we have bert-base-uncased. Or its easier if you go with Colab, which can run any model in less than 4 minutes.

to run final.py file
open the folder in the vscode 

better create a virtualenv for installing the libraries without disturbing the local version
pythom -m venv venv
.\venv\Scripts\Activate.ps1

Use the requirement.txt file 
pip install -r requirements.txt

now run it by using python final.py

or open the final_colab.py file in the colab and run it 

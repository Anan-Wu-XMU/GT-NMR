# README
## Table of Contents

### installation environment with conda
1. create a conda environment
   ```bash
   conda create --name gtnmr python=3.8.16
   conda activate gtnmr
   ```
2. install dependencies
   ```bash
   pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
   pip install torch_geometric
   pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
   pip install yacs
   pip install ogb
   pip install opt-einsum
   pip install rdkit==2023.03.2
   pip install torchmetrics
   pip install tensorboardX

### Start training GT-NMR
   #### you can start training GT-NMR by running the following command:
   the following command will train the model on the 13C NMR and 1H NMR, respectively.
   ```bash
   python main.py --cfg configs/gt13C.yaml accelerator "cuda:0" seed 1
   python main.py --cfg configs/gt1H.yaml accelerator "cuda:0" seed 1
   ```

### Use pretrained model for prediction
#### you can use the pretrained model for prediction 13C NMR with the following ways:
1. Fast prediction by inputting a SMILES string, 'CN1C(CCC1=N)C(O)=O' for example.
   ```bash
   python main.py --cfg configs/cunstom_inference.yaml accelerator "cuda:0" dataset.inference 'CN1C(CCC1=N)C(O)=O'
   ```
   ```angular2html
   the output will be:
   
   ```
2. Prediction by inputting a SMILES file, 'smiles.csv' for example, the file should be in the format of 'smiles' column. and the path of the file is ./inference_files
   ```bash
    python main.py --cfg configs/cunstom_inference.yaml accelerator "cuda:0" dataset.inference 'smiles.csv'
    ```
    ```angular2html
   
    ```
3. Prediction by inputting a mol file, the path of the mol file is ./inference_files
   ```bash
    python main.py --cfg configs/cunstom_inference.yaml accelerator "cuda:0" dataset.inference 'mol_example.mol'
    ```
    ```angular2html
   
    ```
4. Prediction by inputting mol files in a folder, the path of the folder is ./inference_files/mols
   ```bash
    python main.py --cfg configs/cunstom_inference.yaml accelerator "cuda:0" dataset.inference 'mols'
    ```
    ```angular2html
   
    ```




   
   
   


# DeepAdapter
Accurate identification and correction of true biological signals from diverse undesirable variations in large-scale transcriptomes is essential for downstream discoveries. Herein, we develop a universal deep neural network, called DeepAdapter, to eliminate various undesirable variations from transcriptomic data. The innovation of our approach lies in automatic learning of the corresponding denoising strategies to adapt to different situations. The data-driven strategies are flexible and highly attuned to the transcriptomic data that requires denoising, yielding significant improvement in reducing undesirable variation originating from batches, sequencing platforms, and bio-samples with varied purity beyond manually designed schemes. Comprehensive evaluations across multiple batches, different RNA measurement technologies and heterogeneous bio-samples demonstrate that DeepAdapter can robustly correct diverse undesirable variations and accurately preserve biological signals. Our findings indicate that DeepAdapter can act as a versatile tool for the comprehensive denoising of the large and heterogeneous transcriptome across a wide variety of application scenarios.   
For more results about variation removal and signal conservation, please refer to our article (XXX.XXX).

# Repository strucuture
`codes`: the directory contains the notebooks for reproducing the results  
`data`: the directory contains the input files of different unwanted variations  
`models`: the directory contains the trained models for different variation removal tasks  
`imgs`: the directory contains the results images
# Usage
The main packages for this work are:
* torch==2.0.1
* scikit-learn==1.3.2
* umap-learn==0.5.3

Please enter the following commands to reproduce the results files.
```sh
$ # Install dependencies
$ pip install -r requirements.txt
$ # Launch jupyter notebook
$ jupyter notebook
```
# Results
Focusing on bulk transcriptome analysis, we defined 4 types of unwanted variations and corrected them with DeepAdapter: `batch varation`, `platform variation`, `purity variation`, `unknown/mixed variation`. The performances were assessed from 2 aspects: `variation correction` and `biological signal conservation`.  

For `variation removal`, we demonstrated that DeepAdapter achieved the highest performances in unwanted variations elimination compared with classical baseline methods.
<div align=center>
<img src="https://github.com/mjDelta/DeepAdapter/blob/main/img/variation removal.png" width="90%" height="90%">
</div>  

For `biological signal conservation`, we revealed that DeepAdapter efficiently preserved the true signals of interest.  

We conceptualized the **donor-wise information** as true signals for integrated analysis <ins>**across batches**</ins>. LINCS and Quartet projects collected profiles from 4 participants.
<div align=center>
<img src="https://github.com/mjDelta/DeepAdapter/blob/main/img/batch.png" width="90%" height="90%">
</div>  

We conceptualized the **cancer types** as the biological signals for integrated transcriptome investigations <ins>**across RNA-seq and microarray**</ins>. CCLE and GDSC projects collected profiles from 25 cancer types.
<div align=center>
<img src="https://github.com/mjDelta/DeepAdapter/blob/main/img/platform.png" width="90%" height="90%">
</div>  

We conceptualized the **lineages** as the biological signals for integrated transcriptome investigations <ins>**across cell lines and tumor tissues**</ins>. CCLE and TCGA projects collected profiles from 25 lineages.
<div align=center>
<img src="https://github.com/mjDelta/DeepAdapter/blob/main/img/purity.png" width="90%" height="90%">
</div>


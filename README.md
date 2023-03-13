# TransNetR: Transformer-based Residual Network for Polyp Segmentation with Multi-Center Out-of-Distribution Testing (MIDL 2023)

TransNetR is an encoder decoder network which begins with a pre-trained ResNet50 as the encoder. 

## In-distribution and Out-of-distributuion dataset
<img src="IntroTransNetR.png">
<em>**Figure 1: Illustration  of  different  scenarios  expected  to  arise  in  real-world  settings. The proposed work conducted both in-distribution and out-of-distribution validation process.  C1 to C6 represent the different centers data present in PolypGen dataset**</em>

## Architecture
<img src="Architecture.jpg">
<em>*Figure 2: Block diagram of TransNetR along with the Residual Transformer block*</em>

## Results (Qualitative results)
<img src="supplementry_C1.jpeg">
<em>Figure 3: </em>

## Qualitative results
<img src="supplementry_C6.jpg">
<em>*Figure 4:*</em>


## Citation
Please cite our paper if you find the work useful: 
<pre>
  @INPROCEEDINGS{9183321,
  author={D. {Jha} and M. A. {Riegler} and D. {Johansen} and P. {Halvorsen} and H. D. {Johansen}},
  booktitle={Proceedings of the Medical Imaging with Deep Learning}, 
  title={ TransNetR: Transformer-based Residual Network for Polyp Segmentation with Multi-Center Out-of-Distribution Testing}, 
  year={2023}}
</pre>



## Contact
please contact debesh.jha@northwestern.edu for any further questions. 

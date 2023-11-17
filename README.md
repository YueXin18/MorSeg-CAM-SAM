# MorSeg-CAM-SAM
## Introduction
We propose a novel weakly supervised breast lesion segmentation framework comprising four main modules: a traditional segmentation module based on morphology, a semantic information extraction and lesion localization module, an information fusion module, and a SAM fine-grained segmentation module. The traditional segmentation module utilizes morphology to perform initial segmentation and extract contour information from medical images, focusing on the shape, edge, and direction of lesions. The semantic information extraction and lesion localization module, leveraging image-level category labels, trains a classification network and achieves a fuzzy localization of lesions through the heat map provided by CAM. The information fusion module then adeptly combines the outputs from these two modules, generating a more comprehensive lesion area. Finally, SAM utilizes this area as a prompt for segmenting lesions, refining the segmentation process and enhancing the results through post-processing. 
![framework](FIG/frame.png)
## Code Explanation
1. Traditional segmentation based on morphological feature

   `MorSeg` contains code for image preprocessing, automatic color enhancement, clustering, and threshold segmentation. The image normalization range for ACE operation and the threshold value for threshold segmentation can be adjusted as appropriate during image processing. In addition, this section contains the code `layer-select.py` for extracting the parenchymal layer of the breast, and `mor.py` for filtering the parenchymal layer of the breast lesions according to their morphological characteristics.
2. CAM-Guided classification model for lesion localization

   Run `cam.py` to implement breast lesion localization, where the trained classification network is needed for that part.
3. Feature fusion and region synthesis

   Run `fusion.py` to implement information fusion and lesion region synthesis.
4. SAM-Optimized lesion segmentation

   `segment-anything-main`contains code to enhance the segmentation results after fusion. ·sam-point.py` implements optimisation of the segmentation results after region synthesis using point cues. `sam.py` implements optimisationn of the segmentation results after region synthesis using box cues.
5. post-processing

   Run `post-processing.py`,the hole regions of the results after SAM segmentation will be filled.

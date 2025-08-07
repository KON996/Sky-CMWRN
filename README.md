# Sky-CMWRN
A Cascaded Framework for Fine-Grained Weather Intensity Recognition

It includes the datasets and models mentioned in the article.

WCID  

To address the shortcomings of traditional weather classification data in meteorological intensity representation and weather coexistence, this study constructs the WCID. WCID is constructed by systematically integrating existing public datasets (MWD-2018, MWD-2017 and DWD). Based on the intensity criterion system established in Section 2.1, a triple screening rule is formulated: (1) eliminate images containing no sky or very low proportion of sky (e.g., indoor scenes); (2) retain images depicting cloudy, sunny, and rainy conditions; (3) retain diverse scene samples such as urban, rural, and natural landscapes to avoid overfitting a single environment. Through strict manual selection, 4820 images are curated to form a multi-task benchmark dataset supporting the regression of SSBI and the classification of rainfall intensity. Limited by strict screening conditions, the current dataset size has certain limitations and needs to be continuously expanded in the future.  

This study commences with the annotation of the WCID. The labeling process adheres to the standards for weather condition intensity established in Section 2.1, utilizing self-developed annotation software to enable collaborative labeling of SSBI regression and rainfall intensity classification (as shown in Figure 3). The annotation interface adopts a dual-view interactive design: The left side displays the image to be annotated in real time, and the right side sets the coordinate system and the rainfall intensity selection module. The procedure is as follows: (1) SSBI labeling: In the coordinate system with fixed y axis (y=0), annotators calibrate the point position along the x axis [-1, +1] according to the sky condition in the picture, and the system records the point's x-value as the SSBI; (2) rainfall intensity labeling: Annotators assign the classification label using a three-level radio button selection (no-rain condition, light-moderate rain, heavy rain); (3) data storage: Annotation results are stored in CSV format. Fields include the image filename, SSBI value (floating-point), and rainfall intensity classification label (integer-encoded). The resulting data distributions are summarized in Tables 5 and 6. Figure 4 shows representative sample images alongside their corresponding SSBI and rainfall intensity distributions.  

SPD  

To support training the sky perception module, samples without sky are extracted from the same data source: (1) Screen indoor and outdoor scene images with missing sky or a particularly small proportion of sky; (2) Retain images containing multi-scale objects (e.g., close-range vehicles, distant buildings) to enhance the robustness of the module. Finally, 4100 images without sky are obtained. Together with the sky images in WCID, these images form the SPD.  

The SPD annotation process employs structured storage. The relative paths and corresponding categories (sky/no-sky) for all 8920 images are recorded in CSV files, forming a standardized training data index. Thus, both the WCID and the SPD are established, jointly providing comprehensive data support for subsequent model training.  


Sky-CMWRN  

To utilize the WCID, this paper presents Sky-CMWRN,an innovative framework that cascades a sky perception module with CMWRN(a Convolutional Neural Network (CNN)-based multi-task weather recognition network). CMWRN, which jointly models two tasks—sky state bias index (SSBI) regression and rainfall intensity classification—enables fine-grained weather intensity recognition in outdoor images. To mitigate sky state misjudgment in sky-less scenes, we incorporate a lightweight ResNet-18-based sky perception module. This module filters input images before multi-task processing by CMWRN, enhancing the model’s generalization to complex scenarios.

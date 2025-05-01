# stress-anxiety-classifier
Pre-made codes with datasets to explore using CNN's and GAN's. Using primarily FER-2013 and RAF-DB.


Standard Operating Procedures 
(SOP) / Manual
 
Initial Canned Code / Source
I spent a week or two studying pre-made codes and repositories that I researched and explored. These are the ones that stuck out to me for the classifier for facial expressions.
•	Omar178
This project uses the FER 2013 dataset and includes preprocessing, a training loop, and model saving. I was very impressed with the code and functionality it offered.
•	prabhuomkar
This repository offered a lot more adaptability than the first CNN. You can easily plug in your own dataset and class mappings. It shows how to fine-tune to ResNet/18/34/50. It was also ready for 4-class stress models.
Then I also researched StyleGAN-2.  
•	NVlabs
This is an extremely professional repository since it is by Nvidia. Original with many references to follow and research. It did require some setup with TensorFlow, and different StyleGAN-2 configs
•	rosinality
This one is the opposite of Nvidia’s, it is much easier to integrate into PyTorch workflows. Worked better for lightweight modifications and smaller dataset. Fun code to test different small datasets for quick results.
Last but not least, PyTorch Lightning was essential to running my code smoothly. 
•	LightningAI
Dataset
The primary dataset that I used for training the model was FER-2013. This database consisted of 30,000 48x48 grayscale images. 
•	FER-2013
I coupled this dataset with Real-world Affective Faces Database (RAF-DB), which offered 15,000 images of varying specifications.
•	RAF-DB

Refined Code
My refined code was created by combining the many GitHub repositories that inspired me, as well as the research I did on my own. I uploaded all of my code and documentation to my GitHub. Here is the link:
•	GitHub
The classifier code is listed under stress-anxiety-classifier.py
The styleGAN2 code is listed under stylegan2.py
The python script that combines FER-2013 and RAF-DB into a large comprehensive dataset is convert-datasets.py
Final Dataset
The final dataset was created using the Python script detailed above. It works by enumerating both datasets. It converts the larger RAF-DB RGB files to black and white 75x75 pixel images to match the FER-2013 images. It uses a 70% training, 10% validation, and 20% testing split.
Steps to use the script:
1.	Download FER-2013 and RAF-DB
2.	Make a parent folder, and define the path in the convert-datasets.py
3.	Save FER-2013 to /fer-dataset, and RAF-DB to /raf-dataset
4.	Run the convert-datasets.py code
Hardware Tools
Since I was doing this individually, creating the code and troubleshooting took me the longest time. I did most of this locally on my personal computer, which had plenty of computing power to run a low number of epochs.
For IDE’s I used Jupyter Notebook and Google Colab. Training on Colab I used the Nvidia T-4 GPUs. 
For libraries I used PyTorch, Torchvision, PyTorch Lightning, Torchmetrics, matplotlib, seaborn

Building the Model
Classifier
For this classifier, I used ResNet-50, which means a residual network that uses 50 layers of reasoning. I used this version because it can capture complex patterns and more subtle expressions. ResNet-50 is pretrained on ImageNet1K V2. The 49 convolution layers connect to the FC layer, which only has the 4 outputs detailed above. 
The loss function that I decided to use was cross-entropy loss, which forces the model to be confident in picking only one of the 4 emotions. The optimizer that I used was Adam W with cosine annealing LR scheduler. This was the most optimal optimizer as it reduced overfitting, it is stable and efficient, especially when using a pretrained model like ResNet-50.
To properly evaluate and visualize our results, I implemented a multitude of different evaluation metrics. First of all, after each Epoch, it saved the loss, accuracy, precision, recall, F1, and Matthew’s Correlation Coefficient (MCC). 
•	Loss: How wrong the model’s predictions were versus the actual labels
•	Accuracy: Percentage of correct predictions from all predictions
•	Precision: How many claimed positives were actually positive
•	Recall: How well the model finds all correct examples
•	F1: The mean of precision and recall
•	MCC: A balanced metric that considers both true/false positives and negatives
In addition to these numerical metrics, I also have the code output several visualizations. These include a loss curve graph and a confusion matrix. A confusion matrix works by comparing predicted positive/negative and actual positive/negative. 
GAN
This script combines images from the FER and RAF-DB datasets, remaps their emotion classes to four stress-related categories (relaxed, mild stress, anxious, PTSD), and splits them into structured training, validation, and test folders using a 70/10/20 ratio.
A conditional GAN (based on a simplified StyleGAN like architecture) is then trained on this processed data to generate synthetic stress-expressive face images.
The model uses 100-dimensional latent vectors, class embeddings, and logs both training loss and FID scores during validation. Images are generated every epoch, and full sample grids are saved every 25 epochs.
The output:
•	Generated_samples/ directory with snapshots
•	Lightning_logs/ have TensorBoard logs
•	Trained generator state at final checkpoint
•	Validated on val/ folder using FID 
Challenges
I faced many challenges throughout the entire course of developing my project. I knew that my model was only going to be as accurate as my dataset. The first issue that I dealt with was the class imbalances, especially dealing with PTSD being underrepresented. Even after trying to duplicate and flip images to make classes more balanced, my results were not very affected by this.
Another large challenge that took me a couple of weeks to finally troubleshoot was a coding one. My initial GAN model was a DBGAN, but after checking the instructions, I realized I had to upgrade my model to a StyleGAN-2.  I ran into issues when “pip installing the StyleGan-2 files”. I had to download the files locally and then I had to edit a bunch of the files within the folder. I fixed this by starting over in a new IDE and redownloading all dependencies.
One key logical issue is that I am manually classifying emotions into a psychological state. Although the mappings between the two have correlations, this doesn’t mean that they are accurate. This causes some issues for GAN as the output is often biased for a detailed emotion that is not generating clear stress levels. To solve this, I tried to remove bias, balance, and normalize the datasets.
Final Results	
Classifier
 
My train loss was good, showing constant learning from the steady decreasing from 1.271 → 0.529. My Validation Loss was not as optimal, it dropped then leveled out, with epoch 9 at its best being 0.412. 
 
My final values were:
•	Train Loss: 0.529
•	Validation Loss: 0.875
•	Validation Accuracy: 0.748
•	Validation Precision: 0.744
•	Validation Recall: 0.748
•	Validation F1: 0.744
•	Validation MCC: 0.635
My validation accuracy improved, then stabilized. From 0.436 → 0.748, peaked around epoch 10–14. My validation’s precision, recall, and F1 all increased together and continued to stabilize at later epochs. F1-score peaked at 0.748, good consistency after epoch 10. My MCC improved sharply, then plateaued. From 0.050 → 0.635, shows balanced, consistent classification.
Metric	Best Value	Epoch
Accuracy	0.753	10
F1-Score	0.753	10
MCC	0.648	10
Val Loss	0.412	9–10

 
The confusion matrix is very helpful in identifying where the model gets confused. The model is very successful at detecting anxious faces. There is a reasonable separation between mild stress and anxiety, though some overlap. Relaxed face was detected decently. 
However, the model performed weakly at detecting PTSD. The model often mistakes PTSD for Anxious, which makes some sense. There is also moderate confusion between Relaxed and Mild Stress. If I had more time to collect more images and run more epochs, it would help determine these slight differences.
GAN
I ran my DBGAN locally, and my StyleGAN-2 in AWS. However, when I was uploading to git, my generated samples got deleted. I saved my epoch_2000 locally beforehand luckily. I re-ran the script a couple of days ago and only got to epoch_625.
All of my generated samples are in my GitHub under generated_samples.
This is epoch 2000:
 
Through the images, we can see the different emotions and their respective stress levels.
Here is the iterations of epochs in multiples of 25, that would have the FID scoring run.
 
The image shows many important metrics. That image was a screenshot of the StyleGAN-2 running.
•	V_num: Version number
•	g_loss: generator loss
•	d_loss: discriminator loss
•	FID: Frechet Inception Distance
FID is a very important measure of accuracy, comparing the generated images to real images.
 
References:
eztravis101. “EZTRAVIS101/Stress-Anxiety-Classifier: Pre-Made Codes with Datasets to Explore Using CNN’s and Gan’s. Using Primarily Fer-2013 and RAF-DB.” GitHub, github.com/eztravis101/stress-anxiety-classifier/tree/main. Accessed 1 May 2025. 
Lawton, George. “What Is Fréchet Inception Distance (FID)?: Definition from TechTarget.” Search Enterprise AI, TechTarget, 21 Nov. 2024, www.techtarget.com/searchenterpriseai/definition/Frechet-inception-distance-FID. 
Lightning-AI. “Lightning-Ai/Pytorch-Lightning: Pretrain, Finetune Any AI Model of Any Size on Multiple Gpus, Tpus with Zero Code Changes.” GitHub, github.com/Lightning-AI/pytorch-lightning. Accessed 1 May 2025. 
NVlabs. “NVlabs/Stylegan2: StyleGAN2 - Official Tensorflow Implementation.” GitHub, github.com/NVlabs/stylegan2. Accessed 1 May 2025. 
NVlabs. “NVlabs/Stylegan2: StyleGAN2 - Official Tensorflow Implementation.” GitHub, github.com/NVlabs/stylegan2. Accessed 1 May 2025. 
otaha178. “OTAHA178/Emotion-Recognition: Real Time Emotion Recognition.” GitHub, github.com/otaha178/Emotion-recognition. Accessed 1 May 2025. 
Prabhuomkar. “Prabhuomkar/Pytorch-CPP: C++ Implementation of Pytorch Tutorials for Everyone.” GitHub, github.com/prabhuomkar/pytorch-cpp. Accessed 1 May 2025. 
“Real-World Affective Faces Database.” Real-World Affective Faces (RAF) Database, www.whdeng.cn/RAF/model1.html. Accessed 1 May 2025. 
Sambare, Manas. “Fer-2013.” Kaggle, 19 July 2020, www.kaggle.com/datasets/msambare/fer2013. 


## Instructions to Run the Code

### 1, Prepare raw data
-	download Chest X-ray data from https://www.kaggle.com/nih-chest-xrays/data/data
-	Upload 12 image zip files onto Gatech remote server by ftp tools, and unzip them into one folder “images” under “~/spark-warehouse/” folder.
-	Upload “upload “Data_Entry_2017.csv” file to Gatech server and put it to hadoop under input folder.
-	Upload freezed ResNet50 model “resnet50.pth” to Gatech server and put it to hadoop under models folder
-	Run the following command to setup environment: source /scratch/dfs/etc/env.sh
To upload images to hadoop, run:  hdfs dfs -put images .
To upload “Data_Entry_2017.csv” to hadoop, run: hdfs dfs -put Data_Entry_2017.csv input
To upload “resnet50.pth” to hadoop, run: hdfs dfs -put resnet50.pth models

### 2. Feature extraction by Freezed Resnet50 layers:
-	Upload and put the “BottleNeckFeatureExtraction_Server.py” and “btneck.sh” file under “~/spark-warehouse/” folder on the Gatech server.
-	Run the following command to setup environment:
source /scratch/dfs/etc/env.sh
export PATH=/usr/local/anaconda3/bin:$PATH
-	Run the following command to start feature process:
		nohup ./btneck.sh &

### 3. Download train_dataset, valid_dataset, and test_dataset from hadoop outputs folder, and copy to local disk by ftp tools.

### 4. Train the three layers neural net models for multilabel classification using local machine with Nvidia GeForce GTX 1070.
-	Pip Install pandas, numpy, matplotlib, torch, torchvision, and pyspark
-	Setup pyspark environment by:
     	export PYSPARK_DRIVER_PYTHON=jupyter    
export PYSPARK_DRIVER_PYTHON_OPTS='notebook'
-	Run “pyspark” in terminal, and load “fcNN.ipynb”, and then run the code cells in order. ( change the values of path_to_train_dataset, path_to_valid_dataset, path_to_test_dataset if necessary).

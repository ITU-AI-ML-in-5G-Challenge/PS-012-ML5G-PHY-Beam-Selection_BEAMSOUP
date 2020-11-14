# ML5G_challenge

#### Building the environment

1 - Dependencies

1.1 System dependencies
- Hardware: CUDA Capable GPU
- Libraries: CUDA V10.0.130 
- Python 3.7

1.2 - Python Dependencies
Python packages:
- numpy==1.17.2
- open3d-python==0.7.0.0
- pandas==1.0.3
- h5py==2.10.0
- tensorflow-gpu==1.14.0
- keras==2.3.1
- matplotlib==3.1.1
- Please run the script  setup.sh to build the Python environment required
to run our code

The folder contains additional file other than the 4 required these are included in the submission and they are:
- mimo_channels.py
- CSVHandler.py
- models.py



2 - Running the code

After building the environment, you can build the costum features as shown in 2.1 and
train the model as shown in 2.2. To test the model follow 2.3.
Step 2.1 saves the dataset that will be used to train the model in 2.2 and to test  it in 2.3
Step 2.2 stores the train model as trained_model.h5
Step 2.3 creates the predictions beam_test_pred.csv

2.1 - Getting the features

IMPORTANT: to run the costumized front end is necessary to specify a base folders of s008, s009 and s010 that contain the unzipped raw_data.

This parameter is in the first line of beam_train_frontend (-> base_folder_008, base_folder_009) and beam_test_frontend (-> base_folder_010).

It assumes that the structure is the same as the one provided for the challenge, namely:

The csv file , lidar and ray-tracing data of s008 are respectively in
- base_folder+'/raw_data/CoordVehiclesRxPerScene_s008.csv'
- base_folder+'/raw_data/s008_Blensor_rosslyn_scans_lidar/'
- base_folder+'/raw_data/ray_tracing_data_s008_carrier60GHz/'

The csv file , lidar and ray-tracing data of s009 are respectively in
- base_folder+'/raw_data/CoordVehiclesRxPerScene_s009.csv'
- base_folder+'/raw_data/s009_Blensor_rosslyn_scans_lidar/'
- base_folder+'/raw_data/ray_tracing_data_s009_carrier60GHz/'

The csv file and lidar data of s010 are respectively in
- base_folder+'/raw_data/CoordVehiclesRxPerScene_s010.csv'
- base_folder+'/raw_data/rosslyn_scans/'

Note that the lidar data is assumed to be inside the specified folder, unzipped and in the subfolder format "scans_run00000", "scans_run00001"... each of them with then flow__.pcd files

Once the folder are well specified the script can be run and it will produce:

- folder obstacles_008 : containing a .npz file with the lidar data for each episode in s008
- folder obstacles_009 : containing a .npz file with the lidar data for each episode in s009
- folder obstacles_010 : containing a .npz file with the lidar data for each episode in s010
- lidar_008.npz : aggregated lidar data for training
- lidar_009.npz : aggregated lidar data for validation
- lidar_010.npz : aggregated lidar data for testing
- coords_labels.h5 : coordinates + labels of train and val
- coords_labels_test.h5 : coordinates of testing

NOTE: Generating lidar data takes time, two hours or so.

2.2 - Training the model

Training the model can be done by simply running the script beam_train_frontend.py, in fact it used the data generated from the previous step and there is no need to specify folders.

The output will be the trained weigths of the network which is named as 'trained_model', training curves are also saved.

2.3 - Testing the model

The trained model can be tested on the s010 dataset by running the beam_test_model.py script. It generates the file with the predicted labels named beam_test_pred.csv

3 - Pre-trained model and weights

The folder already contains the file trained_model.h5 with the trained network.

IF YOU HAVE ANY PROBLEM CONTACT ME AT: zecchin@eurecom.fr

import os
import csv
import numpy as np
import scipy.spatial.distance as dist
import open3d as o3d
from CSVHandler import CSVHandler
import pandas as pd


base_folder_010='E:\ML5G_challenge'
filePath=os.path.dirname(os.path.abspath(__file__))
#LOCATION OF THE FILE 'CoordVehiclesRxPerScene_s010.csv'
coord_path=base_folder_010+'/raw_data/CoordVehiclesRxPerScene_s010.csv'
#LOCATION OF THE FOLDER 's010_Blensor_rosslyn_scans_lidar' CONTAINING THE SUBFODLERS "scans_run00000", "scans_run00001"...
lidar_data_path=base_folder_010+'/raw_data/rosslyn_scans/'


def getCoordTest(filename):
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        coordinates = []
        for row in reader:
            if row['Val'] == 'V':
                coordinates.append([float(row['x']), float(row['y']), float(row['z'])])
    return coordinates



def base_run_dir_fn(i):  # the folders will be run00001, run00002, etc.
    """returns the `run_dir` for run `i`"""
    return "scans_run{:05d}".format(i)


def base_vehicle_pcd(flow):  # the folders will be run00001, run00002, etc.
    V_id = flow.replace('flow', '')
    # return 'flow{:6f}'.format(V_id)
    return 'flow{}00000'.format(V_id)


def episodes_dict(csv_path):
    with open(csv_path) as csvfile:
        reader = csv.DictReader(csvfile)
        EpisodeInMemory = -1
        SceneInMemory = -1
        episodesDict = {}
        usersDict = {}
        for row in reader:
            if str(row['Val']) == 'I':
                continue
            Valid_episode = int(row['EpisodeID'])
            Valid_Scene = int(row['SceneID'])
            Valid_Rx = base_vehicle_pcd(str(row['VehicleName']))
            key_dict = str(Valid_episode) + ',' + str(Valid_Scene)
            if EpisodeInMemory != Valid_episode:
                episodesDict[Valid_episode] = []
                usersDict[key_dict] = []
                EpisodeInMemory = Valid_episode
                SceneInMemory = -1
            if SceneInMemory != Valid_Scene:
                episodesDict[Valid_episode] = []
                SceneInMemory = Valid_Scene
                episodesDict[Valid_episode].append(Valid_Scene)
            Rx_info = [Valid_Rx, float(row['x']), float(row['y']), float(row['z']), int(row['VehicleArrayID'])]
            usersDict[key_dict].append(Rx_info)
    return episodesDict, usersDict


def main(num,max,csv_path,lidar_path):
    fileToRead = csv_path
    starting_episode = 0
    last_episode = max
    type_data = '3D'
    outputFolder = './obstacles_' + num + '/'
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    # Configuration of parameters
    dictvehicle = {1.59: 5, 3.2: 9.5, 4.3: 13}  # CarSize/BusSize/TruckSize
    # Quantization parameters
    QP = {'Xp': 1, 'Yp':1, 'Zp': 1, 'Xmax': 841, 'Ymax': 680, 'Zmax': 10, 'Xmin': 661, 'Ymin': 350,
          'Zmin': 0}  # X Y Z
    # Tx position
    Tx = [746, 560, 4]
    max_dist_LIDAR = 150  # in meters
    dx = np.arange(QP['Xmin'], QP['Xmax'], QP['Xp'])
    dy = np.arange(QP['Ymin'], QP['Ymax'], QP['Yp'])
    # initializing variables
    should_stop = False
    episodeID = int(starting_episode)
    numScenesPerEpisode = 1
    scans_path = lidar_path
    total_num_scenes = starting_episode  # all processed scenes
    # Dicts
    scenes_in_ep, RX_in_ep = episodes_dict(fileToRead)

    if type_data == '3D':
        dz = np.arange(QP['Zmin'], QP['Zmax'], QP['Zp'])
        # Assumes 10 Tx/Rx pairs per scene
        # TO-DO: Support for episodes with more than 1 scene
        zeros_array = np.zeros((10, np.size(dx), np.size(dy), np.size(dz)), np.int8)
    else:
        zeros_array = np.zeros((10, np.size(dx), np.size(dy)), np.int8)

    while not should_stop:



        if episodeID > int(last_episode):
            print('\nLast desired episode ({}) reached'.format(int(last_episode)))
            break

        for s in range(numScenesPerEpisode):
            obstacles_matrix_array = np.zeros((10, np.size(dx), np.size(dy), np.size(dz)), np.int8)
            tmpdir = './tmp/scans'
            if not os.path.exists(tmpdir):
                os.makedirs(tmpdir)
            scans_dir = scans_path + base_run_dir_fn(episodeID)
            key_dict = str(episodeID) + ',' + str(s)
            if not(key_dict in RX_in_ep):
                break
            RxFlow = RX_in_ep[key_dict]
            if not os.path.exists(scans_dir):
                print('\nWarning: could not find file ', scans_dir, ' Stopping...')
                should_stop = True
                break


            for vehicle in RxFlow:
                pcd_path = scans_dir + '/' + vehicle[0] + '.pcd'
                pcd = o3d.io.read_point_cloud(pcd_path)
                pc = np.asarray(pcd.points) #points
                ind = np.where(pc[:, 2] > 0.2)  # remove Lidar wavefronts point
                pc = pc[ind]
                vehicle_position = [[vehicle[1], vehicle[2], vehicle[3]]]
                D = dist.cdist(vehicle_position, pc.tolist(), 'euclidean')
                ind2 = np.where(D[0] < max_dist_LIDAR)  # MaxSizeLIDAR
                fffCloud = pc[ind2[0],:]
                indx = quantizeJ(fffCloud[:,0], dx)
                indx = [int(i) for i in indx]
                indy = quantizeJ(fffCloud[:,1], dy)
                indy = [int(i) for i in indy]
                Rx_q_indx = quantizeJ([vehicle[1]], dx)
                Rx_q_indy = quantizeJ([vehicle[2]], dy)
                Tx_q_indx = quantizeJ([Tx[0]], dx)
                Tx_q_indy = quantizeJ([Tx[1]], dy)
                indz = quantizeJ(fffCloud[:,2], dz)
                indz = [int(i) for i in indz]
                z = np.array(indz)
                Rx_q_indz = quantizeJ([vehicle[3]], dz)
                Tx_q_indz = quantizeJ([Tx[2]], dz)
                MD = np.zeros((np.size(dx), np.size(dy), np.size(dz)))
                # Obstacles
                MD[np.asarray(indx), np.asarray(indy), np.asarray(indz)]=1
                # Tx -1 Rx -2
                MD[int(Tx_q_indx[0]), int(Tx_q_indy[0]), int(Tx_q_indz[0])] = -1
                MD[int(Rx_q_indx[0]), int(Rx_q_indy[0]), int(Rx_q_indz[0])] = -2
                obstacles_matrix_array[int(vehicle[4]), :] = MD

            total_num_scenes += 1

        npz_name = os.path.join(outputFolder, 'obstacles_e_' + str(episodeID) + '.npz')
        #print('==> Wrote file ' + npz_name)
        np.savez_compressed(npz_name, obstacles_matrix_array=np.int8(obstacles_matrix_array))
        #print('Saved file ', npz_name)
        print("Ep"+ str(episodeID))
        episodeID += 1


def quantizeJ(signal, partitions):
    xmin = min(signal)
    xmax = max(signal)
    M = len(partitions)
    delta = partitions[2] - partitions[1]
    quantizerLevels = partitions
    xminq = min(quantizerLevels)
    xmaxq = max(quantizerLevels)
    x_i = (signal - xminq) / delta  # quantizer levels
    x_i = np.round(x_i)
    ind = np.where(x_i < 0)
    x_i[ind] = 0
    ind = np.where(x_i > (M - 1))
    x_i[ind] = M - 1;  # impose maximum
    x_q = x_i * delta + xminq;  # quantized and decoded output

    return list(x_i)

def processLidarData(num,csv_path):
    csvHand = CSVHandler()
    print('Generating LIDAR ...')
    lidarDataDir = './obstacles_' + num + '/'
    inputDataDir = './'
    coordFileName = csv_path
    coordURL =  coordFileName
    if not(os.path.exists(inputDataDir)):
        os.mkdir(inputDataDir)
        print("Directory '% s' created" % inputDataDir)
    nSamples, lastEpisode, epi_scen  = csvHand.getEpScenValbyRec(coordURL)
    obstacles_matrix_array_lidar = np.ones((nSamples,180,330,10), np.int8)
    with open(coordURL) as csvfile:
        reader = csv.DictReader(csvfile)
        id_count = 0
        alreadyInMemoryEpisode = -1
        for row in reader:
            episodeNum = int(row['EpisodeID'])
            #if (episodeNum < numEpisodeStart) | (episodeNum > numEpisodeEnd):
            #    continue #skip episodes out of the interval
            isValid = row['Val'] #V or I are the first element of the list thisLine
            if isValid == 'I':
                continue #skip invalid entries
            if episodeNum != alreadyInMemoryEpisode: #just read if a new episode
                if(episodeNum%10==0):
                    print('Reading Episode '+str(episodeNum)+' ...')
                currentEpisodesInputs = np.load(os.path.join(lidarDataDir,'obstacles_e_'+str(episodeNum)+'.npz'))
                obstacles_matrix_array = currentEpisodesInputs['obstacles_matrix_array']
                alreadyInMemoryEpisode = episodeNum #update for other iterations
            r = int(row['VehicleArrayID']) #get receiver number
            obstacles_matrix_array_lidar[id_count] = obstacles_matrix_array[r]
            id_count = id_count + 1
    lidar_inputs_train = np.int8(obstacles_matrix_array_lidar)
    #train
    np.savez_compressed(inputDataDir+'lidar_'+num+'.npz',input=lidar_inputs_train)

num='010'
print('Processing '+num)
coords=np.asarray(getCoordTest(coord_path))
data010 = {'X': coords[:,0], 'Y': coords[:,1], 'Z': coords[:,2]}
df = pd.DataFrame(data010)
df.to_hdf('coords_labels_test.h5', key='test')
main('010',10,coord_path,lidar_data_path)
processLidarData('010',coord_path)
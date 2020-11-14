import sys
import os
import csv
import shutil
import numpy as np
import scipy.spatial.distance as dist
import open3d as o3d
import matplotlib.pyplot as plt
from CSVHandler import CSVHandler
from mimo_channels import getNarrowBandULAMIMOChannel, getDFTOperatedChannel
import h5py
from math import ceil
import pandas as pd


base_folder_008='E:\ML5G_challenge'
base_folder_009='E:\ML5G_challenge'

filePath=os.path.dirname(os.path.abspath(__file__))
#LOCATION OF THE FILE 'CoordVehiclesRxPerScene_s008.csv'
coord_path=base_folder_008+'/raw_data/CoordVehiclesRxPerScene_s008.csv'
#LOCATION OF THE FOLDER 's008_Blensor_rosslyn_scans_lidar' CONTAINING THE SUBFODLERS "scans_run00000", "scans_run00001"...
lidar_data_path=base_folder_008+'/raw_data/s008_Blensor_rosslyn_scans_lidar/'
#LOCATION OF THE FOLDER 'ray_tracing_data_s008_carrier60GHz'
ray_data_path=base_folder_008+'/raw_data/ray_tracing_data_s008_carrier60GHz/'
#LOCATION OF THE FILE 'CoordVehiclesRxPerScene_s009.csv'
coord_path_val=base_folder_009+'/raw_data/CoordVehiclesRxPerScene_s009.csv'
#LOCATION OF THE FOLDER 's009_Blensor_rosslyn_scans_lidar' CONTAINING THE SUBFODLERS "scans_run00000", "scans_run00001"...
lidar_data_path_val=base_folder_009+'/raw_data/s009_Blensor_rosslyn_scans_lidar/'
#LOCATION OF THE FOLDER 'ray_tracing_data_s009_carrier60GHz'
ray_data_path_val=base_folder_009+'/raw_data/ray_tracing_data_s009_carrier60GHz/'


def getCoord(filename):
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        coordinates = []
        for row in reader:
            if row['Val'] == 'V':
                coordinates.append([float(row['x']), float(row['y']), float(row['z']), int(row['LOS'] == 'LOS=1')])
    return coordinates


def getCoordTest(filename):
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        coordinates = []
        for row in reader:
            if row['Val'] == 'V':
                coordinates.append([float(row['x']), float(row['y']), float(row['z'])])
    return coordinates


def processBeamsOutput(csv_path,raytracing_path, rep, max):
    print('Generating Beams ...')
    # file generated with ak_generateInSitePlusSumoList.py:
    # need to use both LOS and NLOS here, cannot use restricted list because script does a loop over all scenes
    insiteCSVFile = csv_path
    numEpisodes = max  # total number of episodes
    # parameters that are typically not changed
    inputPath = raytracing_path+'/rosslyn_mobile_60GHz_ts0.1s_V_Lidar_e'
    normalizedAntDistance = 0.5
    angleWithArrayNormal = 0  # use 0 when the angles are provided by InSite

    # ULA
    number_Tx_antennas = 32
    number_Rx_antennas = 8

    # initialize variables
    numOfValidChannels = 0
    numOfInvalidChannels = 0
    numLOS = 0
    numNLOS = 0
    count = 0

    '''use dictionary taking the episode, scene and Rx number of file with rows e.g.:
    0,0,0,flow11.0,Car,753.83094753535,649.05232524135,1.59,D:/insitedata/noOverlappingTx5m/run00000,LOS=0
    0,0,2,flow2.0,Car,753.8198286576,507.38595866735,1.59,D:/insitedata/noOverlappingTx5m/run00000,LOS=1
    0,0,3,flow2.1,Car,749.7071175056,566.1905128583,1.59,D:/insitedata/noOverlappingTx5m/run00000,LOS=1'''

    with open(insiteCSVFile, 'r') as f:
        insiteReader = csv.DictReader(f)
        insiteDictionary = {}
        numExamples = 0
        for row in insiteReader:
            isValid = row['Val']  # V or I are the first element of the list thisLine
            if isValid == 'V':  # filter the valid channels
                numExamples += 1
                thisKey = str(row['EpisodeID']) + ',' + str(row['SceneID']) + ',' + str(row['VehicleArrayID'])
                insiteDictionary[thisKey] = row
        lastEpisode = int(row['EpisodeID'])
    allOutputs = np.nan * np.ones((numExamples, number_Rx_antennas * number_Tx_antennas), np.float32)

    for e in range(numEpisodes):
        # print("Episode # ", e)
        b = h5py.File(inputPath + str(e) + '.hdf5', 'r')
        allEpisodeData = b.get('allEpisodeData')
        numScenes = allEpisodeData.shape[0]
        numReceivers = allEpisodeData.shape[1]
        # store the position (x,y,z), 4 angles of strongest (first) ray and LOS or not
        receiverPositions = np.nan * np.ones((numScenes, numReceivers, 8), np.float32)
        # store two integers converted to 1
        episodeOutputs = np.nan * np.ones((numScenes, numReceivers, number_Rx_antennas, number_Tx_antennas),
                                          np.float32)

        if (e % 50 == 0):
            print(e / numEpisodes)
        for s in range(numScenes):
            for r in range(numReceivers):  # 1
                insiteData = allEpisodeData[s, r, :, :]
                # if insiteData corresponds to an invalid channel, all its values will be NaN.
                # We check for that below
                numNaNsInThisChannel = sum(np.isnan(insiteData.flatten()))
                if numNaNsInThisChannel == np.prod(insiteData.shape):
                    numOfInvalidChannels += 1
                    continue  # next Tx / Rx pair
                thisKey = str(e) + ',' + str(s) + ',' + str(r)

                try:
                    thisInSiteLine = list(insiteDictionary[thisKey].items())  # recover from dic
                except KeyError:
                    print('Could not find in dictionary the key: ', thisKey)
                    print('Verify file', insiteCSVFile)
                    exit(-1)
                # tokens = thisInSiteLine.split(',')
                if numNaNsInThisChannel > 0:
                    numOfValidRays = int(thisInSiteLine[8][1])  # number of rays is in 9-th position in CSV list
                    # I could simply use
                    # insiteData = insiteData[0:numOfValidRays]
                    # given the NaN are in the last rows, but to be safe given that did not check, I will go for a slower solution
                    insiteDataTemp = np.zeros((numOfValidRays, insiteData.shape[1]))
                    numMaxRays = insiteData.shape[0]
                    validRayCounter = 0
                    for itemp in range(numMaxRays):
                        if sum(np.isnan(insiteData[itemp].flatten())) == 1:  # if insite version 3.2, else use 0
                            insiteDataTemp[validRayCounter] = insiteData[itemp]
                            validRayCounter += 1
                    insiteData = insiteDataTemp  # replace by smaller array without NaN
                receiverPositions[s, r, 0:3] = np.array(
                    [thisInSiteLine[5][1], thisInSiteLine[6][1], thisInSiteLine[7][1]])

                numOfValidChannels += 1
                gain_in_dB = insiteData[:, 0]
                timeOfArrival = insiteData[:, 1]
                # InSite provides angles in degrees. Convert to radians
                # This conversion is being done within the channel function
                AoD_el = insiteData[:, 2]
                AoD_az = insiteData[:, 3]
                AoA_el = insiteData[:, 4]
                AoA_az = insiteData[:, 5]
                RxAngle = insiteData[:, 8][0]
                RxAngle = RxAngle + 90.0
                if RxAngle > 360.0:
                    RxAngle = RxAngle - 360.0
                # Correct ULA with Rx orientation
                AoA_az = - RxAngle + AoA_az  # angle_new = - delta_axis + angle_wi;

                # first ray is the strongest, store its angles
                receiverPositions[s, r, 3] = AoD_el[0]
                receiverPositions[s, r, 4] = AoD_az[0]
                receiverPositions[s, r, 5] = AoA_el[0]
                receiverPositions[s, r, 6] = AoA_az[0]

                isLOSperRay = insiteData[:, 6]
                pathPhases = insiteData[:, 7]

                # in case any of the rays in LOS, then indicate that the output is 1
                isLOS = 0  # for the channel
                if np.sum(isLOSperRay) > 0:
                    isLOS = 1
                    numLOS += 1
                else:
                    numNLOS += 1
                receiverPositions[s, r, 7] = isLOS
                mimoChannel = getNarrowBandULAMIMOChannel(AoD_az, AoA_az, gain_in_dB, number_Tx_antennas,
                                                          number_Rx_antennas, normalizedAntDistance,
                                                          angleWithArrayNormal)
                equivalentChannel = np.abs(getDFTOperatedChannel(mimoChannel, number_Tx_antennas, number_Rx_antennas))
                for rpt in range(0, rep):
                    mimoChannel = getNarrowBandULAMIMOChannel(AoD_az, AoA_az, gain_in_dB, number_Tx_antennas,
                                                              number_Rx_antennas, normalizedAntDistance,
                                                              angleWithArrayNormal)
                    equivalentChannel = equivalentChannel + np.abs(
                        getDFTOperatedChannel(mimoChannel, number_Tx_antennas, number_Rx_antennas))
                episodeOutputs[s, r] = np.abs(equivalentChannel) / rep
                allOutputs[count] = np.ndarray.flatten(episodeOutputs[s, r])
                count += 1
            # finished processing this episode
    return allOutputs


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


num='008'
print('Processing '+num)
filePath=os.path.dirname(os.path.abspath(__file__)) #Absolute path, the script should be in the directory contaning "CoordVehiclesRxPerScene_s008.csv" and the raytracing data folder
coords=np.asarray(getCoord(coord_path))
beams=processBeamsOutput(coord_path,ray_data_path,5,2086)
p=beams.tolist()
data008 = {'X': coords[:,0], 'Y': coords[:,1], 'Z': coords[:,2], 'LOS': coords[:,3],'Labels': beams.tolist()}
df= pd.DataFrame(data008)
df.to_hdf('coords_labels.h5', key='train')


num='009'
print('Processing '+num)
filePath=os.path.dirname(os.path.abspath(__file__)) #Absolute path, the script should be in the directory contaning "CoordVehiclesRxPerScene_s008.csv" and the raytracing data folder
coords=np.asarray(getCoord(coord_path_val))
beams=processBeamsOutput(coord_path_val,ray_data_path_val,5,2000)
p=beams.tolist()
data009 = {'X': coords[:,0], 'Y': coords[:,1], 'Z': coords[:,2], 'LOS': coords[:,3],'Labels': beams.tolist()}
df = pd.DataFrame(data009)
df.to_hdf('coords_labels.h5', key='val')

main('008',2086,coord_path,lidar_data_path)
processLidarData('008',coord_path)

main('009',2000,coord_path_val,lidar_data_path_val)
processLidarData('009',coord_path_val)


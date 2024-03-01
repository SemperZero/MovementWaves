import os
import sys
import numpy as np
#import plotly
import pandas as pd
import math
import plotly.express as px
import plotly.graph_objs as go
from flextail_python_reconstruction import calc_reconstruction
import time
from openTSNE import TSNE as fastTSNE
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import webbrowser
from multiprocessing import Process
from hover_data import *
import cv2
import plotly.offline as pyo
import multiprocessing
from scipy.interpolate import griddata
from scipy import stats
from sklearn.preprocessing import LabelEncoder



os.chdir(r"D:\programming\MovementWaves")
print(os.getcwd())
print(os.listdir("."))
port = 8502
acts = []
def get_activity_name(file_name):
    # count_ = 0
    # i = 0
    # while count_<5:
    #     if file_name[i] == "_":
    #         count_+=1
    #     i+=1
    # activity=file_name[i:].split("_")[0]
    # print(activity, file_name.split("_")[5])
    # exit()
    return file_name.split("_")[5]

def get_personid(file_name):
    # count_ = 0
    # i = 0
    # while count_<3:
    #     if file_name[i] == "_":
    #         count_+=1
    #     i+=1
    # p_id = file_name[:i-1]
    return "_".join(file_name.split("_")[1:3])

#@jit(nopython = True)
def rotate(x, y, angle):
    sin = math.sin(angle)
    cos = math.cos(angle)
    ret_x = x * cos - y * sin
    ret_y = x * sin + y * cos
    return ret_x, ret_y


def get_2d_reconstruct_image(x, y):
    
    #x,y = coords_y, coords_z
    fig = go.FigureWidget()
    fig.add_trace(go.Scatter(x = x, y=y, mode='lines', opacity=1, line={'width': 2 }))
    fig.update_layout(width=300, height=400, xaxis_range=[-10,10])
    #fig.show()      
   # fig.show()
    return fig.to_image(format = "png")

def open_browser(port):
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        webbrowser.open_new('http://127.0.0.1:%d/'%port)
    

def run_plot(fig, hover_clusters, cluster_labels, hover_width, text_labels, hover_indiv, callback_type, port):
    hover_figure = DisplayWithHover(fig, hover_clusters, cluster_labels, hover_width, text_labels, hover_indiv, callback_type)
    print(port)
    open_browser(port)
    hover_figure.run_server(port = port, debug=False)


def get_3d_reconstruct_image(alpha, beta, coords_x, coords_y, coords_z):
    fig = px.scatter_3d(x=coords_x, y=coords_y, z=coords_z, opacity=0.8, color_continuous_scale='magma')
    fig.update_traces(marker_size = 3)
    fig.show()
    #x,y = rotate(coords_x,coords_z, pitch * math.pi/180)
    fig = go.FigureWidget()

    fig.add_trace(go.Scatter(x = list(range(len(alpha))), y=alpha, mode='lines', opacity=1, line={'width': 1 }, name = "alpha"))
    fig.add_trace(go.Scatter(x = list(range(len(beta))), y=beta, mode='lines', opacity=1, line={'width': 1 }, name = "beta"))

    fig.update_layout(xaxis_range=[-10,10])
    fig.update_layout(
        width=600, height=800     
    )
    fig.show()

def get_heatmap_plot(heatmap_data, size = 18, colorbar_x = None, max_scale_value = None):
        max_value = 50
        if max_scale_value:
            zmax = max_scale_value
        else:
            zmax = None

        heatmap_trace = go.Heatmap(
            z=heatmap_data,
            x=list(range(-max_value, max_value)),
            y=["V"+str(i) for i in range(size)],
            colorscale='ice',
            colorbar=dict(title='Value Density'),
            xgap=0, ygap=0,
            colorbar_x = colorbar_x,
            zmin=0, 
            zmax=zmax

        )

        return heatmap_trace

def video_one_folder(folder, fps):
    imgs = os.listdir(folder)

    first_img = Image.open(f"{folder}/{imgs[0]}")

    nr_frames_per_second = fps
    print(first_img.width, first_img.height)
    out_video_color = cv2.VideoWriter(f"{folder}/0result.mp4", cv2.VideoWriter_fourcc('V','X','I','D'), nr_frames_per_second, (first_img.width-100, first_img.height))

    for img_file in sorted([img for img in imgs if ".png" in img], key = lambda x : float(x.split("_")[-1].split(".")[0]), reverse = False):
 
        print(img_file)
        img = Image.open(f"{folder}/{img_file}")#.crop(crop_coord)
        #img = cv2.imread(os.path.join(folder, img_file))
        img = img.crop((0, 0, img.width - 100, img.height))
        np_image = np.array(img)
        
        #cv_image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        img_bgr_array = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

        out_video_color.write(img_bgr_array)

    out_video_color.release()


def stack_raw_data():
    
    activities_data = {}
    rec_data = {}
    count = 0
    for data_file in os.listdir("activies_of_daily_living_data"):
        #print(count, data_file)
        count+=1
        file_name, extension = data_file.split(".")
        if extension == "arrow":
            continue
        activity = get_activity_name(file_name)
        #print(activity)
        if activity not in activities_data:
            activities_data[activity] = []
            rec_data[activity] = {"rec_x":[], "rec_y":[]}


        d = pd.read_parquet(f"activies_of_daily_living_data\\{data_file}", engine='pyarrow')
        corrutped = 0
        for row in d.itertuples():
            alpha, beta, coords_x, coords_y, coords_z, case_orientation = calc_reconstruction(row.left, row.right, row.acc)

            pitch, roll = case_orientation

            x, y = rotate(np.array(coords_y), np.array(coords_z), pitch * math.pi/180)
            train_row = beta
            train_row.append(x[-1])
            train_row.append(y[-1])
            train_row+=list(row.gyro)
            if len(train_row) !=23:
                corrutped+=1
                continue
            
            activities_data[activity].append(np.array(train_row))

            rec_data[activity]["rec_x"].append(x)
            rec_data[activity]["rec_y"].append(y)
        if(corrutped!=0):
            print("file", file_name,"corrupted:", corrutped)
        
        
    for a in activities_data:
        train_matrix = np.concatenate(activities_data[a]).reshape(-1, 23)
        np.save(f"processed_data\\train_data\\merged_{a}.npy", train_matrix)
       
        rec_df_x = pd.DataFrame.from_dict(rec_data[a])
        rec_df_x.to_parquet(f"processed_data\\reconstructed_data\\reconstructed_{a}.parquet")

def parse_raw_data():
    #THE MAIN FUNCTION FOR THE FIRST PROCESS OF THE RAW DATASET
    
    activities_data = {}
    count = 0
    for data_file in os.listdir("activies_of_daily_living_data"):
       # print(count, data_file)
        count+=1
        file_name, extension = data_file.split(".")
        if extension == "arrow":
            continue
        activity = get_activity_name(file_name)
        p_id = get_personid(file_name)
      
        
        if "_5Hz_" in data_file:
            #print(data_file)
            #exit()
            refresh_rate = "5Hz"
        elif "_15Hz_" in data_file:
            refresh_rate = "15Hz"
        
        else:
            print("ERROR, unrecognized refresh rate")
            print(data_file)
            exit()

        split_name = activity + "_" + refresh_rate


        
        if split_name not in activities_data:
            activities_data[split_name] = { "left":[], "right":[], "acc":[] ,"alpha":[], "beta":[], "rec_x":[], "rec_y":[], "rec_z":[], "back_x":[], "back_y":[], "side_x":[], "side_y":[], "case_orientation":[], "pitch": [], "roll": [], "gyro": [], "p_id":[] }


        d = pd.read_parquet(f"activies_of_daily_living_data\\{data_file}", engine='pyarrow')
        for row in d.itertuples():
            alpha, beta, coords_x, coords_y, coords_z, case_orientation = calc_reconstruction(row.left, row.right, row.acc)

            pitch, roll = case_orientation

            back_x, back_y = rotate(np.array(coords_y), np.array(coords_z), pitch * math.pi/180)
            side_x, side_y = rotate(np.array(coords_x), np.array(coords_z), pitch * math.pi/180)

          
            activities_data[split_name]["left"].append(row.left)
            activities_data[split_name]["right"].append(row.right)
            activities_data[split_name]["acc"].append(row.acc)
            activities_data[split_name]["alpha"].append(alpha)
            activities_data[split_name]["beta"].append(beta)
            activities_data[split_name]["rec_x"].append(coords_x)
            activities_data[split_name]["rec_y"].append(coords_y)
            activities_data[split_name]["rec_z"].append(coords_z)
            activities_data[split_name]["back_x"].append(back_x)
            activities_data[split_name]["back_y"].append(back_y)
            activities_data[split_name]["side_x"].append(side_x)
            activities_data[split_name]["side_y"].append(side_y)
            activities_data[split_name]["case_orientation"].append(case_orientation)
            activities_data[split_name]["pitch"].append(pitch)
            activities_data[split_name]["roll"].append(roll)
            activities_data[split_name]["gyro"].append(row.gyro)
            activities_data[split_name]["p_id"].append(p_id)


    for a in activities_data:
        p = pd.DataFrame(activities_data[a])
        activity = a.split("_")[0]
        refresh_rate = a.split("_")[1]

        folder_p = f"processed_data\\train_data\\train_data_{refresh_rate}"
        if not os.path.isdir(folder_p):
            os.makedirs(folder_p)

        p.to_parquet(f"{folder_p}\\{activity}.parquet")
        
        
    # for a in activities_data:
    #     train_matrix = np.concatenate(activities_data[a]).reshape(-1, 23)
    #     np.save(f"processed_data\\train_data\\merged_{a}.npy", train_matrix)
       
    #     rec_df_x = pd.DataFrame.from_dict(rec_data[a])
    #     rec_df_x.to_parquet(f"processed_data\\reconstructed_data\\reconstructed_{a}.parquet")

def parse_raw_data_individual():
    #THE MAIN FUNCTION FOR THE FIRST PROCESS OF THE RAW DATASET
    
    activities_data = {}
    count = 0
    for data_file in os.listdir("activies_of_daily_living_data"):
       # print(count, data_file)
        count+=1
        file_name, extension = data_file.split(".")
        if extension == "arrow":
            continue
        activity = get_activity_name(file_name)
        p_id = get_personid(file_name)
       # print(p_id)
       # exit()
        
        if "_5Hz_" in data_file:
            #print(data_file)
            #exit()
            refresh_rate = "5Hz"
        elif "_15Hz_" in data_file:
            refresh_rate = "15Hz"
        
        else:
            print("ERROR, unrecognized refresh rate")
            print(data_file)
            exit()

        split_name = activity + "_" + p_id + "_" + refresh_rate


        
        if split_name not in activities_data:
            activities_data[split_name] = { "left":[], "right":[], "acc":[] ,"alpha":[], "beta":[], "rec_x":[], "rec_y":[], "rec_z":[], "back_x":[], "back_y":[], "side_x":[], "side_y":[], "case_orientation":[], "pitch": [], "roll": [], "gyro": [], "p_id":[] }


        d = pd.read_parquet(f"activies_of_daily_living_data\\{data_file}", engine='pyarrow')
        for row in d.itertuples():
            alpha, beta, coords_x, coords_y, coords_z, case_orientation = calc_reconstruction(row.left, row.right, row.acc)

            pitch, roll = case_orientation

            back_x, back_y = rotate(np.array(coords_y), np.array(coords_z), pitch * math.pi/180)
            side_x, side_y = rotate(np.array(coords_x), np.array(coords_z), pitch * math.pi/180)

          
            activities_data[split_name]["left"].append(row.left)
            activities_data[split_name]["right"].append(row.right)
            activities_data[split_name]["acc"].append(row.acc)
            activities_data[split_name]["alpha"].append(alpha)
            activities_data[split_name]["beta"].append(beta)
            activities_data[split_name]["rec_x"].append(coords_x)
            activities_data[split_name]["rec_y"].append(coords_y)
            activities_data[split_name]["rec_z"].append(coords_z)
            activities_data[split_name]["back_x"].append(back_x)
            activities_data[split_name]["back_y"].append(back_y)
            activities_data[split_name]["side_x"].append(side_x)
            activities_data[split_name]["side_y"].append(side_y)
            activities_data[split_name]["case_orientation"].append(case_orientation)
            activities_data[split_name]["pitch"].append(pitch)
            activities_data[split_name]["roll"].append(roll)
            activities_data[split_name]["gyro"].append(row.gyro)
            activities_data[split_name]["p_id"].append(p_id)


    for a in activities_data:
        p = pd.DataFrame(activities_data[a])
        activity = a.split("_")[0]
        p_id = "_".join(a.split("_")[1:])

        folder_p = f"processed_data\\train_data\\train_data_individual\\{p_id}"
        if not os.path.isdir(folder_p):
            os.makedirs(folder_p)

        p.to_parquet(f"{folder_p}\\{activity}.parquet")
        
        
    # for a in activities_data:
    #     train_matrix = np.concatenate(activities_data[a]).reshape(-1, 23)
    #     np.save(f"processed_data\\train_data\\merged_{a}.npy", train_matrix)
       
    #     rec_df_x = pd.DataFrame.from_dict(rec_data[a])
    #     rec_df_x.to_parquet(f"processed_data\\reconstructed_data\\reconstructed_{a}.parquet")

def compute_hover_reconstructed_clusters(rec_x_vals, rec_y_vals, cluster_labels, reconstruct_individual = False):
    hover_reconstruct_individual = {}
    
    max_y = 300
    max_x = 10
    scale_y = 0.05# make y have only 60 buckets
    scale_x = 5#make x have 23*5 buckets
    heatmap_data_all = {i : np.zeros((int(max_y*scale_y), max_x*scale_x*2)) for i in np.unique(cluster_labels)}
    i=0
    for rec_x, rec_y in zip(rec_x_vals, rec_y_vals):
        #if i % 5!=0:
        #    continue

        if reconstruct_individual:
            hover_reconstruct_individual[i] = get_2d_reconstruct_image(rec_x, rec_y)

        for x,y in zip(rec_x, rec_y):
            if y>0 and y < 300 and x>-max_x and x<max_x:
                cluster_label = cluster_labels[i]
                squished_y = int(y*scale_y)
                squished_x = int((x+max_x)*scale_x)
                heatmap_data_all[cluster_label][squished_y][squished_x]+=1
        i+=1

    cluster_hover_images = {}
    for label in np.unique(cluster_labels):
        # print(heatmap_data_all[label])
        
        
       # def get_heatmap_plot(heatmap_data, size = 18, colorbar_x = None, max_scale_value = None):
        size = int(max_y*scale_y)
        max_value = 50
        heatmap_plot = go.Heatmap(
            z=heatmap_data_all[label],
            x=list(range(-max_value, max_value)),
            y=["V"+str(i) for i in range(size)],
            colorscale='ice',
            colorbar=dict(title='Value Density'),
            xgap=0, ygap=0,
            colorbar_x = max_x,
            zmin=0, 
            zmax=None

        )
        heatmap_plot.update(showscale=False)

        heatmap_fig = go.Figure()
        heatmap_fig.add_trace(heatmap_plot)
        
        heatmap_fig.update_layout(width=300, height=400)#, title = {'text': "Cluster NR %d"%label, 'font': { 'family': 'Times New Roman', 'size': 15}} )

        cluster_hover_images[label] = heatmap_fig.to_image(format = "png")

    return cluster_hover_images, hover_reconstruct_individual


def compute_hover_reconstructed_interval(rec_x_vals, rec_y_vals):
    
    max_y = 400
    max_x = 300
    scale_y = 0.05# make y have only 60 buckets
    scale_x = 0.05
    heatmap_data_all = np.zeros((int(max_y*scale_y), int(max_x*scale_x*2)))
    i=0
    for rec_x, rec_y in zip(rec_x_vals, rec_y_vals):

        for x,y in zip(rec_x, rec_y):
            if y>0 and y < max_y and x>-max_x and x<max_x:
                squished_y = int(y*scale_y)
                squished_x = int((x+max_x)*scale_x)
                heatmap_data_all[squished_y][squished_x]+=1
        i+=1


    size = int(max_y*scale_y)
    max_value = max_x
    heatmap_plot = go.Heatmap(
        z=heatmap_data_all,
        x=list(range(-max_value, max_value)),
        y=["V"+str(i) for i in range(size)],
        colorscale='ice',
        colorbar=dict(title='Value Density'),
        xgap=0, ygap=0,
        colorbar_x = max_x,
        zmin=0, 
        zmax=None

    )
    heatmap_plot.update(showscale=False)

    heatmap_fig = go.Figure()
    heatmap_fig.add_trace(heatmap_plot)
    
    heatmap_fig.update_layout(width=800, height=1000)#, title = {'text': "Cluster NR %d"%label, 'font': { 'family': 'Times New Roman', 'size': 15}} )
   
    xvals=list(range(-max_value, max_value))
    yals=["V"+str(i) for i in range(size)]

    heatmap_fig.add_annotation(
            x=xvals[8],
            y=yals[16],
            text="back position",
            showarrow=False,
            font=dict(color='white', size=14)
        )
    
    return heatmap_fig.to_image(format = "png")


        
        


def cluster_beta_data(stacked_data):
    kmeans = KMeans(n_clusters=10, n_init = 1, max_iter = 300)
    kmeans.fit(stacked_data)
    return kmeans.labels_


def process_images(png_images):
    #print(len(png_images))
    imgs = [Image.open(io.BytesIO(png_img)) for png_img in png_images if png_img is not None]
    
    if len(imgs) == 0:
        return None, None

    border_size = 50
    crop_top = 100

    imgs = [img.crop((border_size, crop_top, img.width - border_size, img.height - border_size)) for img in imgs]
    
    width = sum([img.width for img in imgs])
    height = max([img.height for img in imgs])

    concatenated_image = Image.new('RGBA', (width, height))
    current_width = 0
    for img in imgs:
        concatenated_image.paste(img, (current_width, 0))
        current_width+=img.width

    buffer = io.BytesIO()
    
    #concatenated_image.save(os.path.join(folder_path, "concatenated%d.png"%label))
    #concatenated_image.save(buffer, format="png")
    
    return buffer.getvalue(), current_width #need to return

def concat_images(png_images, crop0, crop1):
    #print(len(png_images))
    imgs = [Image.open(io.BytesIO(png_img)) for png_img in png_images if png_img is not None]
    imgs[0] = imgs[0].crop((crop0[0],crop0[1],imgs[0].width-crop0[2], imgs[0].height-crop0[3]))
    imgs[1] = imgs[1].crop((crop1[0],crop1[1],imgs[1].width-crop1[2], imgs[1].height-crop1[3]))


    if len(imgs) == 0:
        return None, None


    #imgs = [img.crop((border_size, crop_top, img.width - border_size, img.height - border_size)) for img in imgs]
    
    width = sum([img.width for img in imgs])
    height = max([img.height for img in imgs])

    concatenated_image = Image.new('RGBA', (width, height))
    concatenated_image.paste( (0,0,0), (0, 0, width, height))

    current_width = 0
    for img in imgs:
        concatenated_image.paste(img, (current_width, 0))
        current_width+=img.width

    #buffer = io.BytesIO()
    
    #concatenated_image.save(os.path.join(folder_path, "concatenated%d.png"%label))
    #concatenated_image.save(buffer, format="png")
    
    return concatenated_image, current_width #need to return

def concat_images_vertical(imgs, crop0, crop1):

    imgs[0] = imgs[0].crop((crop0[0],crop0[1],imgs[0].width-crop0[2], imgs[0].height-crop0[3]))
    imgs[1] = imgs[1].crop((crop1[0],crop1[1],imgs[1].width-crop1[2], imgs[1].height-crop1[3]))


    if len(imgs) == 0:
        return None, None

    
    width = max([img.width for img in imgs])
    height = sum([img.height for img in imgs])

    concatenated_image = Image.new('RGBA', (width, height))
    concatenated_image.paste( (0,0,0), (0, 0, width, height))

    current_height = 0
    for img in imgs:
        concatenated_image.paste(img, (0, current_height))
        current_height+=img.height

    #buffer = io.BytesIO()
    
    #concatenated_image.save(os.path.join(folder_path, "concatenated%d.png"%label))
    #concatenated_image.save(buffer, format="png")
    
    return concatenated_image, current_height #need to return


def analyze_patterns_solo_rec():
    global port
    DATA_SIZE = 2300
    for i, file_name in enumerate(os.listdir(r"processed_data\train_data")):
        labels = []
        big_matrix = np.load(f"processed_data\\train_data/{file_name}")[:DATA_SIZE]
        activity_name = file_name.split("_")[-1].split(".")[0]
        labels = range(len(big_matrix))
        rec_df = pd.read_parquet(f"processed_data\\reconstructed_data\\reconstructed_{activity_name}.parquet", engine='pyarrow')[:DATA_SIZE]
        
        cluster_labels = cluster_beta_data(big_matrix)
        cluster_hover_images, _ = compute_hover_reconstructed_clusters(rec_df["rec_x"].values, rec_df["rec_y"].values, cluster_labels, reconstruct_individual = False)

        print(big_matrix.shape)
        fast_tsne = fastTSNE(n_components = 2, n_jobs = 4, random_state = 42, perplexity = 23)
        tsne_transform = fast_tsne.fit(big_matrix)
        color_range = labels
        
        fig = px.scatter(x=tsne_transform[:, 0], y=tsne_transform[:, 1], opacity=1, color = color_range, color_continuous_scale="magma")
        fig.update_traces(marker_size = 3)
        fig.update_layout(
            title=f"{activity_name}",
            xaxis_title="x",
            yaxis_title="y",
            width=1000, height=1000,
            template =  "plotly_dark",

        )
        fig.update_xaxes(range=[-130,130])
        fig.update_yaxes(range=[-130,130])
       
        #hover_clusters = {label : None for label in np.unique(cluster_labels)}
        hover_width = {label : 50 for label in np.unique(cluster_labels)}


        text_labels = labels
        hover_indiv = {}
        for label in np.unique(cluster_labels):
            cluster_hover_images[label], hover_width[label] = process_images([cluster_hover_images[label]])
        
        process = Process(target=run_plot, args = [fig, cluster_hover_images, cluster_labels, hover_width, text_labels, hover_indiv, "scatter", port])
        process.start()
        port+=1
        fig.show()




def analyze_patterns_global_rec():
    global port
    DATA_START = 1000
    DATA_END = 2000

    big_matrix = []
    big_reconstructed = pd.DataFrame()
    rec_df = []
    labels = []
    activities = []
    for i, file_name in enumerate(os.listdir("processed_data\\train_data")):
        print(i, file_name)
        saved_data = np.load(f"processed_data\\train_data\\{file_name}")[DATA_START:DATA_END]
        big_matrix.append(saved_data)
        activity_name = file_name.split("_")[-1].split(".")[0]
        activities.append(activity_name)
        labels += [activity_name]*len(saved_data)
        rec_df = pd.read_parquet(f"processed_data\\reconstructed_data\\reconstructed_{activity_name}.parquet", engine='pyarrow')[DATA_START:DATA_END]
        big_reconstructed = pd.concat([big_reconstructed, rec_df], axis=0)
    
    labels_df = pd.DataFrame()
    labels_df['l']=np.array(labels)
    print(labels_df['l'].value_counts())

    big_matrix = np.concatenate(big_matrix).reshape(-1,23)

    cluster_labels = cluster_beta_data(big_matrix)
    cluster_hover_images, _ = compute_hover_reconstructed_clusters(big_reconstructed["rec_x"].values, big_reconstructed["rec_y"].values, cluster_labels, reconstruct_individual = False)


    print(big_matrix.shape)
    fast_tsne = fastTSNE(n_components = 2, n_jobs = 4, random_state = 42, perplexity = 23)
    tsne_transform = fast_tsne.fit(big_matrix)
    print(tsne_transform)

    color_range = labels
    print("leng data", len(tsne_transform[:, 0]))
    fig = px.scatter(x=tsne_transform[:, 0], y=tsne_transform[:, 1], opacity=1, color = range(len(labels)), color_continuous_scale="magma")
    fig.update_traces(marker_size = 3)
    fig.update_layout(
        title=f"all_activities",
        xaxis_title="x",
        yaxis_title="y",
        width=1000, height=1000,
        template = "plotly_dark",

    )
    fig.update_xaxes(range=[-130,130])
    fig.update_yaxes(range=[-130,130])
    #fig.show()
   # time.sleep(1000)
    hover_width = {label : 50 for label in np.unique(cluster_labels)}


    text_labels = {i: label for i, label in enumerate(labels)}
    
    hover_indiv = {}
    for label in np.unique(cluster_labels):
        cluster_hover_images[label], hover_width[label] = process_images([cluster_hover_images[label]])
   # print(text_labels)
    print("labelslen", len(labels))
    process = Process(target=run_plot, args = [fig, cluster_hover_images, cluster_labels, hover_width, text_labels, hover_indiv, "scatter", port])
    process.start()
    port+=1
    fig.show()





def analyze_patterns_global():
    DATA_START = 0
    DATA_SIZE = 100000
    DATA_END = DATA_START + DATA_SIZE
    global_data = pd.DataFrame()
    labels = []
    activities = []
    for i, file_name in enumerate(os.listdir("processed_data\\train_data")):
        print(i, file_name)
        activity_name = file_name.split(".")[0]
        activities.append(activity_name)
        activity_data = pd.read_parquet(f"processed_data\\train_data\\{file_name}")[DATA_START:DATA_END]
        labels += [activity_name]*min(len(activity_data), DATA_SIZE)
        global_data = pd.concat([global_data, activity_data], axis=0)
    
    
    global_data['activity']=np.array(labels)
    print(global_data['activity'].value_counts())

    big_matrix, rec_x, rec_y = stack_data(global_data)

    print(big_matrix.shape)
    fast_tsne = fastTSNE(n_components = 2, n_jobs = 4, random_state = 42, perplexity = 23)
    tsne_transform = fast_tsne.fit(big_matrix)
    print(tsne_transform)

    color_range = labels
    print("leng data", len(tsne_transform[:, 0]))
    fig = px.scatter(x=tsne_transform[:, 0], y=tsne_transform[:, 1], opacity=1, color = color_range, color_continuous_scale="rainbow")
    fig.update_traces(marker_size = 3)
    fig.update_layout(
        title=f"global: acc, gyro",
        xaxis_title="x",
        yaxis_title="y",
        width=1000, height=1000,
        template = "plotly_dark",

    )
    fig.update_xaxes(range=[-130,130])
    fig.update_yaxes(range=[-130,130])
    fig.show()





def plot_broken_down():
    DATA_SIZE = 300
    activity_waves = break_down_data(DATA_SIZE)
    broken_down_pd = activity_waves["cycling"]
    activity_name = "cycling"
    fig = go.FigureWidget()
    for col in broken_down_pd.iteritems():
        if "alpha_" in col[0] :
            values = col[1].values
            fft_result = np.fft.fft(values)
            num_components = 100
            

            fft_result[num_components+1:] = 0
            approximated_signal = np.fft.ifft(fft_result)

            amplitudes = np.abs(fft_result[:num_components])
            phases = np.angle(fft_result[:num_components])

            # Print the parameters of the sine and cosine components
            for i, (amplitude, phase) in enumerate(zip(amplitudes, phases)):
                print(f"Component {i+1}: Amplitude = {amplitude:.2f}, Phase = {phase:.2f} radians")

            # print(approximated_signal)

            
            fig.add_trace(go.Scatter( y=values, name = col[0], mode='lines', opacity=1, line={'width': 1 }))
            fig.add_trace(go.Scatter( y=approximated_signal.real, name = f"{col[0]}_fft", mode='lines', opacity=1, line={'width': 1 }))

    fig.update_xaxes(title_text="time")
    fig.update_yaxes(title_text="values")
    fig.update_layout(title_text=activity_name, title_x=0.5, template = "plotly_dark")    
    fig.show()

def plot_moving_dataset():
    DATA_SIZE = 3000
    activity_waves = break_down_data(DATA_SIZE)
    broken_down_pd = activity_waves["cycling"]
    activity_name = "cycling"
    
    nr_points_displayed = 50
    nr_total_frames = 600
    #1000 to scale to 
    x1 = (np.linspace(0, 1, nr_total_frames//2)/2)**2 * 1000
    x2 = (2-(np.linspace(0, 1, nr_total_frames//2) -1 )**2) * 1000

    x = np.concatenate([x1, x2])
    #print(len(x))
    #exit()
    frame_nr = 0


    for k in x:
        k = int(k)
        fig = go.FigureWidget()
        for col in broken_down_pd.iteritems():
            if "alpha_" in col[0] :
                values = col[1].values

                
                fig.add_trace(go.Scatter( y=values[k:k+nr_points_displayed], name = col[0], mode='lines', opacity=1, line={'width': 1 }))


        fig.update_xaxes(title_text="time")
        fig.update_yaxes(title_text="values")
        fig.update_layout(title_text=activity_name, title_x=0.5, template = "plotly_dark", width = 2000, height = 1500,
                             # xaxis=dict(range=[0, max_x]),
                              yaxis=dict(range=[-0.07, 0.07])
            )    
        #fig.show()
        fig.write_image(f"moving_waves\\frame_{frame_nr:04d}.png")
        frame_nr+=1

def moving_dots_on_line():
    DATA_SIZE = 2000
    DATA_START = 1000
    activity_waves = read_saved_broken_down(DATA_START, DATA_SIZE)# break_down_data(DATA_SIZE, DATA_START = 5000)
    activity_name = "cycling"
    broken_down_pd = activity_waves[activity_name]
    
    nr_points_displayed = 200
    frame_nr = 0
    
    WIDTH = 2000
    HEIGHT = 1500


    image_folder = r'D:\programming\MovementWaves\animations\moving_dots2'

    video_name = f'{image_folder}\\output_video4.mp4'

    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, fps, (WIDTH, HEIGHT))


    for k in range(nr_points_displayed):
        fig = go.FigureWidget()
        dots = []
        for col in broken_down_pd.iteritems():
            if "alpha_" in col[0]:
                values = col[1].values
                #fig.add_trace(go.Scatter( y=values, name = col[0], mode='lines', opacity=1, line={'width': 1 }))
                #dots.append(values[k])
                
                fft_result = np.fft.fft(values)
                num_components = 100
                fft_result[num_components+1:] = 0
                approximated_signal = np.fft.ifft(fft_result)
               # amplitudes = np.abs(fft_result[:num_components])
                #phases = np.angle(fft_result[:num_components])
                fig.add_trace(go.Scatter( y=approximated_signal.real, name = col[0], mode='lines', opacity=1, line={'width': 2 }))
                dots.append(approximated_signal.real[k])
                fig.add_trace(go.Scatter( x = [k], y=[values[k]], opacity=1, mode='markers', marker_line_width=6, marker_size=20))


       # fig.add_trace(go.Scatter( x = [k]*len(dots), y=dots, opacity=1, mode='markers', line=dict(color='red'), marker_line_width=4, marker_size=20))

        fig.update_xaxes(title_text="time")
        fig.update_yaxes(title_text="values")

       # fig.update_traces(mode='markers', )

        fig.update_layout(title_text=activity_name, title_x=0.5, template = "plotly_dark", width = WIDTH, height = HEIGHT,
                              xaxis=dict(range=[0, nr_points_displayed]),
                              yaxis=dict(range=[-0.07, 0.07])
            )    
        fig.show()
        exit()

        #fig.write_image(f"animations\\moving_dots2\\frame_{frame_nr:04d}.png")
        frame_image = fig.to_image(format = "jpg")
        frame_bgr = cv2.imdecode(np.frombuffer(frame_image, np.uint8), -1)
        video.write(frame_bgr)
        
        frame_nr+=1
    video.release()

def moving_segment_on_line():
    DATA_SIZE = 1500#this is higher than nr_points_displayed so that 
    DATA_START = 5000
    activity_waves = read_saved_broken_down(DATA_START, DATA_SIZE)# break_down_data(DATA_SIZE, DATA_START = 5000)
    activity_name = "walking"
    nr_points_displayed = 30
    broken_down_pd = activity_waves[activity_name]
    
    
    segment_length = 50
    frame_nr = 0
    
    WIDTH = 1920 #1280
    HEIGHT =1080 # 720


    image_folder = r'D:\programming\MovementWaves\animations\moving_dots2'

    video_name = f'{image_folder}\\moving_segment11.mp4'

    fps = 60
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, fps, (WIDTH, HEIGHT))


    for k in range(nr_points_displayed):
        fig = go.FigureWidget()
        dots = []
        for col in broken_down_pd.iteritems():
            if "alpha_" in col[0]:
                values = col[1].values
                if np.mean(values) < 0:
                    continue
                #fig.add_trace(go.Scatter( y=values, name = col[0], mode='lines', opacity=1, line={'width': 1 }))
                #dots.append(values[k])
                
                fft_result = np.fft.fft(values)
                num_components = 100
                fft_result[num_components+1:] = 0
                approximated_signal = np.fft.ifft(fft_result)
                fig.add_trace(go.Scatter(y=approximated_signal.real[:nr_points_displayed], mode='lines', opacity=1, line={'width': 1 }))
                
              #  dots.append(approximated_signal.real[(k+segment_length)//2])
                
                fig.add_trace(go.Scatter(x = list(range(k, k+segment_length)), y=approximated_signal.real[k:k+segment_length], mode='lines', opacity=1, line={'width': 3  }))
               # fig.add_trace(go.Scatter( x = [k+segment_length//2], y=[approximated_signal.real[k+segment_length//2]], opacity=1, mode='markers', marker_line_width=6, marker_size=20))
             #   fig.add_trace(go.Scatter( x = [k], y=[approximated_signal.real[k]], opacity=1, mode='markers', marker_line_width=6, marker_size=20))


       
        #fig.add_trace(go.Scatter( x = [(k+segment_length)//2]*len(dots), y=dots, opacity=1, mode='markers', line=dict(color='red'), marker_line_width=4, marker_size=20))

        fig.update_xaxes(title_text="time")
        fig.update_yaxes(title_text="values")

       # fig.update_traces(mode='markers', )

        fig.update_layout(title_text=activity_name, title_x=0.5, template = "plotly_dark", width = WIDTH, height = HEIGHT,
                                xaxis=dict(range=[0, nr_points_displayed]),
                                yaxis=dict(range=[-0.04, 0.04]),
                                
            )    
       # fig.show()
       # exit()
        #fig.write_image(f"animations\\moving_dots2\\frame_{frame_nr:04d}.png")
        frame_image = fig.to_image(format = "jpg")
        frame_bgr = cv2.imdecode(np.frombuffer(frame_image, np.uint8), -1)
        video.write(frame_bgr)
        
        frame_nr+=1
    video.release()

def read_saved_broken_down(DATA_START, DATA_SIZE, HZ = "blind"):
    activity_waves = {}
    for dataset in os.listdir("processed_data\\saved_broken_down"):
        activity_name = dataset.split(".parquet")[0]
        activity_waves[activity_name] = pd.read_parquet(f"processed_data\\saved_broken_down\\{dataset}", engine = "pyarrow")[DATA_START : DATA_START + DATA_SIZE]

    return activity_waves


def break_down_data(DATA_SIZE, HZ = "blind", DATA_START = 0):
    global acts
    #TODO: separate those with different frequencies or normalize the freq
    DATA_START = 0
    
    DATA_END = DATA_START + DATA_SIZE
    to_analyze = ["alpha", "beta", "gyro", "acc"]


    activity_waves = {}

    for i, file_name in enumerate(os.listdir(f"processed_data\\train_data\\train_data_{HZ}")):
        print(i, file_name)
        activity_name = file_name.split(".")[0]
        # if activity_name != "cycling":
        #     continue
        
        activity_data = pd.read_parquet(f"processed_data\\train_data\\train_data_{HZ}\\{file_name}")#[DATA_START:DATA_END]

        broken_down_pd = {}
        for col in activity_data.items():
            if col[0] == "p_id":
                continue
            if col[0] not in to_analyze:
                continue

            if isinstance(col[1].values[0], np.ndarray):
                for i in range(len(col[1].values[0])):
                    split_arr = np.array([v[i] for v in col[1].values])
                    max_val = max(np.max(split_arr), abs(np.min(split_arr)))
                    if(max_val) == 0:
                        max_val = 0.00001
                    #broken_down_pd[f"{col[0]}_{i}"] = split_arr/max_val#np.tanh(split_arr)#/max_val
                    #broken_down_pd[f"{col[0]}_{i}"] = np.tanh(split_arr)
                    broken_down_pd[f"{col[0]}_{i}"] = split_arr
            else:
                max_val = max(np.max(col[1].values), abs(np.min(col[1].values)))
                if(max_val) == 0:
                    max_val = 0.00001
               # broken_down_pd[f"{col[0]}"] = col[1].values/max_val#np.tanh(col[1].values)#/max_val
                #broken_down_pd[f"{col[0]}"] = np.tanh(col[1].values)
                broken_down_pd[f"{col[0]}"] = col[1].values
        broken_down_pd = pd.DataFrame(broken_down_pd)
        #print([r for r in broken_down_pd.columns])
        activity_waves[activity_name] = broken_down_pd
        broken_down_pd.to_parquet(f"processed_data\\saved_broken_down\\{activity_name}.parquet")
        print(broken_down_pd.head)
    #exit()

    #TODO IMPORTANT: why does it build the same result if i concat it or just say activity_waves[activity_name] = broken_down_pd. same length on train nn. wtf
   # for a in activity_waves:
       # activity_waves[a] = activity_waves[a][:1000]
    acts = activity_waves

    #WTF all of them should have just 1000 now. why does it still have 20k? there is something weird going on. this function may not be called or that global stuff is broken
    return activity_waves

def break_down_data_individual(DATA_SIZE, individual_folder):
    DATA_START = 0
    
    DATA_END = DATA_START + DATA_SIZE
    to_analyze = ["alpha", "beta", "gyro", "acc"]

    
    activity_waves = {}
    for i, file_name in enumerate(os.listdir(f"processed_data\\train_data\\train_data_individual\\{individual_folder}")):
        print(i, file_name)
        activity_name = file_name.split(".")[0]
        # if activity_name != "cycling":
        #     continue
        
        activity_data = pd.read_parquet(f"processed_data\\train_data\\train_data_individual\\{individual_folder}\\{file_name}")[DATA_START:DATA_END]

        broken_down_pd = {}
        for col in activity_data.items():
            if col[0] == "p_id":
                continue
            if col[0] not in to_analyze:
                continue

            if isinstance(col[1].values[0], np.ndarray):
                for i in range(len(col[1].values[0])):
                    split_arr = np.array([v[i] for v in col[1].values])
                    max_val = max(np.max(split_arr), abs(np.min(split_arr)))
                    if(max_val) == 0:
                        max_val = 0.00001
                    #broken_down_pd[f"{col[0]}_{i}"] = split_arr/max_val#np.tanh(split_arr)#/max_val
                    #broken_down_pd[f"{col[0]}_{i}"] = np.tanh(split_arr)
                    broken_down_pd[f"{col[0]}_{i}"] = split_arr
            else:
                max_val = max(np.max(col[1].values), abs(np.min(col[1].values)))
                if(max_val) == 0:
                    max_val = 0.00001
               # broken_down_pd[f"{col[0]}"] = col[1].values/max_val#np.tanh(col[1].values)#/max_val
                #broken_down_pd[f"{col[0]}"] = np.tanh(col[1].values)
                broken_down_pd[f"{col[0]}"] = col[1].values
        broken_down_pd = pd.DataFrame(broken_down_pd)
        #print([r for r in broken_down_pd.columns])
        activity_waves[activity_name] =  broken_down_pd
        print(broken_down_pd.head)
    #exit()

    return activity_waves

def build_rolling_ftts_activity(broken_down_pd, num_components, bucket_size, step_size, min_size_dataset):
    #TODO: write this more efficiently and 
    ROLLING_BUCKET_SIZE = bucket_size
    ROLLING_STEP = step_size

    train_ftt_waves = []
    first_leng = -1

    intervals = np.array([i for i in range(0, len(broken_down_pd)-2*ROLLING_STEP, ROLLING_STEP)])
    
   # print("min_size = ", min_size_dataset)
   # print("df size = ", len(broken_down_pd))
   # print("intervals_size = ", len(intervals))
    nr_random_elements = min_size_dataset-2*ROLLING_STEP

    print("this leng vs min leng", len(intervals), min_size_dataset)
    
   # print(len(intervals))
    
    random_indices = np.random.choice(len(intervals), nr_random_elements, replace=False)
    random_intervals = intervals[random_indices]
   # random_intervals = intervals
    for i in random_intervals:
        merged_row = []
        max_len = len(broken_down_pd)
        ii = 0
        for col in broken_down_pd.items():

            if ii == max_len-1:
                break

            values = col[1].values[i:i+ROLLING_BUCKET_SIZE]

           # max_val = np.max([0.0001, abs(np.min(values)), np.max(values)])
           # values = values / max_val

            fft_result = np.fft.fft(values)

            fft_result[num_components+1:] = 0
            # approximated_signal = np.fft.ifft(fft_result)

            amplitudes = np.abs(fft_result[:num_components])
           # print(amplitudes.shape)
            phases = np.angle(fft_result[:num_components])
           # print(len(fft_result))
           # print(len(fft_result[:num_components]))

            merged_row.append(np.concatenate((amplitudes, phases)))

            ii+=1
            
            
            #print(train_ftt_waves[col[0]])

            #for i, (amplitude, phase) in enumerate(zip(amplitudes, phases)):
            #    print(f"Component {i+1}: Amplitude = {amplitude:.2f}, Phase = {phase:.2f} radians")

        merged_row = np.concatenate(merged_row)
        

        if first_leng ==-1:
            first_leng = len(merged_row)

        if len(merged_row)!= first_leng:
            print((len(merged_row), first_leng))
            continue
        train_ftt_waves.append(merged_row)
            #exit()
   # train_ftt_waves = pd.DataFrame(train_ftt_waves)
   # train_ftt_waves['Merged_Column'] = train_ftt_waves.apply(lambda row: np.concatenate(row), axis=1)
   # train_ftt_waves.drop(columns=[col for col in train_ftt_waves.columns if col != 'Merged_Column'], inplace=True)


    #TODO: add check all rows have the same length
    return np.array(train_ftt_waves), first_leng




def analyze_waves_stack():
    DATA_SIZE = 500
    #activity_waves = break_down_data(DATA_SIZE)
    labels = []
    full_train_matrix = []
    
    NR_COMPONENTS = 20
    BUCKET_SIZE = 50
    STEP_SIZE = 1
   
    activity_waves = {}
    min_size_dataset = 9999999
    for a in ['walking', 'walkingUpstairs', 'cycling', 'walkingDownstairs', 'couch', 'standing', 'laying', 'walkingBarefoot', 'driving', 'sitting', 'other']: 
        activity_waves[a] = pd.read_parquet(f"processed_data\\dumps\\waves_broken_down_{a}.parquet", engine='pyarrow')[:DATA_SIZE]
        if len(activity_waves[a]) < min_size_dataset:
            min_size_dataset = len(activity_waves[a])

    for activity in activity_waves:
        print(activity)
        train_ftt_waves, first_leng = build_rolling_ftts_activity(activity_waves[activity], NR_COMPONENTS, BUCKET_SIZE, STEP_SIZE, len(activity_waves[activity]))#, min_size_dataset)   
      
        if not len(full_train_matrix):
            full_train_matrix = train_ftt_waves
        else:
            full_train_matrix = np.concatenate([full_train_matrix, train_ftt_waves])
        labels += [activity]*len(train_ftt_waves)

        print("first_leng", first_leng)
        
    print(set(labels))

    #TODO: check all row_lengths are the same
    full_train_matrix = np.array(full_train_matrix)
    print("shapes")
    print(full_train_matrix.shape)

    full_train_matrix = full_train_matrix.reshape(-1, first_leng)
    if len(labels) != len(full_train_matrix):
        print("LENGTHS ARE DIFFERENT", len(labels), len(full_train_matrix))
        exit()

        
   # np.save(f"processed_data\\waves_data\\full_wave_matrix_{NR_COMPONENTS}_{BUCKET_SIZE}_{STEP_SIZE}.npy", full_train_matrix)
    #np.save(f"processed_data\\waves_data\\labels_{NR_COMPONENTS}_{BUCKET_SIZE}_{STEP_SIZE}.npy", np.array(labels))
    

    
    print(full_train_matrix.shape)

    fast_tsne = fastTSNE(n_components = 2, n_jobs = 4, random_state = 42, perplexity = 50)
    tsne_transform = fast_tsne.fit(full_train_matrix)
    print(tsne_transform)

    color_range = labels
    print("leng data", len(tsne_transform[:, 0]))
    fig = px.scatter(x=tsne_transform[:, 0], y=tsne_transform[:, 1], opacity=1, color = color_range, color_continuous_scale="magma")
   
    fig.update_traces(marker_size=3)

    fig.update_layout(
        title=f"DATA_SIZE: {min_size_dataset} NR_COMPONENTS: {NR_COMPONENTS} BUCKET_SIZE: {BUCKET_SIZE} STEP_SIZE: {STEP_SIZE}",
        xaxis_title="x",
        yaxis_title="y",
        width=1000, height=1000,
        template = "plotly_dark",

    )
    fig.update_xaxes(range=[-130,130])
    fig.update_yaxes(range=[-130,130])
    fig.show()
    pyo.plot(fig, filename=f"processed_data\\waves_graphs\\labels_{NR_COMPONENTS}_{BUCKET_SIZE}_{STEP_SIZE}.html")


    
    #print(train_data.head)


def analyze_waves(BUCKET_SIZE, HZ = "blind"):
    global acts

    
    for a in acts:
        acts[a].to_parquet(f"processed_data\\dumps\\waves_broken_down_{a}.parquet")

    activity_waves = {}
    min_size_dataset = 999999999
   # for a in ['walking', 'walkingUpstairs', 'cycling', 'walkingDownstairs', 'couch', 'standing', 'laying', 'walkingBarefoot', 'driving', 'sitting', 'other']:
    for a in ['sitting','cycling', 'walking', 'other', 'standing', 'driving']:#, 'couch']:

        activity_waves[a] = pd.read_parquet(f"processed_data\\dumps\\waves_broken_down_{a}.parquet", engine='pyarrow')
        if len(activity_waves[a]) < min_size_dataset:
            min_size_dataset = len(activity_waves[a])
    

    
    #for NR_COMPONENTS in range(BUCKET_SIZE//10, BUCKET_SIZE+1, BUCKET_SIZE//10):
    for NR_COMPONENTS in [5,15]:    

        for STEP_SIZE in [1]:#in range(1, BUCKET_SIZE, BUCKET_SIZE//3):
            full_train_matrix = {}
            for activity in activity_waves:
                print(activity)
                train_ftt_waves, row_length = build_rolling_ftts_activity(activity_waves[activity], NR_COMPONENTS, BUCKET_SIZE, STEP_SIZE, min_size_dataset)   
                if activity not in full_train_matrix:
                    full_train_matrix[activity] = train_ftt_waves
                else:
                    full_train_matrix[activity] = np.concatenate([full_train_matrix[activity], train_ftt_waves])
                
                full_train_matrix[activity] = np.array(full_train_matrix[activity]).reshape(-1, row_length)
                np.save(f"processed_data\\waves_data_{HZ}\\full_wave_matrix_{activity}_{NR_COMPONENTS}_{BUCKET_SIZE}_{STEP_SIZE}.npy", full_train_matrix[activity])


def analyze_waves_individual(individual_folder, activity_waves):
    min_size_dataset = 999999999

    for a in ['sitting','cycling', 'walking', 'other', 'standing', 'driving']:
        if a in activity_waves:
            if len(activity_waves[a]) < min_size_dataset:
                min_size_dataset = len(activity_waves[a])
    
   # print([a for a in activity_waves])
    STEP_SIZE = 1
    for BUCKET_SIZE in [50, 10]:
        if BUCKET_SIZE == 50:
            components = [15]#[10,15,20]
        if BUCKET_SIZE == 10:
            components = [5]#[3,5,8]
        for NR_COMPONENTS in components:    
            full_train_matrix = {}
            for activity in activity_waves:
                if activity not in ['sitting','cycling', 'walking', 'other', 'standing', 'driving']:
                    continue
                print(activity)
                train_ftt_waves, row_length = build_rolling_ftts_activity(activity_waves[activity], NR_COMPONENTS, BUCKET_SIZE, STEP_SIZE, min_size_dataset)   
                if activity not in full_train_matrix:
                    full_train_matrix[activity] = train_ftt_waves
                else:
                    full_train_matrix[activity] = np.concatenate([full_train_matrix[activity], train_ftt_waves])
                
                full_train_matrix[activity] = np.array(full_train_matrix[activity]).reshape(-1, row_length)
                
                folder_path = f"processed_data\\waves_data_individual\\{individual_folder}"
                if not os.path.isdir(folder_path):
                    os.makedirs(folder_path)
                np.save(f"{folder_path}\\full_wave_matrix_{activity}_{NR_COMPONENTS}_{BUCKET_SIZE}_{STEP_SIZE}.npy", full_train_matrix[activity])

def stack_data(activity_data):
    corrutped = 0
    train_matrix = []
    print(activity_data.head)
    for row in activity_data.itertuples():
        train_row = list(row.alpha)
        train_row += list(row.beta)
        train_row += list(row.acc)
        train_row += list(row.gyro)
       
        
        train_matrix.append(np.array(train_row))

    
    print("corrupted:", corrutped)
    
    print(len(train_matrix))
    print(len(train_row))
    train_matrix = np.concatenate(train_matrix).reshape(-1, len(train_row) )
       
    return train_matrix, activity_data["back_x"].values, activity_data["back_y"].values

def analyze_patterns_solo():

    DATA_SIZE = 2000
    DATA_START = 1000

    DATA_END = DATA_START+DATA_SIZE
    activity_waves = read_saved_broken_down(DATA_START, DATA_SIZE)# break_down_data(DATA_SIZE, DATA_START = 5000)
    activity_name = "cycling"
    broken_down_pd = activity_waves[activity_name]
    
    nr_points_displayed = 200
    frame_nr = 0
    
    WIDTH = 2000
    HEIGHT = 1500

    labels = []
    hardcode_activity = "cycling"
    activity_data = pd.read_parquet(f"processed_data\\train_data\\train_data_5Hz\\{hardcode_activity}.parquet", engine='pyarrow')[DATA_START:DATA_END]
    # print(activity_data)
    #exit()
    big_matrix, rec_x, rec_y = stack_data(activity_data)

    activity_name = hardcode_activity #file_name.split("_")[-1].split(".")[0]

    labels = range(len(big_matrix))
    
    print(big_matrix.shape)
    fast_tsne = fastTSNE(n_components = 2, n_jobs = 8, random_state = 42, perplexity = 23)
    # fast_tsne = TSNE(n_components = 2, n_jobs = 8, random_state = 42, perplexity = 23)
    tsne_transform = fast_tsne.fit(big_matrix)
    color_range = labels
    
    BATCH_SIZE = 50
    frame_nr = 0
    step_size = 1
    for step in range(0, DATA_SIZE,step_size):
        fig = px.scatter(x=tsne_transform[:, 0][step:step+BATCH_SIZE], y=tsne_transform[:, 1][step:step+BATCH_SIZE], opacity=1, color = color_range[step:step+BATCH_SIZE], color_continuous_scale="magma")
        fig.update_traces(marker_size = 3)
        fig.update_layout(
            title=f"{activity_name}",
            xaxis_title="x",
            yaxis_title="y",
            width=1000, height=800,
            template =  "plotly_dark",

        )
        fig.update_xaxes(range=[-130,130])
        fig.update_yaxes(range=[-130,130])

        rec_img = compute_hover_reconstructed_interval(rec_x[step:step+BATCH_SIZE], rec_y[step:step+BATCH_SIZE])
        distrib_img = fig.to_image(format = "png")

        concat_img, _ = concat_images([distrib_img, rec_img], [0,0,100,0], [110,110,110,90])
        concat_img.save(f"animations\\tsne_wave\\1\\frame_{frame_nr}.png")
        #fig.write_image(f"processed_data\\wave_function_visualization\\frames\\cycling\\frame_{frame_nr}.png")
        frame_nr+=1


def moving_segment_on_line_with_tsne():
    DATA_SIZE = 1500#this is higher than nr_points_displayed so that 
    DATA_START = 5000


    activity_waves = read_saved_broken_down(DATA_START, DATA_SIZE)# break_down_data(DATA_SIZE, DATA_START = 5000)
    activity_name = "walking"
    nr_points_displayed = 900
    broken_down_pd = activity_waves[activity_name]
    segment_length = 15
    frame_nr = 0
    
    activity_data = pd.read_parquet(f"processed_data\\train_data\\train_data_5Hz\\{activity_name}.parquet", engine='pyarrow')[DATA_START:DATA_START+DATA_SIZE]
    big_matrix, rec_x, rec_y = stack_data(activity_data)
    labels = range(len(big_matrix))
    fast_tsne = fastTSNE(n_components = 2, n_jobs = 8, random_state = 42, perplexity = 23)
    # fast_tsne = TSNE(n_components = 2, n_jobs = 8, random_state = 42, perplexity = 23)
    tsne_transform = fast_tsne.fit(big_matrix)
    color_range = labels
    


    WIDTH = 1920-260 #1280
    HEIGHT =1080 # 720


    image_folder = r'D:\programming\MovementWaves\animations\moving_segment_tsne\4'


    for k in range(nr_points_displayed):

        fig = px.scatter(x=tsne_transform[:, 0][k:k+segment_length], y=tsne_transform[:, 1][k:k+segment_length], opacity=1, color = list(range(segment_length)), color_continuous_scale="magma")
        fig.update_traces(marker_size = 3)
        fig.update_layout(
            #title=f"{activity_name}",
            xaxis_title="x",
            yaxis_title="y",
            width=1000, height=800,
            template =  "plotly_dark",

        )
        fig.update_xaxes(range=[-130,130])
        fig.update_yaxes(range=[-130,130])

        rec_img = compute_hover_reconstructed_interval(rec_x[k:k+segment_length], rec_y[k:k+segment_length])
        distrib_img = fig.to_image(format = "png")

        concat_img, _ = concat_images([distrib_img, rec_img], [0,0,100,0], [110,110,100,90])
        

        fig = go.FigureWidget()
        dots = []
        for col in broken_down_pd.iteritems():
            if "alpha_" in col[0]:
                values = col[1].values
                if np.mean(values) < 0:
                    continue

                fft_result = np.fft.fft(values)
                num_components = 100
                fft_result[num_components+1:] = 0
                approximated_signal = np.fft.ifft(fft_result)
                fig.add_trace(go.Scatter(y=approximated_signal.real[:nr_points_displayed], mode='lines', opacity=0.7, line={'width': 1 }))
        
                fig.add_trace(go.Scatter(x = list(range(k, k+segment_length)), y=approximated_signal.real[k:k+segment_length], mode='lines', opacity=1, line={'width': 3  }))
             #   fig.add_trace(go.Scatter( x = [k+segment_length//2], y=[approximated_signal.real[k+segment_length//2]], opacity=1, mode='markers', marker_line_width=4, marker_size=15))
                #fig.add_trace(go.Scatter( x = [k], y=[approximated_signal.real[k]], opacity=1, mode='markers', marker_line_width=6, marker_size=20))
        #fig.add_trace(go.Scatter( x = [(k+segment_length)//2]*len(dots), y=dots, opacity=1, mode='markers', line=dict(color='red'), marker_line_width=4, marker_size=20))

        fig.update_xaxes(title_text="time")
        fig.update_yaxes(title_text="values")
        fig.update_layout(title_text=activity_name, title_x=0.5, template = "plotly_dark", width = WIDTH, height = HEIGHT,
                                xaxis=dict(range=[0, nr_points_displayed]),
                                yaxis=dict(range=[-0.04, 0.04]),
                                
            )    

        frame_image = fig.to_image(format = "png")

        concat_img, _ = concat_images_vertical([Image.open(io.BytesIO(frame_image)), concat_img], [40,30,130,500], [0,0,0,0])

        concat_img = replace_ugly_blue(concat_img)

        concat_img.save(f"{image_folder}\\frame_{frame_nr}.png")
        frame_nr+=1

def replace_ugly_blue(image):
    image = image.convert('RGBA')

    target_color = (3, 5, 18)  
    replacement_color = (17, 17, 17)

    pixel_data = list(image.getdata())

    new_pixel_data = []
    # pixel_dist = {}
    # print(len(pixel_data))
    # for pixel in pixel_data:
    #     if pixel in pixel_dist:
    #         pixel_dist[pixel]+=1
    #     else:
    #         pixel_dist[pixel] = 0

    # for p in pixel_dist:
    #     if pixel_dist[p] > 5000:
    #         print(p, pixel_dist[p])
   # exit()

    # (17, 17, 17, 255) 1471800
    # (40, 52, 66, 255) 10712
    # (3, 5, 18, 255) 403165
    # (0, 0, 0, 255) 7999
    # (13, 14, 33, 255) 9716
    # (8, 9, 25, 255) 11356
    # print(pixel)
    for pixel in pixel_data:
        if pixel[:3] == target_color:
            new_pixel_data.append(replacement_color)
        else:
            new_pixel_data.append(pixel)

    modified_img = Image.new('RGBA', image.size)
    modified_img.putdata(new_pixel_data)
    image = modified_img

    return image

def analyze_wave_distribution():
    DATA_START = 20000
    DATA_SIZE = 9*3600
    DATA_END = DATA_START + DATA_SIZE

    
    big_matrix = []
    labels = []
    activities = ["walking", "cycling", "driving", "standing", "sitting"]
    for hardcode_activity in activities:
        

        activity_data = pd.read_parquet(f"processed_data\\train_data\\{hardcode_activity}.parquet", engine='pyarrow')[DATA_START:DATA_END]

        activity_matrix, rec_x, rec_y = stack_data(activity_data)
        big_matrix.append(activity_matrix)
        labels += [hardcode_activity] * len(activity_matrix)



    big_matrix = np.concatenate(big_matrix)
    print(big_matrix.shape)
    fast_tsne = fastTSNE(n_components = 2, n_jobs = 8, random_state = 42, perplexity = 23)
    tsne_transform = fast_tsne.fit(big_matrix)
    
    color_range = labels
    fig = px.scatter(x=tsne_transform[:, 0], y=tsne_transform[:, 1], opacity=1, color = color_range, color_continuous_scale="rainbow")
    fig.update_traces(marker_size = 3)
    fig.update_layout(
        title=f"global raw data activities",
        xaxis_title="x",
        yaxis_title="y",
        width=1000, height=1000,
        template = "plotly_dark",

    )
    fig.update_xaxes(range=[-130,130])
    fig.update_yaxes(range=[-130,130])
    fig.show()

   # pyo.plot(fig, filename=f"D:\\programming\\MovementWaves\\processed_data\\wave_function\\global_waves.html")


    for BATCH_SIZE in [10,50,100]:
        for a, hardcode_activity in enumerate(activities):


            activity_name = hardcode_activity #file_name.split("_")[-1].split(".")[0]

            



            
            frame_nr = 0
            step_size = 1
            density_maps = []
            for step in range(a*DATA_SIZE, a*DATA_SIZE + DATA_SIZE, step_size):
            # fig = px.scatter(x=tsne_transform[:, 0][step:step+BATCH_SIZE], y=tsne_transform[:, 1][step:step+BATCH_SIZE], opacity=1, color = color_range[step:step+BATCH_SIZE], color_continuous_scale="magma")
                
                
                x=tsne_transform[:, 0][step:step+BATCH_SIZE]
                y=tsne_transform[:, 1][step:step+BATCH_SIZE]
                
                data = np.vstack([x, y]).T

                kde = stats.gaussian_kde(data.T)

                x_grid, y_grid = np.meshgrid(np.linspace(-100, 100, 100), np.linspace(-100, 100, 100))
                grid_coords = np.vstack([x_grid.ravel(), y_grid.ravel()])

                density_values = kde(grid_coords)

                density_map = density_values.reshape(x_grid.shape)

              #  density_map = density_map / np.max(density_map)
                density_maps.append(density_map)

           # max_val_a = np.max(np.concatenate(density_maps).flatten())
           # for i in range(len(density_maps)):
               # density_maps[i] /= max_val_a


            for density_map in density_maps:

                sca_a = {"walking":1.5, "cycling":3, "driving":10, "standing":6, "sitting":1}
                sca = {10: 0.0001, 50: 0.00025 ,100: 0.0005}

                s_scale = sca[BATCH_SIZE] * sca_a[hardcode_activity]
                if np.max(density_map) > s_scale:
                    density_map = density_map/s_scale

                fig_3d = go.Figure(data=[go.Surface(z=density_map, colorscale='magma', showscale=False)])
                fig_3d.update_layout(scene=dict({}), width=1000, height=1000, template = "plotly_dark", title = f"{hardcode_activity} Lag: {BATCH_SIZE} Wave Estimation for ")

                fig_3d.update_layout(scene_camera=dict(eye=dict(x=1.8*(-0.7 +0.5*1.5), y=1.8*(-1.5 +0.5*1.5), z=1.8*(1))))




                fig_3d.update_layout(
                    scene=dict(
                        xaxis=dict(range=[-50, 150]),
                        yaxis=dict(range=[-50, 150]),
                        zaxis=dict(range=[0, s_scale]),
                    )
                )
                fig_3d.update_coloraxes(showscale=False)


                folder_path = f"processed_data\\wave_function\\{hardcode_activity}_{BATCH_SIZE}_z3"
                if not os.path.isdir(folder_path):
                    os.makedirs(folder_path)

                fig_3d.write_image(f"{folder_path}\\prob_dist_{frame_nr}.png")
                frame_nr+=1
                if frame_nr > 150:
                    break

def analyze_wave_distribution_global():
    DATA_START = 10000
    DATA_SIZE = 10000
    DATA_END = DATA_START + DATA_SIZE
    MAX_FRAMES = 150

    
    big_matrix = []
    labels = []
    activities = ["walking", "cycling", "driving", "standing", "sitting"]
    for hardcode_activity in activities:
        

        activity_data = pd.read_parquet(f"processed_data\\train_data\\train_data_5Hz\\{hardcode_activity}.parquet", engine='pyarrow')[DATA_START:DATA_END]

        activity_matrix, rec_x, rec_y = stack_data(activity_data)
        big_matrix.append(activity_matrix)
        labels += [hardcode_activity] * len(activity_matrix)



    big_matrix = np.concatenate(big_matrix)
    print(big_matrix.shape)
    fast_tsne = fastTSNE(n_components = 2, n_jobs = 8, random_state = 42, perplexity = 23)
    tsne_transform = fast_tsne.fit(big_matrix)
    
    color_range = labels
    fig = px.scatter(x=tsne_transform[:, 0], y=tsne_transform[:, 1], opacity=1, color = color_range, color_continuous_scale="rainbow")
    fig.update_traces(marker_size = 3)
    fig.update_layout(
        title=f"global: acc, gyro",
        xaxis_title="x",
        yaxis_title="y",
        width=1000, height=1000,
        template = "plotly_dark",

    )
    fig.update_xaxes(range=[-130,130])
    fig.update_yaxes(range=[-130,130])
    fig.show()

    pyo.plot(fig, filename=f"D:\\programming\\MovementWaves\\processed_data\\wave_function\\global_waves.html")


    for BATCH_SIZE in [10,50,100]:

        frame_nr = 0
        step_size = 1
        density_maps = {a:[] for a in activities}
        for s in range(DATA_SIZE):
            
            for a, hardcode_activity in enumerate(activities):
                step = a*DATA_SIZE+s
            # fig = px.scatter(x=tsne_transform[:, 0][step:step+BATCH_SIZE], y=tsne_transform[:, 1][step:step+BATCH_SIZE], opacity=1, color = color_range[step:step+BATCH_SIZE], color_continuous_scale="magma")
                
                
                x=tsne_transform[:, 0][step:step+BATCH_SIZE]
                y=tsne_transform[:, 1][step:step+BATCH_SIZE]
                
                data = np.vstack([x, y]).T

                kde = stats.gaussian_kde(data.T)

                x_grid, y_grid = np.meshgrid(np.linspace(-100, 100, 100), np.linspace(-100, 100, 100))
                grid_coords = np.vstack([x_grid.ravel(), y_grid.ravel()])

                density_values = kde(grid_coords)

                density_map = density_values.reshape(x_grid.shape)
                
               # density_map = density_map / np.max(density_map)

                density_maps[hardcode_activity].append(density_map)

            #use avg or other if they get too small
            frame_nr+=1
            if frame_nr > MAX_FRAMES:
                break

        for a in activities:
            max_val_a = np.max(np.concatenate(density_maps[a]).flatten())
            for i in range(len(density_maps[a])):
                density_maps[a][i] /= max_val_a

        frame_nr = 0        
        for s in range(DATA_SIZE):
            frame_density_maps = [density_maps[a][s] for a in activities ]

            colorscales = ["magma","ice","viridis","RdBu","rainbow"]
            fig_3d = go.Figure(data=[go.Surface(z=density_map, colorscale=colorscale, showscale=False) for (density_map, colorscale) in zip(frame_density_maps,colorscales)])

            fig_3d.update_layout(scene=dict({}), width=1000, height=1000, template = "plotly_dark", title = f"{hardcode_activity} Lag: {BATCH_SIZE} Wave Estimation for ")

            fig_3d.update_layout(scene_camera=dict(eye=dict(x=-0.7 +0.5*1.5, y=-1.5 +0.5*1.5, z=1.5)))

            sca = {10: 0.025, 50: 0.125 ,100: 0.25}
            sca_a = {"walking":1, "cycling":1, "driving":1, "standing":1, "sitting":1}

            s_scale = sca[BATCH_SIZE] * sca_a[hardcode_activity]
            #if 

            fig_3d.update_layout(
                scene=dict(
                    xaxis=dict(range=[-20, 120]),
                    yaxis=dict(range=[-20, 120]),
                  #  zaxis=dict(range=[0, s_scale]),
                )
            )
            fig_3d.update_coloraxes(showscale=False)


            folder_path = f"processed_data\\wave_function\\global_{BATCH_SIZE}"
            if not os.path.isdir(folder_path):
                os.makedirs(folder_path)

            fig_3d.write_image(f"{folder_path}\\prob_dist_{frame_nr}.png")
            frame_nr+=1
            if frame_nr > MAX_FRAMES:
                break

def process_individual():
    DATA_SIZE = 10000000000
    for individual_folder in os.listdir(f"processed_data\\train_data\\train_data_individual"):
        activity_waves = break_down_data_individual(DATA_SIZE, individual_folder)
        analyze_waves_individual(individual_folder, activity_waves)

def threaded_analyze_waves():
    
    #analyze_waves_stack()
    DATA_SIZE = 10000000000
    break_down_data(DATA_SIZE)

    nr_threads = 2
    p = multiprocessing.Pool(processes = nr_threads)
  #  rez = p.map(analyze_waves, [200, 175, 150, 125, 100, 75, 50, 40, 30, 20, 10])
    rez = p.map(analyze_waves, [50, 40, 30, 20, 10])

    p.close()
    p.join()




def spawn_global_points():
    DATA_START = 6000
    DATA_SIZE = 5000
    DATA_END = DATA_START + DATA_SIZE

    
    big_matrix = []
    labels = []
    activities = ["walking", "cycling", "driving", "standing", "sitting"]
    for hardcode_activity in activities:
        activity_data = pd.read_parquet(f"processed_data\\train_data\\train_data_5Hz\\{hardcode_activity}.parquet", engine='pyarrow')[DATA_START:DATA_END]

        activity_matrix, rec_x, rec_y = stack_data(activity_data)
        big_matrix.append(activity_matrix)
        labels += [hardcode_activity] * len(activity_matrix)

    # big_matrix = np.concatenate(big_matrix)

    # fast_tsne = TSNE(n_components = 3, n_jobs = 8, random_state = 42, perplexity = 23)
    # tsne_transform = fast_tsne.fit_transform(big_matrix)
    # tsne_transform = tsne_transform[::-1]

    #np.save(r"D:\programming\MovementWaves\animations\spawn_dataset\tsne_full.npy", tsne_transform)
    
    tsne_transform = np.load(r"D:\programming\MovementWaves\animations\spawn_dataset\tsne_full.npy")
    integer_labels = LabelEncoder().fit_transform(labels)

    # fig = go.Figure(go.Scatter3d(x=tsne_transform[:, 0], y=tsne_transform[:, 1], z = tsne_transform[:, 2], 
    #                              marker=dict(size=5, color=list(range(len(integer_labels))), colorscale='rainbow', opacity=0.7), 
    #                            #  line = dict(color=list(range(len(integer_labels))), colorscale='rainbow'),
    #                              mode = "markers"
    #                              ))

    nr_total_frames = 900
   # x = (np.linspace(0, 1, nr_total_frames))**2 * DATA_SIZE * len(activities)
    x = (np.linspace(0, 1, nr_total_frames)) * DATA_SIZE * len(activities)

    image_folder = r'D:\programming\MovementWaves\animations\spawn_dataset\2'
    video_name = f'{image_folder}\\spawn_points.mp4'

    fps = 60
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, fps, (1500,1500))

    frame_nr = 0
    color_values = list(range(len(integer_labels)))
    for k in x:
        k = int(k)
        copy_tsne = tsne_transform.copy()
        copy_tsne[k+1:, :] = -150
       # print(copy_tsne)
       # exit()
        # fig = go.Figure(go.Scatter(x=copy_tsne[:, 0], y=copy_tsne[:, 1], 
        #                         mode = "markers",
        #                         marker=dict(size=5, color=color_values, colorscale='rainbow', opacity=0.7), 
        #                         #  line = dict(color=list(range(len(integer_labels))), colorscale='rainbow')
        #                             ))

        fig = go.Figure(go.Scatter3d(x=copy_tsne[:, 0], y=copy_tsne[:, 1], z = copy_tsne[:, 2], 
                                    marker=dict(size=5, color=color_values, colorscale='rainbow', opacity=0.7), 
                                    #line = dict(color=color_values, colorscale='rainbow'),
                                    mode = "markers"
                                    ))
        fig.update_traces(marker_size = 3)
        fig.update_layout(
            xaxis_title="x",
            yaxis_title="y",
            width=1500, height=1500,
            template = "plotly_dark",

        )
        fig.update_xaxes(range=[-130,130])
        fig.update_yaxes(range=[-130,130])
        #fig.show()
        #fig.write_image(f"{image_folder}\\frame_{frame_nr:04d}.png")
        frame_image = fig.to_image(format = "jpg")
        frame_bgr = cv2.imdecode(np.frombuffer(frame_image, np.uint8), -1)
        video.write(frame_bgr)
        
        frame_nr+=1
    video.release()



#for better untanglement, add a new variable after fft transform, an average of that interval
#  parse_raw_data_individual()
#  for individual_folder in os.listdir(r'D:\programming\MovementWaves\processed_data\train_data\train_data_individual'):
#      activity_waves = break_down_data_individual(100000000, individual_folder)
#      analyze_waves_individual(individual_folder, activity_waves)
#      exit()

if __name__ == '__main__':

    

    #analyze_waves_stack()
    spawn_global_points()





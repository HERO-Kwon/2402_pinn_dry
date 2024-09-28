from flask import Flask, request, render_template, jsonify, send_from_directory, g, session, send_file
import pandas as pd
import numpy as np
import os
import re
import io
import base64
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
global path_result
path_result = 'D:\\'

max_udm0 = 65.830364
min_udm0 = 0
file_list = [file for file in os.listdir(path_result) if file.endswith('udm0.tiff')]

def get_max_udm0(file):
    im = Image.open(path_result + file)
    imarray = np.array(im)
    im_udm0 = imarray[imarray!=0]
    udm0 = im_udm0/255 * (max_udm0 - min_udm0) + min_udm0

    if len(udm0) == 0:
        return 0
    else:
        return(np.max(udm0))
    
# controller 1
@app.route("/", methods=['GET','POST'])
def index() :
    if request.method == 'POST':
        action = request.form['action']
        if action == 'main_input':
            # collect input as dictionary
            input_data = {
                'z01': request.form.getlist('z01[]'),
                'z02': request.form.getlist('z02[]'),
                'z03': request.form.getlist('z03[]'),
                'z04': request.form.getlist('z04[]'),
                'z05': request.form.getlist('z05[]'),
                'z06': request.form.getlist('z06[]'),
                'z07': request.form.getlist('z07[]'),
                'z08': request.form.getlist('z08[]'),
                'z09': request.form.getlist('z09[]'),
                'z10': request.form.getlist('z10[]'),
                'z11': request.form.getlist('z11[]'),
                'z12': request.form.getlist('z12[]'),
            }
            # convert to dataframe
            input_df = pd.DataFrame(input_data)
            input_df = input_df.T
            input_df.columns=['steam','temperature','c_fan_rpm','e_fan_rpm','h_fan_temperature','h_fan_rpm']

            # save
            input_df.to_csv('df.csv')
            # model run
            '''
            model run
            '''
            # read result data
            id_files = pd.DataFrame()
            for file in file_list:
                found_num = re.findall(r'\d+',file)
                id_num = pd.Series(found_num[:3],index=['zone','seq','y'])
                df_id = pd.DataFrame(id_num).T
                df_id['file'] = file
                df_id['max_udm0'] = get_max_udm0(file)
                id_files = pd.concat([id_files,df_id],ignore_index=True)

            profile_udm0 = id_files[id_files['seq']=='4'].groupby(['zone','seq'])['max_udm0'].mean()
            
            fig,ax = plt.subplots(figsize=(10,2))
            ax.plot(profile_udm0.values)
            ax.set(xlabel='zone',ylabel='Average Udm0')

            # convert to png
            pngImage = io.BytesIO()
            FigureCanvas(fig).print_png(pngImage)
            # Base64 encoding
            pngImageB64String = "data:image/png;base64,"
            pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')
            return render_template('main_page.html',image=pngImageB64String)
        elif action == 'add_input':
            # additional input data
            zone_num = request.form['zone_num']
            elapsed = int(request.form['elapsed'])
            parameter = request.form['parameter']

            # add_df

            str_zone_num = str(zone_num).zfill(2)
            str_elapsed = str(int(elapsed*4/100))
            str_startswith = 'zn' + str_zone_num + '_' + str_elapsed
            add_file_list = [file for file in os.listdir(path_result) if (file.startswith(str_startswith) & file.endswith(parameter+'.tiff'))]

            # create figure

from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
import torch
import cv2
import time

app = Flask(__name__)

def load_model(model_type):
    model = torch.hub.load('ultralytics/yolov5', model_type)
    return model

def plot_bbox_per_frame(frame, dets, frame_save_path):
    for i in dets:
        text_x = int(i[0])
        text_y = int(i[1]) - 10

        cv2.rectangle(frame, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (0, 0, 255), 3)
        cv2.putText(frame, str(int(i[4])), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imwrite(frame_save_path, frame)

def video_in_img_out(vid_inp_path, out_path, model):
    vidObj = cv2.VideoCapture(vid_inp_path)
    count = 0
    success = 1
    df = pd.DataFrame()
    start_time = time.time()

    while success:
        success, image = vidObj.read()
        if not success:
            break

        results = model(image)

        frame_save_path = os.path.join(out_path, str(count) + '.jpg')
        plot_bbox_per_frame(image, results.xyxy[0], frame_save_path)

        df_per_frame = results.pandas().xyxy[0]
        # df = df.append(df_per_frame, ignore_index=True)
        df = pd.concat([df, df_per_frame], ignore_index=True) 

        count += 1

    end_time = time.time()
    inference_time = end_time - start_time
    print(f"Inference time: {inference_time} seconds")

    df.to_csv("./data_vid/output_img_generalise/result1.csv")
    return df

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        model_type = request.form['model_type']
        video_path = request.form['video_path']
        out_path = "./data_vid/output_img_generalise"

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        model = load_model(model_type)
        df = video_in_img_out(video_path, out_path, model)

        return render_template('results.html', df=df.to_html())

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

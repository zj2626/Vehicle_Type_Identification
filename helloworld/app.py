# coding=utf-8
from flask import Flask
from flask import request
from flask import jsonify
from flask import render_template
from werkzeug.utils import secure_filename
import datetime
import random
import os
import run_classification as classify
import run_detection as detect
import _init_paths

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')


@app.route('/temp', methods=['GET', 'POST'])
def temp():
    return render_template('index2.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


def defind_name():
    nowTime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # 生成当前时间
    randomNum = random.randint(0, 100)  # 生成的随机整数n，其中0<=n<=100
    if randomNum <= 10:
        randomNum = str(0) + str(randomNum)
    uniqueNum = str(nowTime) + str(randomNum)
    return uniqueNum


@app.route('/upload', methods=['POST'])
def upload():
    upload_file = request.files['fileList']
    if upload_file and allowed_file(upload_file.filename):
        filename = secure_filename(upload_file.filename)
        newname = defind_name() + filename[filename.index('.'):]
        upload_file.save(
            os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], newname))

        address = app.root_path + '/' + app.config['UPLOAD_FOLDER'] + newname
        data = {'address': address}
        return jsonify(data)
    else:
        return 'Sorry, file format is not supported'


@app.route('/detection', methods=['POST'])
def detection():
    address = request.values.get("address")
    print('start detect .................', address)
    data = {}
    total_time = 0

    try:
        full_address, t_time = detect.picture_detection(address, app)
        data['full_address'] = full_address
        data['message'] = 0
        total_time += t_time
    except Exception as e:
        data['message'] = 1
        print(e.message)

    data['time'] = str(total_time)[:5]

    return jsonify(data)


@app.route('/classification', methods=['POST'])
def classification():
    new_address = []
    address = request.values.get("address")
    dir_files = [(i, os.stat(app.config['UPLOAD_FOLDER'] + i).st_mtime) for i in
                 os.listdir(app.config['UPLOAD_FOLDER'])]
    for inx, i in enumerate(sorted(dir_files, key=lambda x: x[1], reverse=True)):
        if i[0].startswith("cut_" + address.split("/")[-1].split(".")[0]):
            new_address.append(app.config['UPLOAD_FOLDER'] + i[0])

    if len(new_address) == 0:
        new_address.append(address)
    print('start classify.................', new_address)
    data = {}
    total_time = 0

    try:
        data['predict'] = []
        for inx, new_add in enumerate(new_address):
            ifFUllImage = not new_add.startswith(app.config['UPLOAD_FOLDER'] + "cut_")
            top, t_time = classify.picture_classification(new_add, ifFUllImage)
            if (not ifFUllImage) and top[0]['prob'] < 68:
                top2, t_time2 = classify.picture_classification(address, not ifFUllImage)
                if top2[0]['prob'] > top[0]['prob']:
                    top = top2
                    t_time = t_time2

            data['predict'].append([top])
            total_time += t_time
        data['message'] = 0
    except Exception as e:
        data['message'] = 1
        print(e.message)

    data['time'] = str(total_time)[:5]

    return jsonify(data)


if __name__ == '__main__':
    # app.run(threaded=True)
    app.run()

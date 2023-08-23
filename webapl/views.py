from flask import request, redirect, url_for, render_template, flash
from webapl import app
import face_recognition
import cv2
import os
import shutil


@app.route('/', methods=['GET', 'POST'])
def upload():
    # URLでhttp://127.0.0.1:5000/uploadを指定したときはGETリクエストとなるのでこっち
    if request.method == 'GET':
        return render_template('upload.html')
    # formでsubmitボタンが押されるとPOSTリクエストとなるのでこっち
    elif request.method == 'POST':
        file = request.files['img']
        if not file.filename:
            flash('ファイルを選択してください')
            return redirect(url_for('upload'))
        file.save(os.path.join('./webapl/static/image', file.filename))
        value = request.form["send"]
        if value == "mosaic":
            return redirect(url_for('face_mosaic', filename = file.filename))
        elif value == "oil":
            return redirect(url_for('oil_painting', filename = file.filename))


@app.route('/mosaic<filename>')
def face_mosaic(filename):
    # 画像を読み込む
    load_image = face_recognition.load_image_file(os.path.join('./webapl/static/image', filename))
    img = cv2.imread(os.path.join('./webapl/static/image', filename))
    target_dir = './webapl/static/image'
    shutil.rmtree(target_dir)
    os.mkdir(target_dir)
    # 認識させたい画像から顔検出する
    face_locations = face_recognition.face_locations(load_image)
    for i in range(len(face_locations)):
        # roi を抽出する。
        roi = img[face_locations[i][0]:face_locations[i][2], face_locations[i][3]:face_locations[i][1]]

        # モザイク処理を行い、結果を roi に代入する。
        roi[:] = mosaic(roi)
        cv2.imwrite(os.path.join('./webapl/static/image', "result.png"), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
        
    # 結果の画像を表示する
    return render_template("uploaded_file.html")
@app.route('/oil<filename>')
def oil_painting(filename):
    # 画像を読み込む
    load_image = face_recognition.load_image_file(os.path.join('./webapl/static/image', filename))
    img = cv2.imread(os.path.join('./webapl/static/image', filename))
    target_dir = './webapl/static/image'
    shutil.rmtree(target_dir)
    os.mkdir(target_dir)
    # 油絵エフェクトパラメータ
    # size, dynRatio
    prms = [(3,10)]

    for (size, dynRatio) in prms:
        # 油絵エフェクト
        dst = cv2.xphoto.oilPainting(img, size, dynRatio, cv2.COLOR_BGR2Lab)
        cv2.imwrite(os.path.join('./webapl/static/image', "result.png"), dst, [cv2.IMWRITE_JPEG_QUALITY, 100])
    # 結果の画像を表示する
    return render_template("uploaded_file.html")

def mosaic(img, scale=0.1):
    h, w = img.shape[:2]  # 画像の大きさ

    # 画像を scale (0 < scale <= 1) 倍に縮小する。
    dst = cv2.resize(
        img, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
    )

    # 元の大きさに拡大する。
    dst = cv2.resize(dst, dsize=(w, h), interpolation=cv2.INTER_NEAREST)

    return dst

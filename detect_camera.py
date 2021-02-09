
from tflite_runtime.interpreter import Interpreter

from time import sleep, time
from io import BytesIO
from picamera import PiCamera
from PIL import Image

import json

from detect_image import detect_object, load_labels, annotate_objects

DIR = 'annotated_images'

def output(results, labels):
    print("--------")
    
    for result in results:
        print(f"{labels[result['class_id']]} : {result['score']}")

def count_of(target, results, labels):
    count = 0
    for result in results:
        if labels[result['class_id']] == target:
            count += 1

    return count

def main():
    # ラベルを読み込む
    labels = load_labels('model/coco_labels.txt')

    # モデルを読み込む
    interpreter = Interpreter('model/detect.tflite')
    interpreter.allocate_tensors()

    # カメラを接続する
    with PiCamera() as camera:

        camera.resolution = (1024, 768)
        camera.start_preview()

        i = 0
        while True:
            try:
                sleep(5)
                
                # 画像データの格納場所を用意する
                stream = BytesIO()

                start = time()
                
                # 撮影する
                camera.capture(stream, format='jpeg')

                stream.seek(0)

                # 画像として読み込む
                with Image.open(stream) as image:
                    image.save(f'original_images/original{i}.jpg')
                    
                    _, input_width, input_height, _ = interpreter.get_input_details()[0]['shape'] # 入力サイズを取得する
                    resized_image = image.convert('RGB').resize((input_width, input_height), Image.ANTIALIAS) # 画像を加工する

                    # 推論を実行する
                    results = detect_object(interpreter, resized_image, 0.4)

                    # 結果を出力する
                    output(results, labels)

                    count = count_of('person', results, labels)
                    with open('/tmp/persons.json', 'w') as f:
                        json.dump({'count' : count}, f)

                    end = time()

                    time_delta = end - start
                    
                    print(f"時間：{time_delta}秒")
                    # バウンディングボックスとラベルを付けた画像を出力する
                    annotate_objects(image, results, labels, f'{DIR}/byCamera{i}.jpg')
                    
                    i += 1
            except KeyboardInterrupt:
                print("終了")
                break


        sleep(1) # 正常終了のために必要

if __name__ == '__main__':
    main()
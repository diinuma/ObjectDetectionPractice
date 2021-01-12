
from tflite_runtime.interpreter import Interpreter

from time import sleep
from io import BytesIO
from picamera import PiCamera
from PIL import Image

from detect_image import detect_object, load_labels, annotate_objects

def main():
    
    labels = load_labels('/tmp/coco_labels.txt')

    interpreter = Interpreter('/tmp/detect.tflite')
    interpreter.allocate_tensors()

    camera = PiCamera()
    camera.resolution = (1024, 768)
    camera.start_preview()

    sleep(5)

    stream = BytesIO()
    camera.capture(stream, format='jpeg')

    stream.seek(0)

    with Image.open(stream) as image:
        _, input_width, input_height, _ = interpreter.get_input_details()[0]['shape']
        resized_image = image.convert('RGB').resize((input_width, input_height), Image.ANTIALIAS)

        resized_image.save('resized_img.jpg')

        results = detect_object(interpreter, resized_image, 0.4)

        annotate_objects(image, results, labels, 'byCamera.jpg')


if __name__ == '__main__':
    main()
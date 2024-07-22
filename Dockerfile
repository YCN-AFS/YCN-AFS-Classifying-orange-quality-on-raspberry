FROM python:3.10

ADD OR_STAT.py .
ADD model3_class/keras_model.h5 .
ADD orange_5s.onnx .

RUN pip install tensorflow==2.9.0 opencv-python pillow deep-sort-realtime ultralytics

CMD ["python", "./OR_STAT.py"]
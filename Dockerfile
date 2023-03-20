# Must use a Cuda version 11+
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /

# Install git
RUN apt-get update && apt-get install -y git ffmpeg libsm6 libxext6

# Install python packages
RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Install Opencv and controlnet-aux
RUN pip install -q opencv-contrib-python
RUN pip install -q controlnet_aux

# We add the banana boilerplate here
ADD server.py .

# Add your model weight files 
# (in this case we have a python script)
ADD download.py .
RUN python3 download.py


# Add your custom app code, init() and inference()
ADD app.py .

EXPOSE 8000

CMD python3 -u server.py

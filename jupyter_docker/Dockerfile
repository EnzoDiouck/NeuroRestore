# Example Training Dockerfile
FROM tensorflow/tensorflow:2.17.0-gpu

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install jupyter pillow "numpy<2" opencv-python matplotlib tqdm albumentations scikit-image

# Create a directory for models
RUN mkdir -p /app/models

EXPOSE 8888

# Launch Jupyter
CMD ["/bin/bash", "-c", \
"jupyter notebook --ip=0.0.0.0 --allow-root --port=${JUPYTER_PORT:-8888} \
--IdentityProvider.token='' --ServerApp.password=''"]
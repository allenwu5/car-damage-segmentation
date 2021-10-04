FROM detectron2-prebuilt:v0

COPY requirements.txt .
RUN pip install --user -r requirements.txt
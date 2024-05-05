FROM python:3

ADD script.py /script.py
ADD requirements.txt /requirements.txt

RUN pip3 install -r /requirements.txt

ENTRYPOINT  ["python", "main.py", "--input", "$inputDataset", "--output", "$outputDir"]



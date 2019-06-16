FROM kaixhin/brainstorm

RUN git clone https://github.com/pinae/MNIST-Brainstorm

RUN cd MNIST-Brainstorm; sed 's/PyCudaHandler()/NumpyHandler(float)/g' -i train.py; sed 's/PyCudaHandler/NumpyHandler/g' -i train.py; python load_dataset.py; pip install -r requirements.txt

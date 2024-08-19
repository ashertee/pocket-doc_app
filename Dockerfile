# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /pocket_doc_app

# Copy the current directory contents into the container at /pocket_doc_app
COPY . /pocket_doc_app

# Create a writable directory for model files
RUN mkdir -p /pocket_doc_app/model

# Install any needed packages specified in requirements.txt
#RUN pip3 install --no-cache-dir -r /pocket_doc_app/requirements.txt
RUN pip3 install pandas
RUN pip3 install tensorflow
RUN pip3 install numpy
RUN pip3 install scikit-learn
RUN pip3 install flask

# Make port 8050 available to the world outside this container
EXPOSE 8050

# Define environment variable
ENV FLASK_pocket_doc_app=pocket_doc_app.py

# Run pocket_doc_app.py when the container launches
CMD ["flask", "run", "--host=0.0.0.0"]

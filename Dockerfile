# Use an official Python runtime as a base image
FROM python:3.11

# Set the working directory in the container
WORKDIR  /harit_model

# Copy the current directory contents into the container at /app
ADD harit_model harit_model


RUN pip install --upgrade pip
# Install dependencies (assuming a requirements.txt file exists)
RUN pip install -r harit_model/requirements.txt

# Expose the port the app runs on (adjust the port number as needed)
EXPOSE 8080

# Define environment variables if needed (optional)
# ENV VAR_NAME value

# Run the command to start the application (adjust based on your project)
CMD ["python", "harit_model/predict.py"]

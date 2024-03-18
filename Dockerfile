# Use a lightweight Python base image
FROM python:3.9

# Create a working directory for the notebooks
WORKDIR /app

# Copy the notebook directory and requirements.txt (if applicable)
COPY . ./
COPY requirements.txt ./  
# Optional: If you have specific Python dependencies
# Install required Python packages (if using requirements.txt)
RUN pip install -r requirements.txt  # Optional

# Install Jupyter Notebook and dependencies
RUN apt-get update && apt-get install -y \
    jupyter \
    python3-notebook \
    ipywidgets  # Optional: Additional Jupyter libraries you might need

# Expose Jupyter Notebook port (default: 8888)
EXPOSE 8888

# Start Jupyter Notebook server when the container runs
CMD ["jupyter", "notebook", "--allow-root", "--ip=0.0.0.0", "--port=8888"]


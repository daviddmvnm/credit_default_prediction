# Use a lightweight Python image
FROM python:3.13-slim


# Set the working directory in the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all necessary files into the container
COPY pipelines.py .
COPY Example_Usage.ipynb .   
COPY log_reg_oversample_model.joblib .
COPY meta_model_log_reg_oversampling_fp.joblib .
COPY application_test.csv .            
COPY credit_card_balance.csv .         
COPY readme.txt .

# Expose Jupyter's default port
EXPOSE 8888
RUN mkdir -p ~/.jupyter && echo "c.NotebookApp.token = ''" > ~/.jupyter/jupyter_notebook_config.py
# Set the default command to start Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

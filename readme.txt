###Credit Risk Prediction Project
Welcome to the Credit Risk Prediction Project, a student project aimed at creating a model to assess loan default risk. 
This project is built within a Docker environment for easy setup and consistency across various machines.

###Project Overview
This project involves a comprehensive analysis and modeling approach to predict credit risk. It includes:
Data preprocessing, feature engineering, and model training
Integration of a meta-model to improve prediction accuracy for false positives
Visualizations and metrics to showcase model performance
For detailed documentation on the project structure, methodology, and results, please refer to the full documentation: 
https://docs.google.com/document/d/1n0rrAw4s099t8K07tVOsV825aBVG7U34CedBlikARcU/edit?tab=t.0#heading=h.7766wl1f53ni

###Usage (if you're seeing this from Docker)
Prerequisites:
Make sure you have Docker installed on your machine. If you are unfamiliar with Docker, consult Docker's official installation guide.

Running the Project
To start the Jupyter Notebook server within the Docker container, use the following command:

bash
docker run -p 8888:8888 credit_risk_project_image

This command exposes the Jupyter Notebook on your local machine at localhost:8888. Once the server is running, 
you can open the provided link in your terminal output to access the project files in Jupyter Notebook.


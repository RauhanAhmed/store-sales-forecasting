
# Dynamic Sales Forecasting with Multiple-Multivariate Time Series

This Python, [ML-based API](https://store-sales-api.onrender.com) with an impressive Normalised Mean Squared Error of 0.039 and Normalised Mean Absolute Error of 0.10 for forecasts of this multiple, multivariate timeseries data on a 15-day window. This POC aims to empowers retailers with data-driven sales forecasting to optimize inventory management and improve profitability. 
This comprehensive solution predicts short-term sales (up to 30 days) with outstanding accuracy and performance metrics. By integrating seamlessly with existing workflows, it enables retailers to:

- Make informed inventory decisions.
- Reduce stockouts and overstocking.
- Improve operational efficiency.
- Boost customer satisfaction.

## Screenshots and Demo

![Prediction Accuracy](https://i.ibb.co/VYv9wcv/Untitled.png)
![FastAPI Swagger UI](https://i.ibb.co/ZM7xm7c/Mozilla-Firefox-2023-12-19-23-06-29.gif)


## Model Performance Metrics (15-day forecast)
| MSE<br>(on normalised data) | MAE<br>(on normalised data) |
|------------------|------------------|
| 0.03967  | 0.10960  |

## The Approach

- Fetching of the data from the MongoDB database using Python's PyMongo library.
- Performing in-depth analysis of data in a Jupyter Environment.
- Performing Feature Engineering to prepare the data for modelling.
- Forecasting oil prices using a LightGBM Model for preparing the covariates for the actual model.
- Fitting another LightGBM Model to forecast sales with the target values and covariates.
- Converting the Jupyter code to modular format for building code pipelines efficiently.
- Setting up the github repository, dockerhub repository using git and docker dekstop.
- Running DVC and Git on the built artifacts during the training process for file versioning and tracking.
- Creating the REST API using Python's FastAPI.
- Creating a Dockerfile for building the docker image for containerization of the api.
- Signing up and logging in on [Render Cloud]("https://render.com/") and creating a web service with the configuration of dockerized deployments.
- Setting up Github Actions and creating a workflow with the jobs of building the docker image, pushing the image to Dockerhub, and finally use Render Cloud's webhook to trigger deployment of the api once a push is made to dockerhub.


## Tech Stack

Built on a robust open-source ecosystem, it leverages:

- **Darts :** For efficient time series operations and forecasting.
- **MongoDB :** For storage and retrieval of data.
- **LightGBM :** To accurately predict covariate and target features.
- **Scikit-learn :** For creating data pipelines.
- **DVC, Git, and Github :** For seamless data and code versioning.
- **Evidently AI:** To check for data drift/target drift.
- **FastAPI :** For building a user-friendly API for model accessibility.
- **Docker and Dockerhub :** For secure and streamlined deployment to the Render Cloud Platform.
- **Github Actions :** For automating the CI/CD pipeline.
- **Render Cloud :** A PaaS for deployment of web apps, api's, etc.


## Run Locally

Clone the project

```bash
  git clone https://github.com/rauhanahmed/store-sales-forecasting
```

Go to the project directory

```bash
  cd store-sales-forecasting
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Start the server

```bash
  uvicorn app:app --reload
```

## Run the Train Pipeline

For running the train pipeline, follow the below steps:

- Create a [MongoDB Atlas]("https://www.mongodb.com/cloud/atlas/register") account, or sign in if already exists.
- Create an environment file, named *secrets.env* in the project's main directory with the keys of environment variables as *MONGODB_USERNAME* and *MONGODB_PASSWORD* and fill their respective values in [URL encoded format]("https://www.mongodb.com/docs/atlas/troubleshoot-connection/#special-characters-in-connection-string-password")
- Install the project's requirements by using the command:
```bash
pip install -r requirements.txt
```
- run the train_pipeline by using the command:
```bash
!python /src/pipelines/train_pipeline.py
```



## Deployment

For deployment of the application via dockers, follow the below steps:

- Create a docker image of the application using the below command, replacing <IMAGE_NAME> and <TAG> after installing docker desktop and signing up on [DockerHub](https://dockerhub.com).
 ```bash
 docker build -t <IMAGE_NAME>:<TAG> .
 ```
 - Test the container locally
 ```bash
 docker run -p 8000:8000 <IMAGE_NAME>:<TAG>
 ```
 - Push to DockerHub replacing <DOCKERHUB_USERNAME>, <IMAGE_NAME> and the <TAG> with the dockerhub username, image name and the image tag respectively 
 ```bash
 docker push <DOCKERHUB_USERNAME>/<IMAGE_NAME>:<TAG>
 ```
 - Sign up on [Render Cloud](https://render.com)
 - Create a Web Service on Render and choose to deploy via dockers
 - Specify the url of the dockerhub repository and start build
- Use the URL provided by render after successful deployment of application after few minutes :)
## Connect with me

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/)

[![portfolio](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/rauhanahmed/)

[![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://instagram.com/ahmed.rauhan)


## Support

For support, email rauhaan.siddiqui@gmail.com. I would be happy to help!


## License

[MIT](https://choosealicense.com/licenses/mit/)


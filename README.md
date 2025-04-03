# dynamic_nba_scheduling
## Dynamic NBA Scheduling Using a Competitive Balance Approach

### Value Proposition
We are going to create a win probability model that will be leveraged to dynamically schedule games to optimize viewership and fan engagement. Currently, the NBA creates their schedule in advance, with preset deals with national television networks, as the non-ML approach. In very few cases, the league may “flex” games to change their scheduling, but this is done on an ad-hoc basis, with no clear defined metrics to change the scheduling. The status quo can create a problem during the season in which two teams who are unevenly matched may compete in primetime and nationally televised games, creating a gap in entertainment value. If we are able to capture the win probability of NBA games, we can shuffle the daily schedule to prioritize evenly matched games to be played in primetime and nationally televised scenarios. We will judge ourselves on the ability to drive viewership to primetime games, which we will not realize in this project.

### Contributors

| Name             | Responsible for                                                      | Link to their commits in this repo |
|------------------|----------------------------------------------------------------------|------------------------------------|
| All team members | Problem definition, setup, integration, continuous X (Units 1, 2, 3) |               TBD                  |
| Will Calandra    | Model training (Units 4 + 5)                                         |               TBD                  |
| Lake Wang        | Model serving and monitoring (Units 6 + 7)                           |               TBD                  |
| Jason Moon       | Data pipeline (Unit 8)                                               |               TBD                  |


### System diagram




### Summary of outside materials


|          | How it was created | Conditions of use |
|----------------------------------------------|------------------------------------------------------------------------------|-------------------|
|[nba_api](https://github.com/swar/nba_api)    | This is an API for nba.com, with the aim to make the NBA APIs easy to use.   | Since we are pitching this to be used by the NBA as an internal tool, we should be in compliance and have consent from the league. This is open source under the MIT License, we must not reproduce these materials for commercial purposes, and we should cite the NBA when using the statistics, as they own them. This is not permissible for gambling or fantasy sports. We will follow the NBA terms of use (https://www.nba.com/termsofuse#nba-statistics). |
|[Pytorch](https://github.com/pytorch/pytorch) | This was created by the developers community                                 |    Open source, where generally, as long as we don’t generate or process inappropriate content, we are within the guidelines for use under the Linux Foundation.   |


### Summary of infrastructure requirements
| Requirement     | How many/when                                     | Justification |
|-----------------|---------------------------------------------------|---------------|
| `m1.medium` VMs | 2 for entire project duration    | We need a VM each for model training and model serving |
| 2 Nvidia A100 40GB GPUs (`liqid01 or 02`) | 3 hour block twice a week   | We plan to use DDP methods to optimize our training process, so we will need 2 GPU nodes. We think training for 3 hours should be enough for our model. We will use the second session for backup, freeing up resources for the rest of the class if we do not need it.    |
| Floating IPs    | 1 for entire project duration, 1 for sporadic use | We need to have a floating IP available for the persistent storage to communicate with VMs, and sporadic use when we train and serve the models on VMs. |
| Persistent Storage | 1 for entire project duration | We need the persistent storage for our data store, model store, and artifacts store to store the data and the model information persistently throughout the whole project duration. |

### Detailed design plan

#### Model training and training platforms
For the modeling approach, we will test a feed-forward PyTorch neural network model (FFN) on our feature set of NBA team and player statistics. Once we have a good estimate of the win probabilities for each game, we will match the closest games with the primetime scheduling slots. We will try a series of experiments where we distribute the data across GPUs (DDP) in addition to the hyperparameter tuning as logged in MLFlow, taking advantage of PyTorch Lightning for automatic experiment logging. We will also try batching to speed up performance for our experiments. As our model is unlikely to be large, we will not try the FSDP approach. Again, we will be sure to store our experimental results and artifacts in our artifact store on persistent storage on Chameleon.

#### Model serving and monitoring platforms
We will integrate our win rate prediction model and scheduling model into a single API endpoint for efficient serving. The requirements include maintaining a medium model size and ensuring acceptable latency, given the scheduling context. To optimize model performance, we will explore techniques such as reduced precision for the win rate model. On the system level, we will consider using FastAPI and implementing dynamic batching to accommodate the required concurrency.
We will complete the difficulty point of developing and evaluating optimized server-grade GPU and server-grade CPU serving options, comparing our results. 
We will evaluate the win rate model through automated offline evaluation, using metrics such as accuracy for overall classification and, potentially, for top franchises. We will also investigate misclassified cases to gain insights into model performance. If retraining is deemed necessary, the updated model will be registered. Following this, we will conduct a load test in the staging environment. Online evaluation will involve simulating matchups to assess the model's performance. Given the dynamic nature of this project, all production data will be stored for potential retraining. Feedback for the win rate model will be gathered in the form of actual game results. Business metrics will focus on fan engagement, viewership, and attendance, as measured by the NBA, with these insights being shared with the engineering team.


#### Data pipeline
First, we will create a volume with block storage to create a persistent storage including the data, as well as the model store and the artifacts store. From our dataset API, we will create an ETL pipeline using Airflow to load the processed data to our data repository. Our data repository will be a PostgreSQL relational database. Our offline dataset consists of NBA team statistics, NBA player statistics, and NBA game schedules. We will generate synthetic online data by randomly generating from the distributions of variables of the datasets. From the original data, we plan to experiment with feature engineering to increase our prediction performance.

#### Continuous X
We will use IaC by defining our infrastructure configuration in version control, potentially with python-chi, stored in Git. We will automate infrastructure setup and configuration with tools such as Ansible to eliminate manual intervention. Our project will use immutable infrastructure, use docker containers with all services, and follow a micro architecture for modularity. We will implement a CI/CD pipeline to automate model retraining, evaluation, optimization, and staged deployment. We will establish staging, canary, and production environments.

# Recommender Systems class - Project 1

Author: Kacper Kociubiński

Adam Mickiewicz University, Faculty of Mathematics and Computer Science, Computer Science, Recmmendation Systems Class - 2021

## Content-based recommender

Implementation of the content-base recommendation algorithm and the extraction of item and user features.
Project is based on data related to hotel rentals.
Project score is based on comparing own algorithm compared to the amazom algorithm on HR@10 metric. 
Implementation includes:
- loading the data,
- data preprocessing,
- extraction of item and user features,
- content-base recommendation algorithm,
- parameters tune,
- evaluation,
- compere between own and amazom algorithm.

## Preparing your computer

1. Install [Anaconda](https://www.anaconda.com/products/individual) with Python 3.8.


2. Install [Git](https://git-scm.com/downloads).


3. Install [Jupyter](https://jupyter.org/install).


4. Fork this repository to your GitHub account.


5. Go to the chosen folder on your machine where you want to have a local copy of the repository. Right-click in the folder and from the context menu choose "Git Bash Here". Run the following command to clone the forked repository on your GitHub account to your local machine:

	<pre>git clone <i>your_repository_address_which_you'll_find_in_your_github</i></pre>

	Alternatively, open Git Bash (installed with Git), change the path to the folder where you want to have a local copy of the repository, execute the above command.


6. Prepare your conda environment (instructions given for Windows, but it should be similar on other systems):

	1. Open Anaconda Prompt as administrator.

	2. Make sure you're in the repository main folder. Run the following command:
			
			conda env create --name rs-class-env -f environment.yml

		You can replace *rs-class-env* with your own environment name.
		
		You may need to install a C++ compiler to install certain packages.


7. In Git Bash open the repository folder and activate just created environment with the following command:

		conda activate rs-class-env
	

8. In Git Bash type:

		jupyter notebook

	A new tab with Jupyter Notebook should open in your browser.


9. In Jupyter Notebook open project_1_data_preparation.ipynb and ran all cells.

10. Next in Jupyter Notebook open project_1_recommender_and_evaluation.ipynb and ran all cells.

11. Last cell in project_1_recommender_and_evaluation.ipynb has comparison in HR@10 form MLPRegressorCBUIRecommender, AmazonRecommender, LinearRegressionCBUIRecommender.




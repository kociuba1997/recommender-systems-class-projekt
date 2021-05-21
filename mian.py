import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown, display, HTML
from collections import defaultdict

from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

from recommenders.recommender import Recommender

# Fix the dying kernel problem (only a problem in some installations - you can remove it, if it works without it)
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

data_path = os.path.join("data", "hotel_data")

interactions_df = pd.read_csv(os.path.join(data_path, "hotel_data_interactions_df.csv"), index_col=0)
base_item_features = ['term', 'length_of_stay_bucket', 'rate_plan', 'room_segment', 'n_people_bucket', 'weekend_stay']

column_values_dict = {
    'term': ['WinterVacation', 'Easter', 'OffSeason', 'HighSeason', 'LowSeason', 'MayLongWeekend', 'NewYear', 'Christmas'],
    'length_of_stay_bucket': ['[0-1]', '[2-3]', '[4-7]', '[8-inf]'],
    'rate_plan': ['Standard', 'Nonref'],
    'room_segment': ['[0-160]', '[160-260]', '[260-360]', '[360-500]', '[500-900]'],
    'n_people_bucket': ['[1-1]', '[2-2]', '[3-4]', '[5-inf]'],
    'weekend_stay': ['True', 'False']
}

interactions_df.loc[:, 'term'] = pd.Categorical(
    interactions_df['term'], categories=column_values_dict['term'])
interactions_df.loc[:, 'length_of_stay_bucket'] = pd.Categorical(
    interactions_df['length_of_stay_bucket'], categories=column_values_dict['length_of_stay_bucket'])
interactions_df.loc[:, 'rate_plan'] = pd.Categorical(
    interactions_df['rate_plan'], categories=column_values_dict['rate_plan'])
interactions_df.loc[:, 'room_segment'] = pd.Categorical(
    interactions_df['room_segment'], categories=column_values_dict['room_segment'])
interactions_df.loc[:, 'n_people_bucket'] = pd.Categorical(
    interactions_df['n_people_bucket'], categories=column_values_dict['n_people_bucket'])
interactions_df.loc[:, 'weekend_stay'] = interactions_df['weekend_stay'].astype('str')
interactions_df.loc[:, 'weekend_stay'] = pd.Categorical(
    interactions_df['weekend_stay'], categories=column_values_dict['weekend_stay'])

display(HTML(interactions_df.head(15).to_html()))


def prepare_users_df(interactions_df):

    #df = interactions_df.groupby('user_id')['term'].apply(lambda x: x.mode().iat[0]).reset_index()
    one_hot_df = pd.get_dummies(data=interactions_df, columns=['term', 'room_segment', 'rate_plan', 'length_of_stay_bucket', 'n_people_bucket', 'weekend_stay'], prefix='user')
    users_df = one_hot_df.drop(['item_id'], axis = 1, errors='ignore')
    user_features = list(users_df.columns.values)
    user_features.remove('user_id')

    return users_df, user_features


users_df, user_features = prepare_users_df(interactions_df)

print(user_features)

display(HTML(users_df.loc[users_df['user_id'].isin([706, 1736, 7779, 96, 1, 50, 115])].head(15).to_html()))


def prepare_items_df(interactions_df):
    #df = interactions_df.groupby('item_id')['term'].apply(lambda x: x.mode().iat[0]).reset_index()
    one_hot_df = pd.get_dummies(data=interactions_df, columns=['term', 'room_segment', 'rate_plan', 'length_of_stay_bucket', 'n_people_bucket', 'weekend_stay'], prefix='item')
    items_df = one_hot_df.drop(['user_id'], axis = 1, errors='ignore')
    item_features = list(items_df.columns.values)
    item_features.remove('item_id')

    return items_df, item_features


items_df, item_features = prepare_items_df(interactions_df)

print(item_features)

display(HTML(items_df.loc[items_df['item_id'].isin([0, 1, 2, 3, 4, 5, 6])].head(15).to_html()))


class ContentBasedUserItemRecommender(Recommender):
    """
    Linear recommender class based on user and item features.
    """

    def __init__(self, seed=6789, n_neg_per_pos=5):
        """
        Initialize base recommender params and variables.
        """
        self.model = MLPRegressor(random_state=1, max_iter=2, verbose=True)
        self.n_neg_per_pos = n_neg_per_pos

        self.recommender_df = pd.DataFrame(columns=['user_id', 'item_id', 'score'])
        self.users_df = None
        self.user_features = None

        self.seed = seed
        self.rng = np.random.RandomState(seed=seed)

    def fit(self, interactions_df, users_df, items_df):
        """
        Training of the recommender.

        :param pd.DataFrame interactions_df: DataFrame with recorded interactions between users and items
            defined by user_id, item_id and features of the interaction.
        :param pd.DataFrame users_df: DataFrame with users and their features defined by user_id and the user feature columns.
        :param pd.DataFrame items_df: DataFrame with items and their features defined by item_id and the item feature columns.
        """

        interactions_df = interactions_df.copy()

        # Prepare users_df and items_df

        users_df, user_features = prepare_users_df(interactions_df)

        self.users_df = users_df
        self.user_features = user_features

        items_df, item_features = prepare_items_df(interactions_df)
        items_df = items_df.loc[:, ['item_id'] + item_features]

        # Generate negative interactions

        interactions_df = interactions_df.loc[:, ['user_id', 'item_id']]

        interactions_df.loc[:, 'interacted'] = 1

        negative_interactions = []

        # Nie ma co usuwać z posible_items_id zużytych itemów, bo dla
        # user 706 nie starczy
        # Write your code here
        # Generate tuples (user_id, item_id, 0) for pairs (user_id, item_id) which do not
        # appear in the interactions_df and add those tuples to the list negative_interactions.
        # Generate self.n_neg_per_pos * len(interactions_df) negative interactions
        # (self.n_neg_per_pos per one positive).
        # Make sure the code is efficient and runs fast, otherwise you will not be able to properly tune your model.

        unique_item_id = interactions_df['item_id'].drop_duplicates().values.tolist()

        for user_id, item_gruped_by_user in interactions_df.groupby('user_id'):
            items_by_user = item_gruped_by_user['item_id'].values.tolist()
            posible_items_id = list(set(unique_item_id) - set(items_by_user))
            for item in items_by_user:
                for x in range(self.n_neg_per_pos):
                    random_item_id = self.rng.choice(posible_items_id)
                    negative_interactions.append((user_id, random_item_id, 0))

        interactions_df = pd.concat(
            [interactions_df, pd.DataFrame(negative_interactions, columns=['user_id', 'item_id', 'interacted'])])

        # Get the input data for the model

        interactions_df = pd.merge(interactions_df, users_df, on=['user_id'])
        interactions_df = pd.merge(interactions_df, items_df, on=['item_id'])

        x = interactions_df.loc[:, user_features + item_features].values
        y = interactions_df['interacted'].values

        self.model.fit(x, y)

    def recommend(self, users_df, items_df, n_recommendations=1):
        """
        Serving of recommendations. Scores items in items_df for each user in users_df and returns
        top n_recommendations for each user.

        :param pd.DataFrame users_df: DataFrame with users and their features for which recommendations should be generated.
        :param pd.DataFrame items_df: DataFrame with items and their features which should be scored.
        :param int n_recommendations: Number of recommendations to be returned for each user.
        :return: DataFrame with user_id, item_id and score as columns returning n_recommendations top recommendations
            for each user.
        :rtype: pd.DataFrame
        """

        # Clean previous recommendations (iloc could be used alternatively)
        self.recommender_df = self.recommender_df[:0]

        # Write your code here
        # Prepare users_df and items_df
        # For users_df you just need to merge user features from self.users_df to users_df
        # (the users for which you generate recommendations)
        # For items you have to apply the prepare_items_df method to items_df.

        users_df = pd.merge(self.users_df, users_df, on=['user_id'])
        items_df, item_features = prepare_items_df(items_df)

        # Score the items

        recommendations = pd.DataFrame(columns=['user_id', 'item_id', 'score'])

        for ix, user in users_df.iterrows():

            # Write your code here
            # Create a Carthesian product of users from users_df and items from items_df

            user['key'] = 1
            items_df['key'] = 1
            user = user.to_frame().transpose()
            carthesian_user_items = user.merge(items_df, on='key').drop("key", 1)

            # Write your code here
            # Use self.model.predict method to calculate scores for all records in the just created DataFrame
            # of users and items

            x = carthesian_user_items.loc[:, user_features + item_features].values
            scores = self.model.predict(x)

            # Write your code here
            # Obtain item ids with the highest score and save those ids under the chosen_ids variable
            # Do not exclude already booked items.
            chosen_ids = (-scores).argsort()[:n_recommendations]

            recommendations = []
            for item_id in chosen_ids:
                recommendations.append(
                    {
                        'user_id':  user['user_id'].iloc[0],
                        'item_id': item_id,
                        'score': scores[item_id]
                    }
                )

            user_recommendations = pd.DataFrame(recommendations)

            self.recommender_df = pd.concat([self.recommender_df, user_recommendations])

        return self.recommender_df


class LinearRegressionCBUIRecommender(ContentBasedUserItemRecommender):
    """
    Linear regression recommender class based on user and item features.
    """

    def __init__(self, seed=6789, n_neg_per_pos=5, **model_params):
        """
        Initialize base recommender params and variables.
        """
        super().__init__(seed=seed, n_neg_per_pos=n_neg_per_pos)
        self.model = LinearRegression()


class SVRCBUIRecommender(ContentBasedUserItemRecommender):
    """
    SVR recommender class based on user and item features.
    """

    def __init__(self, seed=6789, n_neg_per_pos=5, **model_params):
        """
        Initialize base recommender params and variables.
        """
        super().__init__(seed=seed, n_neg_per_pos=n_neg_per_pos)
        if 'kernel' in model_params:
            self.kernel = model_params['kernel']
        else:
            self.kernel = 'rbf'
        if 'C' in model_params:
            self.C = model_params['C']
        else:
            self.C = 1.0
        if 'epsilon' in model_params:
            self.epsilon = model_params['epsilon']
        else:
            self.epsilon = 0.1
        self.model = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon, verbose=1)


class RandomForestCBUIRecommender(ContentBasedUserItemRecommender):
    """
    Random forest recommender class based on user and item features.
    """

    def __init__(self, seed=6789, n_neg_per_pos=5, **model_params):
        """
        Initialize base recommender params and variables.
        """
        super().__init__(seed=seed, n_neg_per_pos=n_neg_per_pos)
        if 'n_estimators' in model_params:
            self.n_estimators = int(model_params['n_estimators'])
        else:
            self.n_estimators = 100
        if 'max_depth' in model_params:
            self.max_depth = int(model_params['max_depth'])
        else:
            self.max_depth = 30
        if 'min_samples_split' in model_params:
            self.min_samples_split = int(model_params['min_samples_split'])
        else:
            self.min_samples_split = 30
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth, min_samples_split=self.min_samples_split)


class XGBoostCBUIRecommender(ContentBasedUserItemRecommender):
    """
    XGBoost recommender class based on user and item features.
    """

    def __init__(self, seed=6789, n_neg_per_pos=5, **model_params):
        """
        Initialize base recommender params and variables.
        """
        super().__init__(seed=seed, n_neg_per_pos=n_neg_per_pos)
        if 'n_estimators' in model_params:
            self.n_estimators = int(model_params['n_estimators'])
        else:
            self.n_estimators = 100
        if 'max_depth' in model_params:
            self.max_depth = int(model_params['max_depth'])
        else:
            self.max_depth = 30
        if 'min_samples_split' in model_params:
            self.min_samples_split = int(model_params['min_samples_split'])
        else:
            self.min_samples_split = 30
        if 'learning_rate' in model_params:
            self.learning_rate = model_params['learning_rate']
        else:
            self.learning_rate = 30
        self.model = GradientBoostingRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth, min_samples_split=self.min_samples_split,
            learning_rate=self.learning_rate, verbose=1)

items_df = interactions_df.loc[:, ['item_id'] + base_item_features].drop_duplicates()

from evaluation_and_testing.testing import evaluate_train_test_split_implicit

seed = 6789

from hyperopt import hp, fmin, tpe, Trials
import traceback


def tune_recommender(recommender_class, interactions_df, items_df,
                     param_space, max_evals=1, show_progressbar=True, seed=6789):
    # Split into train_validation and test sets

    shuffle = np.arange(len(interactions_df))
    rng = np.random.RandomState(seed=seed)
    rng.shuffle(shuffle)
    shuffle = list(shuffle)

    train_test_split = 0.8
    split_index = int(len(interactions_df) * train_test_split)

    train_validation = interactions_df.iloc[shuffle[:split_index]]
    test = interactions_df.iloc[shuffle[split_index:]]

    # Tune

    def loss(tuned_params):
        recommender = recommender_class(seed=seed, **tuned_params)
        hr1, hr3, hr5, hr10, ndcg1, ndcg3, ndcg5, ndcg10 = evaluate_train_test_split_implicit(
            recommender, train_validation, items_df, seed=seed)
        return -hr10

    n_tries = 1
    succeded = False
    try_id = 0
    while not succeded and try_id < n_tries:
        try:
            trials = Trials()
            best_param_set = fmin(loss, space=param_space, algo=tpe.suggest,
                                  max_evals=max_evals, show_progressbar=show_progressbar, trials=trials, verbose=True)
            succeded = True
        except:
            traceback.print_exc()
            try_id += 1

    if not succeded:
        return None

    # Validate

    recommender = recommender_class(seed=seed, **best_param_set)

    results = [[recommender_class.__name__] + list(evaluate_train_test_split_implicit(
        recommender, {'train': train_validation, 'test': test}, items_df, seed=seed))]

    results = pd.DataFrame(results,
                           columns=['Recommender', 'HR@1', 'HR@3', 'HR@5', 'HR@10', 'NDCG@1', 'NDCG@3', 'NDCG@5',
                                    'NDCG@10'])

    display(HTML(results.to_html()))

    return best_param_set


#########################################Wyrzuciłem tuningi###############################

cb_user_item_recommender = ContentBasedUserItemRecommender(
    **{'n_neg_per_pos': 9})  # Initialize your recommender here with the best params from tuning

# Give the name of your recommender in the line below
linear_cbui_tts_results = [['LinearRegressionCBUIRecommender'] + list(evaluate_train_test_split_implicit(
    cb_user_item_recommender, interactions_df, items_df))]

linear_cbui_tts_results = pd.DataFrame(
    linear_cbui_tts_results, columns=['Recommender', 'HR@1', 'HR@3', 'HR@5', 'HR@10', 'NDCG@1', 'NDCG@3', 'NDCG@5', 'NDCG@10'])

display(linear_cbui_tts_results.to_string())
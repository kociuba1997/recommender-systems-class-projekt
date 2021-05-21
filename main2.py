import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown, display, HTML
from collections import defaultdict

# Fix the dying kernel problem (only a problem in some installations - you can remove it, if it works without it)
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from data_preprocessing.dataset_specification import DatasetSpecification
from data_preprocessing.data_preprocessing_toolkit import DataPreprocessingToolkit
from data_preprocessing.people_identifier import PeopleIdentifier


data_path = os.path.join("data", "hotel_data")

original_data = pd.read_csv(os.path.join(data_path, "hotel_data_original.csv"), index_col=0)

original_data = original_data.replace({"\\N": ""})
original_data = original_data.fillna("")

numeric_columns = ["n_people", "n_children_1", "n_children_2", "n_children_3",
                   "discount", "accomodation_price", "meal_price", "service_price",
                   "paid"]

for column in numeric_columns:
    original_data.loc[:, column] = pd.to_numeric(original_data.loc[:, column], errors="coerce")

original_data = original_data.astype(
        {
            "date_from": np.datetime64,
            "date_to": np.datetime64,
            "booking_time": np.datetime64,
            "booking_date": np.datetime64,
            "n_people": np.int64,
            "n_children_1": np.int64,
            "n_children_2": np.int64,
            "n_children_3": np.int64,
            "discount": np.float64,
            "accomodation_price": np.float64,
            "meal_price": np.float64,
            "service_price": np.float64,
            "paid": np.float64,
        }
    )

display(HTML(original_data.head(15).to_html()))

preprocessed_data = original_data.copy()

dataset_specification = DatasetSpecification()
dp_toolkit = DataPreprocessingToolkit()

id_column_names = dataset_specification.get_id_columns()

people_identifier = PeopleIdentifier()
preprocessed_data = people_identifier.add_pid(preprocessed_data, id_column_names, "user_id")

preprocessed_data = dp_toolkit.fix_date_to(preprocessed_data)
preprocessed_data = dp_toolkit.add_length_of_stay(preprocessed_data)  # Code this method
preprocessed_data = dp_toolkit.add_book_to_arrival(preprocessed_data)
preprocessed_data = dp_toolkit.add_nrooms(preprocessed_data)
preprocessed_data = dp_toolkit.add_weekend_stay(preprocessed_data)
preprocessed_data = dp_toolkit.clip_book_to_arrival(preprocessed_data)

preprocessed_data = dp_toolkit.sum_npeople(preprocessed_data)

preprocessed_data = dp_toolkit.filter_out_company_clients(preprocessed_data)
preprocessed_data = dp_toolkit.filter_out_long_stays(preprocessed_data)

preprocessed_data = dp_toolkit.aggregate_group_reservations(preprocessed_data)

preprocessed_data = dp_toolkit.add_night_price(preprocessed_data)  # Code this method (remember that there can be many rooms)

preprocessed_data = preprocessed_data.reset_index(drop=True)

assert preprocessed_data.iloc[1]['length_of_stay'] == 3
assert preprocessed_data.iloc[2]['length_of_stay'] == 2
assert preprocessed_data.iloc[3]['length_of_stay'] == 7

assert preprocessed_data.iloc[0]['night_price'] == 330.76
assert preprocessed_data.iloc[1]['night_price'] == 231.13
assert preprocessed_data.iloc[2]['night_price'] == 183.40

display(HTML(preprocessed_data.head(15).to_html()))

preprocessed_data = dp_toolkit.map_date_to_term_datasets(preprocessed_data)
preprocessed_data = dp_toolkit.map_length_of_stay_to_nights_buckets(preprocessed_data)
preprocessed_data = dp_toolkit.map_night_price_to_room_segment_buckets(preprocessed_data)  # Code this method
preprocessed_data = dp_toolkit.map_npeople_to_npeople_buckets(preprocessed_data)

assert preprocessed_data.iloc[0]['room_segment'] == '[260-360]'
assert preprocessed_data.iloc[1]['room_segment'] == '[160-260]'
assert preprocessed_data.iloc[4]['room_segment'] == '[0-160]'

preprocessed_data = dp_toolkit.map_item_to_item_id(preprocessed_data)

preprocessed_data.to_csv(os.path.join(data_path, "hotel_data_preprocessed.csv"))

display(HTML(preprocessed_data.head(15).to_html()))
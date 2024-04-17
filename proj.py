import pandas as pd
import snsql
from snsql import Privacy

from opendp.smartnoise.sql import PandasReader, PrivateReader
from opendp.smartnoise.metadata import CollectionMetadata


injuries_df = pd.read_csv('data/injuries.csv')
players_df = pd.read_csv('data/players.csv')

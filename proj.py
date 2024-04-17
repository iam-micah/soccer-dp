import pandas as pd
import snsql
from snsql import Privacy

from opendp.smartnoise.sql import PandasReader, PrivateReader
from opendp.smartnoise.metadata import CollectionMetadata

metadata = CollectionMetadata.from_file('metadata.yml')

injuries_df = pd.read_csv('data/injuries.csv')
players_df = pd.read_csv('data/players.csv')

privacy = Privacy(epsilon=1.0, delta=0)
reader = PandasReader(
    metadata, {'injuries': injuries_df, 'players': players_df})
private_reader = PrivateReader(reader, privacy)

query = "SELECT injury_type, COUNT(*) AS injury_count FROM injuries GROUP BY injury_type"
result = private_reader.execute(query)
print(result)

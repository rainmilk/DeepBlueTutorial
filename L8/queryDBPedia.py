import sparql
import json
import pandas as pd

endpoint = "http://dbpedia.org/sparql"
s = sparql.Service(endpoint)

movielens_tsv = 'MappingMovielens2DBpedia-1.2.tsv'

data = pd.read_table(movielens_tsv)
maxrow = data.shape[0]
print("Rows: %d"%maxrow)

dataset = []
for i, d in enumerate(data.itertuples(), 1):
    q = """PREFIX movie:<%s>
    select ?director ?writer ?starring
     { optional { movie: dbo:director ?director }
       optional { movie: dbo:writer ?writer }
       optional { movie: dbo:starring ?starring } }"""%(d.DBpedia_uri)

    result = s.query(q)
    cast = set()
    for row in result:
        if row[0]:  # director
            cast.add(row[0].value)
        if row[1]:  # writer
            cast.add(row[1].value)
        if row[2]:  # starring
            cast.add(row[2].value)

    dataset.append({'movie_id': str(d.item_id), 'cast': list(cast)})

    print(cast)
    print("Progress %d/%d"%(i, maxrow))

dataset_file = "movielens_cast.json"
with open(dataset_file, 'w') as jsonfile:
    json.dump({'data': dataset}, jsonfile)


# Using movielens_cast.json to construct a knowledge graph <movie, cast, person>






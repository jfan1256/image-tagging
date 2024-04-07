### image_feats.npy
- Contains image embeddings
- Shape = (4802, 256)

### query_feats.npy
- Contains generated query embeddings
- Shape = (4802, 256)

### tag_feats.npy
- Contains tag embeddings
- Shape = (4802, 256)

### id_and_tags.csv
- Contains index of image ids and 5 columns for 5 tags
- Shape = (5000, 6)

### querys.csv
- Contains index of image ids and 1 column for generated tags
- Shape = (5000, 2)

### selected_files.csv
- Contains 1 column for selected files in Kaggle dataset to process
- Shape = (5000, 1)

### bad_querys.csv
- Contains 1 column for image ids that have bad generated queries (as in query word length over 10)
- Shape = (198, 1)

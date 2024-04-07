## Image Tagging

This repo provides the code to parallelize OpenAI ChatGPT query generation, preprocessing, and embedding retrieval via DLIP (Distilled BLIP) to generate a suitable dataset to train a Multimodal LLM model for Image-Text tagging loss.   
  
The dataset is provided here: https://www.kaggle.com/datasets/greg115/various-tagged-images?resource=download. 

### Steps to follow:
1. Download the dataset (do not unzip it)
2. Clone the repo
3. Move the downloaded zip (should be called archive.zip) under the /data directory
4. Run the image_tagging.ipynb notebook (you can specify how many images to process under the "Retrieve N Images File Randomly...")
5. All files will be saved under /data directory
6. Happy Training!

### Note:
The model used for embedding retrieval was trained via my other repo called "dlip-v2". You can train your own DLIP model through that repo, or you can utilize your preexisting Image-Text models such as BLIP, CLAP, etc.
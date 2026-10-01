# real-ai-face-detection

This is a simple game where player needs to find whether the given image as AI or real.

One photo will be shown and player needs to choose either Real or AI button and then the game says whether the image is real or AI. It also tells whether the model predicted correctly or not.

After 10 photos, summary shows whether player or the model wins and shows the accuracy of both player and model.

Model used here is MobileNet-V3 finetuned upon ImageNet to predict whether given image is AI or real.

This can be played directly using - https://real-ai-face-detection.streamlit.app/

or to run locally, simply clone the repository and run,

``
pip install -r requirements.txt
``

``
streamlit app.py
``
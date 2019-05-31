# TPPNet
Temporal Pyramid Pooling Convolutional Neural Network for Cover Song Identification

## Environment
python  --  3
pytorch --  1.0
librosa --  0.63

## Dataset
Second Hand Songs 100K (SHS100K), which is collected from Second Hand Songs website. 
We provide the download link in "data/SHS100K_url". The total size of the music is about 400G.

## Generate CQT
You can utilize "data/gencqt.py" to get CQT features from your own audio.

## Train 

python main.py multi_train --model='CQTTPPNet' --batch_size=32 --load_latest=False --notes='experiment0'

## Test

python main.py test --model='CQTTPPNet' --load_model_path = 'check_points/best.pth'

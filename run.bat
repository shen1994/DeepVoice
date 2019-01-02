activate py3
# this command will remove the file which is between 2-12s
python speech_utils.py
# this command will generate file-mapping
# specially this command will be operated only once!!!
python text_utils.py
# get one model
python train.py
# to word2vector
# python train_word2vec.py
# run to get the text
python test.py
# run this command to catch sound in real time
python speak.py
# if you want to get the exe, please run this!
PyInstaller --clean --win-private-assemblies -n PICO_DS2 -i D:/logo.jpg test.py

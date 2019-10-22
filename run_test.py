from organ import ORGAN

model = ORGAN('test', 'music_metrics')             # Loads a ORGANIC with name 'test', using music metrics
model.load_training_set('data/music_random.pkl') # Loads the training set
model.set_training_program(['tonality'], [50])     # Sets the training program as 50 epochs with the tonality metric
model.load_metrics()                               # Loads all the metrics
model.train()                                      #
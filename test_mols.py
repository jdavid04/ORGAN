import organ
from organ import ORGAN

model = ORGAN('test', params={'PRETRAIN_DIS_EPOCHS': 2, 'PRETRAIN_GEN_EPOCHS' : 2})
model.load_training_set('data/zinc.csv', num_samples=100)
model.set_training_program(['novelty'], [10])
model.load_metrics()
model.train(ckpt_dir='ckpt')

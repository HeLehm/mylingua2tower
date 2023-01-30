from src.mind import MINDDataLoader
from src.trainer import Trainer, Model

from transformers import BertConfig

train_mind = MINDDataLoader(batch_size=16, validation=False)
eval_mind = MINDDataLoader(batch_size=16, validation=True)

config = BertConfig.from_json_file("./config.json")
model = Model(config)

trainer = Trainer(model)

trainer.fit(epochs=3, train_loader=train_mind, valid_loader=eval_mind)
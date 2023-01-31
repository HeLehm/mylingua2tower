from src.content_tower import MHAContentTower
from src.user_tower import DummyUserTower
from src.utils import TimeDistributed, get_mind_iter, MINDIterator

from src.trainer_new import Trainer


from transformers import BertConfig

config = BertConfig.from_json_file("./config.json")


content_tower = MHAContentTower(config)


time_distributed_content_tower = TimeDistributed(content_tower)
user_tower = DummyUserTower(time_distributed_content_tower)

trainer : Trainer = Trainer(time_distributed_content_tower, user_tower)


mind_iter ,train_news ,train_behav ,_ ,_ = get_mind_iter(
    batch_size=32,
    max_title_length=128,
    history_size=10,
    word_dict=content_tower.word2idx_dict,
    npratio=4, # negative positive ratio (4 = 4 negtuve, 1 positive)
)
data_iter = mind_iter.load_data_from_file(train_news, train_behav)

trainer.fit(data_iter, 10)



import random
from torch.utils.data import TensorDataset, DataLoader
import torch


def convert_triplets_into_index(triplets, tool2index):
    """convert triplets into index to cater to Torch Dataset"""
    # raw texts
    all_step_texts = [trip[0] for trip in triplets]

    formatted_triplets = [[i, tool2index[trip[1]], tool2index[trip[2]]] for i, trip in enumerate(triplets)]
    return all_step_texts, formatted_triplets


class TrainSampler:
    def __init__(self, raw_contents, num_negatives, sample_graph, tool2index, hard_negative=True, batch_size=512):
        self.raw_contents = raw_contents
        self.graph = sample_graph
        self.num_negatives = num_negatives
        self.tools = list(self.graph.keys())
        self.hard_negative = hard_negative
        self.batch_size = batch_size
        self.tool2index = tool2index

    def sample(self, new_graph=None, tmp_print=False, maximum=None, shuffle=False):
        if new_graph:
            self.graph = new_graph

        triplets = []
        for i, (step_text, pos_tool) in enumerate(self.raw_contents):
        # step_text = step_text.replace("Step ", "")
            for _ in range(self.num_negatives):
                neg_tool = random.choice(self.tools)
                while neg_tool == pos_tool:
                    neg_tool = random.choice(self.tools)
                if self.hard_negative:
                    if random.random() > 0.5 and len(self.graph[pos_tool]):
                        neg_tool = random.choice(self.graph[pos_tool])

            triplets.append([step_text, pos_tool, neg_tool])

            if tmp_print and i % 200 == 0:
                print(f"<{step_text}, POS-{pos_tool}, NEG-{neg_tool}>")

        random.shuffle(triplets)
        # print(triplets[:10])
        if maximum:
            triplets = triplets[:maximum]
        
        all_step_texts, formatted_triplets = convert_triplets_into_index(triplets, self.tool2index)

        train_data = TensorDataset(torch.LongTensor(formatted_triplets))
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=shuffle)
        return train_loader, all_step_texts
        
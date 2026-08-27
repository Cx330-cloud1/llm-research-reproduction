from transformers import RobertaModel, RobertaConfig, RobertaTokenizer
import torch.nn as nn
import torch
import numpy as np
import json
from modules.sddp import StructuredDialogueDiscourseParser
from modules.erc import SequentialERC
from modules.decoder import RobertaClassificationHead
from torch_geometric.nn.conv import RGATConv


class FFN(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, dropout):
        super(FFN, self).__init__()
        self.linear = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class GATLayer(nn.Module):
    def __init__(self, dim_in, dim_out, num_relations, dropout):
        super(GATLayer, self).__init__()
        self.conv = RGATConv(in_channels=dim_in, out_channels=dim_out, num_relations=num_relations)
        # self.dropout = nn.Dropout(dropout)
        self.ffn = FFN(dim_in=dim_out, dim_hidden=dim_in // 2, dim_out=dim_out, dropout=dropout)

    def forward(self, x, edge_index, edge_type):
        _x = x
        x, attention_weights = self.conv(x, edge_index, edge_type, return_attention_weights=True)
        x = x + _x
        # x = self.dropout(x)
        # x = self.ffn(x)
        return x, attention_weights


class RobertaBase(nn.Module):
    def __init__(self):
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # RoBERTa encoder
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.roberta_config = RobertaConfig.from_pretrained("roberta-base")
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def encode(self, texts):
        tokens = self.roberta_tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        outputs = self.roberta(input_ids=tokens["input_ids"].to(self.device),
                               attention_mask=tokens["attention_mask"].to(self.device))
        embeddings = outputs.last_hidden_state
        return embeddings[:, 0, :]  # take <s> token (equiv. to [CLS])

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')), strict=False)


class RobertaHeterogeneousGraph(RobertaBase):
    def __init__(self, args, lightmode=True):
        super().__init__()

        self.args = args
        self.lightmode = lightmode
        graph_dim = args.hg_dim

        if "esconv" in args.dataset:
            self.strategy2id = json.load(open('data/esconv/strategies.json', 'r'))
        elif "annomi" in args.dataset:
            self.strategy2id = json.load(open('data/annomi/strategies.json', 'r'))
        self.id2strategy = {v: k for k, v in self.strategy2id.items()}
        self.id2emotion = {0: 'Neutral', 1: 'Anger', 2: 'Disgust', 3: 'Fear', 4: 'Joy', 5: 'Sadness', 6: 'Surprise'}

        # Frozen Pre-trained Models
        if not lightmode:
            self.dialogue_parser = StructuredDialogueDiscourseParser(ckpt_path="pre_trained_models/sddp_stac")
            for name, param in self.dialogue_parser.model.named_parameters():
                param.requires_grad = False
            self.erc = SequentialERC()
            self.erc.load("pre_trained_models/sequential_erc_model.pth")
            for name, param in self.erc.named_parameters():
                param.requires_grad = False

        self.erc_prototypes = nn.Parameter(torch.randn((7, graph_dim)))
        self.softmax = nn.Softmax(dim=-1)
        self.scalar = 100
        self.t = nn.Parameter(torch.tensor(args.erc_temperature / self.scalar))

        # GCN layers
        encoder_hidden_size = self.roberta_config.hidden_size
        self.graph_relation_dict = {"Continuation": 0, "Question-answer_pair": 1, "Contrast": 2, "Q-Elab": 3,
                                    "Explanation": 4, "Comment": 5, "Background": 6, "Result": 7, "Correction": 8,
                                    "Parallel": 9, "Alternation": 10, "Conditional": 11, "Clarification_question": 12,
                                    "Acknowledgement": 13, "Elaboration": 14, "Narration": 15, "Special": 16,
                                    "Self": 17, "Inter": 18}
        self.graph_relation_dict_inverse = {v: k for k, v in self.graph_relation_dict.items()}
        self.conv1 = GATLayer(dim_in=graph_dim, dim_out=graph_dim, num_relations=len(self.graph_relation_dict.keys()),
                              dropout=0.2)
        self.conv2 = GATLayer(dim_in=graph_dim, dim_out=graph_dim, num_relations=len(self.graph_relation_dict.keys()),
                              dropout=0.2)
        self.conv3 = GATLayer(dim_in=graph_dim, dim_out=graph_dim, num_relations=len(self.graph_relation_dict.keys()),
                              dropout=0.2)
        self.dummy_embedding = nn.Parameter(torch.randn(graph_dim))
        self.strategy_embedding = nn.Embedding(num_embeddings=len(self.strategy2id.keys()), embedding_dim=graph_dim)
        self.node_position_embedding = nn.Embedding(num_embeddings=6, embedding_dim=graph_dim)

        # Classification head
        self.num_classes = len(self.strategy2id) - 1 if args.exclude_others else len(self.strategy2id)
        if args.ablation in ["no_graph", "flat"]:
            classifier_input_dim = self.roberta_config.hidden_size
        else:
            graph_readout_dim = graph_dim * 2 if args.ablation == "no_dummy" else graph_dim
            classifier_input_dim = self.roberta_config.hidden_size + graph_readout_dim

        self.classifier = RobertaClassificationHead(
            classifier_input_dim,
            self.num_classes
        )
        # self.classifier = RobertaClassificationHead(graph_dim, self.num_classes)
        # self.classifier = RobertaClassificationHead(self.roberta_config.hidden_size, self.num_classes)


    def forward(self, samples):
        flattened_contexts = []
        dialogues_for_parsing = []
        texts_for_erc = []
        erc_indices = []
        strategy_indices = []
        dialogue_sizes = []
        for i in range(len(samples["dialogue_history"])):
            strategy_history = [int(s.strip()) for s in samples["strategy_history"][i][1:-1].split(",")]
            utterances = samples["dialogue_history"][i].split("</s>")
            speakers = str(samples["speaker_turn"][i]).split(" ")
            dialogue_sizes.append(len(utterances))
            context = " ".join([f"[{speakers[j]}] {utterances[j]}" for j in range(len(utterances))])
            dialogue_for_parsing = []
            text_for_erc = ""
            erc_index = []
            strategy_index = []
            for j in range(len(utterances)):
                turn = {
                    "speaker": speakers[j],
                    "text": utterances[j]
                }
                dialogue_for_parsing.append(turn)
                text_for_erc += " </s> " + utterances[j]
                if speakers[j] == "seeker":
                    erc_index.append(j)
                else:
                    strategy = strategy_history[j]
                    strategy = strategy if strategy != -1 else 0
                    strategy_index.append((j, strategy))
            strategy_indices.append(strategy_index)
            erc_indices.append(erc_index)
            texts_for_erc.append(text_for_erc)
            dialogues_for_parsing.append(dialogue_for_parsing)
            flattened_contexts.append(context)
        # Diagnostic baseline: Flattened Context only.
        # No emotion tags, strategy tags, discourse graph, or graph learning.
        if self.args.ablation == "flat":
            context_embeddings = self.encode(flattened_contexts)
            logits = self.classifier(context_embeddings)

            return {
                "logits": logits,
                "graphs": [],
                "attention_weights": [],
                "erc_logits": torch.empty(0, device=self.device),
            }

        # Discourse dependency parsing
        if self.lightmode:
            parsed_dialogues = samples["parsed_dialogue"]
        else:
            parsed_dialogues = self.dialogue_parser.parse(dialogues_for_parsing)
        # Emotion recognition
        erc_input = {
            "texts": texts_for_erc
        }
        if self.lightmode:
            erc_logits = self.softmax(samples["erc_logits"].to(self.device) / (self.t * self.scalar))
        else:
            erc_logits = self.softmax(self.erc(erc_input)["logits"] / (self.t * self.scalar))
        if self.args.erc_mixed:
            erc_embeddings = erc_logits @ self.erc_prototypes
        else:
            erc_tags = torch.argmax(erc_logits, dim=-1)
            erc_embeddings = self.erc_prototypes[erc_tags, :]

        # Ablation: w/o Graph Learning.
        # Keep emotion and historical strategy information, but insert
        # them directly as tags into the flattened dialogue context.
        if self.args.ablation == "no_graph":
            tagged_contexts = []

            for i in range(len(samples["dialogue_history"])):
                utterances = samples["dialogue_history"][i].split("</s>")
                speakers = str(samples["speaker_turn"][i]).split(" ")
                strategy_history = [
                    int(s.strip())
                    for s in samples["strategy_history"][i][1:-1].split(",")
                ]

                turns = []
                offset = sum(dialogue_sizes[:i])

                for j in range(len(utterances)):
                    if speakers[j] == "seeker":
                        emotion_id = torch.argmax(
                            erc_logits[offset + j], dim=-1
                        ).item()
                        emotion = self.id2emotion[emotion_id]

                        turns.append(
                            f"[{speakers[j]}] [emotion={emotion}] {utterances[j]}"
                        )
                    else:
                        strategy_id = strategy_history[j]
                        strategy_id = strategy_id if strategy_id != -1 else 0
                        strategy = self.id2strategy[strategy_id]

                        turns.append(
                            f"[{speakers[j]}] [strategy={strategy}] {utterances[j]}"
                        )

                tagged_contexts.append(" ".join(turns))

            context_embeddings = self.encode(tagged_contexts)
            logits = self.classifier(context_embeddings)

            return {
                "logits": logits,
                "graphs": [],
                "attention_weights": [],
                "erc_logits": erc_logits,
            }

        # Normal graph-based variants
        context_embeddings = self.encode(flattened_contexts)

        # Build heterogeneous graph
        graphs = []
        graph_inputs = {
            "embeddings": [],
            "edges": [],
            "edge_types": []
        }
        dummy_indices = []
        graph_sizes = []
        for i in range(len(samples["dialogue_history"])):
            nodes = ["DUMMY"] * (dialogue_sizes[i] + 1)
            pos = torch.tensor([dialogue_sizes[i], ] + np.arange(dialogue_sizes[i]).tolist()).to(self.device)
            pos_embeddings = self.node_position_embedding(pos)
            for j in erc_indices[i]:
                nodes[j + 1] = self.id2emotion[torch.argmax(erc_logits[j + sum(dialogue_sizes[:i]), :], dim=-1).item()]
            for j, sid in strategy_indices[i]:
                nodes[j + 1] = self.id2strategy[sid]
            node_embeddings = torch.zeros((len(nodes), self.args.hg_dim)).to(self.device)
            if self.args.ablation != "no_dummy":
                node_embeddings[0, :] = node_embeddings[0, :] + self.dummy_embedding
            erc_indices_1 = np.array(erc_indices[i]) + 1
            erc_indices_2 = np.array(erc_indices[i]) + sum(dialogue_sizes[:i])
            node_embeddings[erc_indices_1, :] = node_embeddings[erc_indices_1, :] + erc_embeddings[erc_indices_2, :]
            strategy_indices_1 = np.array([s[0] for s in strategy_indices[i]]) + 1
            node_embeddings[strategy_indices_1, :] = node_embeddings[strategy_indices_1, :] + self.strategy_embedding(
                torch.tensor([s[1] for s in strategy_indices[i]]).int().to(self.device))
            # node_embeddings = node_embeddings + pos_embeddings
            edges = []
            edge_types = []
            if self.args.ablation == "no_discourse":
                # Ablation: remove discourse parser structure.
                # Connect utterance nodes sequentially instead.
                for j in range(1, len(nodes) - 1):
                    edges.append([j, j + 1])
                    edge_types.append(self.graph_relation_dict["Continuation"])
            else:
                for head, tail, tp in parsed_dialogues[i]:
                    if head != 0:
                        edges.append([head, tail])
                        edge_types.append(tp)
            if self.args.ablation != "no_dummy":
                for j in range(1, len(nodes)):
                    edges.append([j, 0])
                    if j - 1 in erc_indices[i]:
                        edge_types.append(self.graph_relation_dict["Inter"])
                    else:
                        edge_types.append(self.graph_relation_dict["Self"])
            graph = {
                "nodes": nodes,
                "edges": edges,
                "edge_types": edge_types,
            }
            graphs.append(graph)
            dummy_indices.append(sum(graph_sizes))
            graph_inputs["embeddings"].append(node_embeddings)
            for head, tail in edges:
                graph_inputs["edges"].append([head + sum(graph_sizes), tail + sum(graph_sizes)])
            graph_inputs["edge_types"].extend(edge_types)
            graph_sizes.append(len(nodes))
        graph_inputs["embeddings"] = torch.cat(graph_inputs["embeddings"], dim=0)
        batch_edges = [[], []]
        for head, tail in graph_inputs["edges"]:
            batch_edges[0].append(head)
            batch_edges[1].append(tail)
        graph_inputs["edges"] = torch.tensor(batch_edges).to(self.device)
        graph_inputs["edge_types"] = torch.tensor(graph_inputs["edge_types"]).to(self.device)
        # Graph Layers
        graph_embeddings, atten_weights_1 = self.conv1(graph_inputs["embeddings"], graph_inputs["edges"],
                                                       graph_inputs["edge_types"])
        graph_embeddings, atten_weights_2 = self.conv2(graph_embeddings, graph_inputs["edges"],
                                                       graph_inputs["edge_types"])
        graph_embeddings, atten_weights_3 = self.conv3(graph_embeddings, graph_inputs["edges"],
                                                       graph_inputs["edge_types"])
        if self.args.ablation == "no_dummy":
            # Ablation: replace dummy-node readout with
            # concatenated mean-max pooling over real dialogue nodes.
            pooled_graphs = []
            for start, size in zip(dummy_indices, graph_sizes):
                # index `start` is only an internal placeholder;
                # real utterance nodes are start+1 ... start+size-1.
                node_states = graph_embeddings[start + 1:start + size, :]
                mean_pool = torch.mean(node_states, dim=0)
                max_pool = torch.max(node_states, dim=0).values
                pooled_graphs.append(torch.cat([mean_pool, max_pool], dim=-1))

            graph_embeddings = torch.stack(pooled_graphs, dim=0)
        else:
            graph_embeddings = graph_embeddings[dummy_indices, :]

        # Prediction
        embeddings = torch.cat((graph_embeddings, context_embeddings), dim=-1)
        # embeddings = graph_embeddings
        # embeddings = context_embeddings
        logits = self.classifier(embeddings)
        return {
            "logits": logits,
            "graphs": graphs,
            "attention_weights": [atten_weights_1, atten_weights_2, atten_weights_3],
            "erc_logits": erc_logits,
        }

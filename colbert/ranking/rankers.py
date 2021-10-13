import torch

from functools import partial

from colbert.ranking.index_part import IndexPart
from colbert.ranking.faiss_index import FaissIndex
from colbert.utils.utils import flatten, zipstar


class Ranker():
    def __init__(self, args, inference, faiss_depth=1024):
        self.inference = inference
        self.faiss_depth = faiss_depth # 1024

        if faiss_depth is not None:
            self.faiss_index = FaissIndex(args.index_path, args.faiss_index_path, args.nprobe, part_range=args.part_range)
            self.retrieve = partial(self.faiss_index.retrieve, self.faiss_depth)

        self.index = IndexPart(args.index_path, dim=inference.colbert.dim, part_range=args.part_range, verbose=True)

    def encode(self, queries):
        assert type(queries) in [list, tuple], type(queries)

        Q = self.inference.queryFromText(queries, bsize=512 if len(queries) > 512 else None)

        return Q

    def rank(self, Q, pids=None):
        pids = self.retrieve(Q, verbose=True)[0]
        print('pids', len(pids))

        assert type(pids) in [list, tuple], type(pids)
        assert Q.size(0) == 1, (len(pids), Q.size())
        assert all(type(pid) is int for pid in pids)

        scores = []
        if len(pids) > 0:
            Q = Q.permute(0, 2, 1) # torch.Size([1, 128, 32]) (1, dim, maxQ_keyw)
            #print('Q', Q.shape)

            # USE IndexPart module
            scores = self.index.rank(Q, pids)

            scores_sorter = torch.tensor(scores).sort(descending=True)
            # scores_sorter is of (values, indices)
            #print('scores', scores_sorter[0].shape) # torch.Size([6249])
            pids, scores = torch.tensor(pids)[scores_sorter.indices].tolist(), scores_sorter.values.tolist()

        return pids, scores

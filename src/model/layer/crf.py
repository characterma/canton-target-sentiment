# -*- coding: utf-8 -*-

# __author__ = Jason Zhang
# __version__ = v0.1


import torch
import torch.nn as nn

class LinearChainCRF(nn.Module):

    def __init__(self, in_dim, num_tags):
        super(LinearChainCRF, self).__init__()
        # hyper-parameters
        self.initial_score = -1e9
        self.start_idx = num_tags
        self.end_idx = num_tags + 1
        self.Nc_se = num_tags + 2
        # parameteres
        self.linear_projection = nn.Linear(in_dim, num_tags) if in_dim != num_tags else None
        self.transition_matrix = nn.Parameter(torch.FloatTensor(self.Nc_se, self.Nc_se), requires_grad=True) # [i,j] means score from i to j
        self.register_buffer("start_end_state_emit_score", torch.full((1,1,2), self.initial_score, requires_grad=False))
        self.register_buffer("start_step_emit_score", torch.full((1, self.Nc_se), self.initial_score, requires_grad=False))
        self.start_step_emit_score[0, self.start_idx] = 0.0
        
        self.init_weight()

    def init_weight(self):
        torch.nn.init.xavier_normal_(self.transition_matrix.data)
        self.transition_matrix.data[:, self.start_idx] = self.initial_score
        self.transition_matrix.data[self.end_idx, :] = self.initial_score

    def forward(self, x, length_index=None):
        B, L_max, H = x.size()
        mask = torch.ones(B, L_max, dtype=torch.bool, device=x.device) if length_index is None else length_index.view(B, L_max).bool()
        if self.linear_projection is not None:
            x = self.linear_projection(x)
        x = x * mask.view(B, -1, 1) # [B, L_max, Nc_real]
        # add `START` and `END` states in each time steps: from shape [B, L_max, Nc_real] to [B, L_max, Nc_se]
        x = torch.cat([x, self.start_end_state_emit_score.expand(B, L_max, -1)], dim=-1) # [B, L_max, Nc_se]
        return x, mask

    def nll_loss(self, x, y, length_index = None, reduce='mean'):
        ''' Negative-Log-Likelihood used to train CRF-sequence-tagger;
            for a batch of given observations `X` and corresponding golden standard hidden sequences `y`, with CRF model `M`
            NLL(x, y)   = -log(score(y,x|M))
                        = -log( softmax(score(y,x|M)))
                        = -log(exp(score(y,x|M))) + log(sum(exp(score(Y,x|M))))
                        = log_sum_exp(score(Y,x|M)) - score(y,x|M)
                        = all_path_score - ground_truth_score
            where `Y` is all possible hidden sequences can generate `x`
            Arguments:
            x: [B, L_max, H], a batch of observation sequences, such as the ouput of bilstm in BiLSTM-CRF model;
            y: [B, L_max], the ground truth, correspoding hidden sequences of x;
            length_index: [B, L_max], the mask to indicate the real observations and padding
        '''
        x, mask = self.forward(x, length_index=length_index) # [B, L_max, Nc_se], [B, L_max]
        all_path_score = self._forward_alg(x, mask=mask) # log_sum_exp(score(X, Y))
        ground_truth_score = self._path_score(x, y, mask=mask) # score(X, y)
        batch_nll = all_path_score - ground_truth_score # [B]
        if reduce == None:
            return batch_nll
        elif 'mean' == reduce:
            return batch_nll.mean()
        elif 'sum' == reduce:
            return batch_nll.sum()
        else:
            raise ValueError(f"Expect `reduce` is one of ['mean', 'sum'], but got {reduce}")

    def _forward_alg(self, x, mask):
        ''' forward algorithm: given model `M`, observation `x` to calculate the score `score(x|M)`;
            to calculate `score(x|M)` need accumulate (sum up) all possible hidden sequences `Y` which can emit `x`
            Arguments:
            x: [B, L_max, Nc_se], observation sequences with `START` & `END` states in each time step
            mask: [B, L_max], the mask to indicate the real observations and padding
        '''
        B, L_max, Nc_se = x.size()
        mask = mask.transpose(0, 1).contiguous()
        x = x.transpose(0, 1).contiguous()

        # Initialization: at step START
        # v1: a_0 = [B, Nc_se]: [-1e9,...,0,-1e9], emit_score(start)
        alpha_0 = self.start_step_emit_score.expand(B, -1)
        # # v2: a_0 = [B, Nc_se]: transmit(start, :) + emit_score(step_0)
        # alpha_0 = self.transition_matrix[self.start_idx,:].view(1, Nc_se) + x[0] # [B, Nc_se]

        alpha_previous = alpha_0 # [B, Nc_se]
        # Recursion: alpha_t = log_sum_exp(alpha_previous_i + trans_i_j + emit_current_j) for t in [1, last]
        for t in range(0, L_max):
            # [B, Nc_se, Nc_se]
            alpha = alpha_previous.view(B, Nc_se, 1) + self.transition_matrix.view(1, Nc_se, Nc_se) + x[t].view(B, 1, Nc_se)
            alpha_previous = torch.logsumexp(alpha, dim=1) * mask[t].view(B, 1) + alpha_previous * (~mask[t]).view(B, 1) # [B, Nc_se]
        # Termination: at step END, score = last + trans_to_END
        all_path_score = torch.logsumexp(alpha_previous + self.transition_matrix[:,self.end_idx].view(1, Nc_se), dim=-1) # [B]
        return all_path_score

    def _path_score(self, x, y, mask):
        ''' Arguments:
            x: [B, L_max, Nc_se]
            y: [B, L_max], the ground truth, correspoding hidden sequences of x;
            length_index: [B, L_max], the mask to indicate the real observations and padding
        '''
        B, L_max, _ = x.size()
        x = x.transpose(0, 1).contiguous()
        mask = mask.transpose(0, 1).contiguous()
        y = y.transpose(0, 1).contiguous()

        # from START to step 0
        path_score = self.transition_matrix[self.start_idx,:].index_select(0, y[0]) # [B]
        # from step 0 to step L_max - 2
        for t in range(L_max - 1):
            emit_score = x[t].gather(1, y[t].view(B, 1)).squeeze(1) # [B] 
            trans_score = self.transition_matrix[y[t], y[t+1]] # [B]
            path_score += emit_score * mask[t] + trans_score * mask[t+1] # [B]
        # emit score of real last tag
        real_last_tag = y.gather(0, (mask.sum(0).long() - 1).view(1, B)).squeeze(0) # [B]
        emit_score_last_tag = x[-1].gather(1, real_last_tag.view(B, 1)).squeeze(1) * mask[-1] # [B]
        # transition score from real last tag to END (not the pad to END)
        trans_score_to_end = self.transition_matrix[:, self.end_idx].index_select(0, real_last_tag)
        path_score += emit_score_last_tag + trans_score_to_end
        return path_score

    def viterbi_decode(self, x, length_index=None, top_k=1):
        ''' viterbi algorithm: given model and observation sequences to calculate the most possible hidden state sequences;
            Arguments:
            x: [B, L_max, H], a batch of observation sequences, such as the ouput of bilstm in BiLSTM-CRF model;
            length_index: [B, L_max], the mask to indicate the real observations and padding
            top_k: int
        '''
        x, mask = self.forward(x, length_index=length_index) # [B, L_max, Nc_real], [B, L_max]
        Nc_se = x.size(2)

        batch_path_records, batch_path_scores = [], [] # [B, k, L_real], [B, K]
        for ins, ins_mask in zip(x, mask): # [L_max, Nc_se], [L_max]
            ins_rL = torch.index_select(ins, 0, ins_mask.nonzero().squeeze()) # [L_real, Nc_se]
            L_real = ins_rL.size(0)

            path_record = [] # [L_real, k, Nc_se]
            # Initialization: at step START
            v_0 = self.start_step_emit_score # [1, Nc_se]
            v_previous = v_0
            # Recursion: find topk paths to each state at each step
            for t in range(L_real):
                # v_previous_top_k + trans_to_current
                # [k,Nc_se,1] + [Nc_se, Nc_se] = [k, Nc_se, Nc_se] = [k*Nc_se, Nc_se]
                v_t = (v_previous.unsqueeze(2) + self.transition_matrix).view(-1, Nc_se)
                # find topk path_score and corresponding paths to each state at current step
                k = min(v_t.size(0), top_k)
                v_t, topk_path = torch.topk(v_t, k=k, dim=0) # [k, Nc_se], [k, Nc_se]
                # add emit_score of each state at current step into path score
                # [k, Nc_se] + [Nc_se] = [k, Nc_se]
                v_previous = v_t + ins[t]
                path_record.append(topk_path.squeeze())
            # Termination: at step END
            # [k, Nc_se, 1] + [Nc_se, 1] = [k, Nc_se, 1] = [k*Nc_se]
            v_stop = (v_previous.unsqueeze(2) + self.transition_matrix[:, self.end_idx].view(-1,1)).view(-1)
            k = min(v_stop.size(0), top_k)
            v_stop, topk_path = torch.topk(v_stop, k=k, dim=0) # [k]

            # track back topk best paths from END to START
            topk_paths = [] # [k, L_real]
            topk_scores = [] # [k]
            for i in range(k):
                path = [topk_path[i].item()] # [1]
                for bw_step in reversed(path_record):
                    path.append(int(bw_step.view(-1)[path[-1]]))
                path.reverse()
                path = [kNc % Nc_se for kNc in path[1:]]
                topk_paths.append(path)
                topk_scores.append(v_stop[i].item())
            batch_path_records.append(topk_paths)
            batch_path_scores.append(topk_scores)
        return batch_path_records, batch_path_scores

import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import numpy as np


class Encoder(nn.Module): #입력 시퀀스(그래프)를 숨겨진 벡터로 매핑
    """Maps a graph represented as an input sequence
    to a hidden vector"""
    def __init__(self, input_dim, hidden_dim): # 생성자 메소드로 2개의 인수를 받는다.
        # 입력 차원 및 은닉 차원으로 입력 차원은 각 노드의 feature 벡터를 은닉 차원은 lstm의 숨겨진 상태 차원을 의미
        super(Encoder, self).__init__() # nn.module 생성자 호출
        self.hidden_dim = hidden_dim # hidden.dim을 인스턴스 변수로 저장
        self.lstm = nn.LSTM(input_dim, hidden_dim) #lstm 네트워크 초기화
        self.init_hx, self.init_cx = self.init_hidden(hidden_dim) 
        #초기 숨겨진 상태와 셀 상태를 설정, init_hidden을 호출하여 lstm의 초기 상태 생성

    def forward(self, x, hidden): #모델의 순전파 계산을 정의하는 메서드
        output, hidden = self.lstm(x, hidden) # 이전 hidden cell의 값과 입력 시퀀 x를 lstm에 전달하여 계산
        return output, hidden # 새로운 stlm의 출력과 새로운 hidden cell 반환
    
    def init_hidden(self, hidden_dim): # 초기 은닉 상태와 셀 상태를 설정하는 메서드
        """Trainable initial hidden state"""
        std = 1. / math.sqrt(hidden_dim) #hidden dim의 표준편차를 계산해서 초기 값의 범위 결정하는데 사용
        enc_init_hx = nn.Parameter(torch.FloatTensor(hidden_dim)) #초기 은닉상태를 학습 가능한 파라미터로 생성
        enc_init_hx.data.uniform_(-std, std) #초기화되는 텐서의 값을 -std와 std 사이에서 균일분포 사용하여 초기 hidden state 추출

        enc_init_cx = nn.Parameter(torch.FloatTensor(hidden_dim)) # 초기 셀 상태도 마찬가지임
        enc_init_cx.data.uniform_(-std, std)
        return enc_init_hx, enc_init_cx


class Attention(nn.Module): #nn.Module을 상속받는 attention class
    """A generic attention module for a decoder in seq2seq"""
    def __init__(self, dim, use_tanh=False, C=10): # (차원, tanh활성화 함수를 사용할지 여부에 대한 boolean값, 활성화 함수 값 조절을 위한 scaling 상수)
        super(Attention, self).__init__()
        self.use_tanh = use_tanh # use_tan.h 초기화
        self.project_query = nn.Linear(dim, dim) # query를 투영하기 위한 선형 레이어를 초기화
        self.project_ref = nn.Conv1d(dim, dim, 1, 1) #  추후 decoder와 비교될 인코더의 hidden state값들의 출력을 의미
        self.C = C  # tanh exploration
        self.tanh = nn.Tanh()

        self.v = nn.Parameter(torch.FloatTensor(dim)) # value값의 초기값 생성
        self.v.data.uniform_(-(1. / math.sqrt(dim)), 1. / math.sqrt(dim)) # 초기값의 범위 생성
        
    def forward(self, query, ref): # 모델의 순전파 계산을 정의하는 메서드
        """
        Args: 
            query: is the hidden state of the decoder at the current
                time step. batch x dim 
            ref: the set of hidden states from the encoder. 
                sourceL x batch x hidden_dim
        """
        # query 현재 시점의 디코더에서 숨겨진 상태
        # ref 인코더로부터 숨겨진 상태 집합
        # ref is now [batch_size x hidden_dim x sourceL]
        ref = ref.permute(1, 2, 0) # 추후 쿼리와의 배열 맞추기 위함임
        q = self.project_query(query).unsqueeze(2)  # batch x dim x 1
        e = self.project_ref(ref)  # batch_size x hidden_dim x sourceL 
        # expand the query by sourceL
        # batch x dim x sourceL
        #q와 e의 배열을 맞추기 위한 작업
        expanded_q = q.repeat(1, 1, e.size(2)) 
        # batch x 1 x hidden_dim #e.size(2)를 통해 완벽하게 e와 배열을 맞춤
        v_view = self.v.unsqueeze(0).expand(
                expanded_q.size(0), len(self.v)).unsqueeze(1)
        # [batch_size x 1 x hidden_dim] * [batch_size x hidden_dim x sourceL]
        # 가중치 벡터를 곱하기 위한 작업으로 additive attetion 의 한 과정임
        u = torch.bmm(v_view, self.tanh(expanded_q + e)).squeeze(1) # 이것이 attention score
        if self.use_tanh: # tanh르 쓸것이냐 안 쓸것이냐에 대한 값(init부분에 나와있음)
            logits = self.C * self.tanh(u)
        else:
            logits = u  
        return e, logits #매핑된 e와 계산된 attention score 반환


class Decoder(nn.Module):
    def __init__(self, 
            embedding_dim, #출력값 임베딩 차원수
            hidden_dim, # hidden state 차원 수
            tanh_exploration, # tan h 함수이 scaling parameter
            use_tanh, # tanh의 사용 여부
            n_glimpses=1, #glimpse의 수로 context 정보를 살펴보기 위해 몇번의 glimpse를 사용할지 결정
            mask_glimpses=True, # glimpse에서 마스킹을 사용할지 여부
            mask_logits=True): #로짓에서 마스킹을 사용할지 여부
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_glimpses = n_glimpses
        self.mask_glimpses = mask_glimpses
        self.mask_logits = mask_logits
        self.use_tanh = use_tanh
        self.tanh_exploration = tanh_exploration
        self.decode_type = None  # Needs to be set explicitly before use

        self.lstm = nn.LSTMCell(embedding_dim, hidden_dim) #단일 타임 스텝에서 lstm동작을 정의
        self.pointer = Attention(hidden_dim, use_tanh=use_tanh, C=tanh_exploration) # 디코더가 입력 시퀀스의 특정 부분에 주의를 기울이도록 함
        self.glimpse = Attention(hidden_dim, use_tanh=False) # 디코더가 입력 시퀀스를 여러번 살펴보도록 함
        self.sm = nn.Softmax(dim=1) # 로짓을 확률로 변환

    def update_mask(self, mask, selected):
        return mask.clone().scatter_(1, selected.unsqueeze(-1), True)
    #마스크를 업데이트 하는 역할을 하며 선택된 인덱스는 마스크를 true로 설정하여 다음 단계에서 디코더가 선택된 위치를 다시 선택하지 않도록 함

    def recurrence(self, x, h_in, prev_mask, prev_idxs, step, context):

        logit_mask = self.update_mask(prev_mask, prev_idxs) if prev_idxs is not None else prev_mask
        #이전에 사용된 인덱스를 기반으로 마스크를 업데이트 한다.

        logits, h_out = self.calc_logits(x, h_in, logit_mask, context, self.mask_glimpses, self.mask_logits)
        #로짓을 계산해서 다음 hidden state의 out값과 새로운 logit을 도출
    
        # Calculate log_softmax for better numerical stability
        log_p = torch.log_softmax(logits, dim=1)
        probs = log_p.exp()

        if not self.mask_logits:
            # If self.mask_logits, this would be redundant, otherwise we must mask to make sure we don't resample
            # Note that as a result the vector of probs may not sum to one (this is OK for .multinomial sampling)
            # But practically by not masking the logits, a model is learned over all sequences (also infeasible)
            # while only during sampling feasibility is enforced (a.k.a. by setting to 0. here)
            probs[logit_mask] = 0.
            # For consistency we should also mask out in log_p, but the values set to 0 will not be sampled and
            # Therefore not be used by the reinforce estimator

        return h_out, log_p, probs, logit_mask

    def calc_logits(self, x, h_in, logit_mask, context, mask_glimpses=None, mask_logits=None):

        if mask_glimpses is None:
            mask_glimpses = self.mask_glimpses

        if mask_logits is None:
            mask_logits = self.mask_logits

        hy, cy = self.lstm(x, h_in)
        g_l, h_out = hy, (hy, cy)

        for i in range(self.n_glimpses):
            ref, logits = self.glimpse(g_l, context)
            # For the glimpses, only mask before softmax so we have always an L1 norm 1 readout vector
            if mask_glimpses:
                logits[logit_mask] = -np.inf
            # [batch_size x h_dim x sourceL] * [batch_size x sourceL x 1] =
            # [batch_size x h_dim x 1]
            g_l = torch.bmm(ref, self.sm(logits).unsqueeze(2)).squeeze(2)
        _, logits = self.pointer(g_l, context)

        # Masking before softmax makes probs sum to one
        if mask_logits:
            logits[logit_mask] = -np.inf
                """현재 입력과 이전 은닉 상태를 사용하여 LSTM 셀을 업데이트합니다.
                여러 번의 glimpse 어텐션을 통해 컨텍스트 정보를 반복적으로 살펴봅니다.
                Pointer 어텐션을 적용하여 최종 로짓을 계산합니다.
                필요한 경우 로짓에서 마스킹을 적용합니다.
                최종 로짓과 새로운 은닉 상태를 반환합니다.
                """

        return logits, h_out

    def forward(self, decoder_input, embedded_inputs, hidden, context, eval_tours=None):
        """
        Args:
            decoder_input: The initial input to the decoder
                size is [batch_size x embedding_dim]. Trainable parameter.
            embedded_inputs: [sourceL x batch_size x embedding_dim]
            hidden: the prev hidden state, size is [batch_size x hidden_dim]. 
                Initially this is set to (enc_h[-1], enc_c[-1])
            context: encoder outputs, [sourceL x batch_size x hidden_dim] 
        """

        batch_size = context.size(1) # context(encoder output)에서 배치 크기 추출
        outputs = []# 값 리스트 초기화
        selections = [] #인덱스 저장 리스트 초기화
        steps = range(embedded_inputs.size(0)) # 단계(index의 수) 정의
        idxs = None # 초기 인덱스는 none
        mask = Variable(
            embedded_inputs.data.new().byte().new(embedded_inputs.size(1), embedded_inputs.size(0)).zero_(),
            requires_grad=False
        )# 마스크텐서를 초기화하고 마스킹된 값은 역전파동안 기울기 계산 안하도록 설정

        for i in steps:
            hidden, log_p, probs, mask = self.recurrence(decoder_input, hidden, mask, idxs, i, context)
            # select the next inputs for the decoder [batch_size x hidden_dim]
            idxs = self.decode(
                probs,
                mask
            ) if eval_tours is None else eval_tours[:, i] #확률 분포와 마스크를 이용해 다음 입력 인덱스를 선택하며, none일 경우 평가용 경로에서 인덱스를 가져옴

            idxs = idxs.detach()  # Otherwise pytorch complains it want's a reward, todo implement this more properly?
            # 선택된 인덱스를 분리하여 그레디언트가 역전파되지 않도록 함

            # Gather input embedding of selected
            decoder_input = torch.gather(
                embedded_inputs,
                0,
                idxs.contiguous().view(1, batch_size, 1).expand(1, batch_size, *embedded_inputs.size()[2:])
            ).squeeze(0)
            #디코더의 인풋값의 형태를 통일하여 제대로 들어갈 수 있도록 하는 정제 과정

            # use outs to point to next object
            outputs.append(log_p)
            selections.append(idxs) # 초기화된 리스트에 값과 인덱스값 저장
        return (torch.stack(outputs, 1), torch.stack(selections, 1)), hidden
        # 타임스텝 완료후 output과 selection 리스트를 텐서로 변환하여 반하하고 hidden은 최종 은닉 상태를 반환

    def decode(self, probs, mask): # 다음 단계로 진행하기 위한 확률 분포로부터 인덱스를 선택하는 방법을 정의
        if self.decode_type == "greedy": #greedy와 sampling 2가지 방식으로 이루어짐
            _, idxs = probs.max(1) # greedy는 확률 분포에서 가장 큰 값을 가지는 인덱스 선택
            assert not mask.gather(1, idxs.unsqueeze(-1)).data.any(), \
                "Decode greedy: infeasible action has maximum probability"
        elif self.decode_type == "sampling": #확률에 비례하게 값을 추출
            idxs = probs.multinomial(1).squeeze(1)
            # Check if sampling went OK, can go wrong due to bug on GPU
            while mask.gather(1, idxs.unsqueeze(-1)).data.any():
                print(' [!] resampling due to race condition')
                idxs = probs.multinomial().squeeze(1)
        else:
            assert False, "Unknown decode type"

        return idxs


class CriticNetworkLSTM(nn.Module): # 추후 강화학습의 policy 평가에서 사용됨 최종적으로 디코더를 통해 스칼라 값 출력 
    """Useful as a baseline in REINFORCE updates"""
    def __init__(self,
            embedding_dim,
            hidden_dim,
            n_process_block_iters,
            tanh_exploration,
            use_tanh):
        super(CriticNetworkLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_process_block_iters = n_process_block_iters

        self.encoder = Encoder(embedding_dim, hidden_dim)
        
        self.process_block = Attention(hidden_dim, use_tanh=use_tanh, C=tanh_exploration)
        self.sm = nn.Softmax(dim=1)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, inputs):
        """
        Args:
            inputs: [embedding_dim x batch_size x sourceL] of embedded inputs
        """
        inputs = inputs.transpose(0, 1).contiguous() #입력텐서를 변환

        encoder_hx = self.encoder.init_hx.unsqueeze(0).repeat(inputs.size(1), 1).unsqueeze(0) #엔코더의 초기 은닉상태와 셀 상태를 설정
        encoder_cx = self.encoder.init_cx.unsqueeze(0).repeat(inputs.size(1), 1).unsqueeze(0)
        
        # encoder forward pass
        enc_outputs, (enc_h_t, enc_c_t) = self.encoder(inputs, (encoder_hx, encoder_cx)) #인코더의 출력이며 최종 은닉 상태와 셀 상태
        
        # grab the hidden state and process it via the process block 
        process_block_state = enc_h_t[-1] # 최종 encoder의 은닉 상태를 프로세스 블록의 초기 상태로 설정
        for i in range(self.n_process_block_iters): # n process block iters 만큼 반복
            ref, logits = self.process_block(process_block_state, enc_outputs) 
            process_block_state = torch.bmm(ref, self.sm(logits).unsqueeze(2)).squeeze(2)
        # produce the final scalar output# 인코더의 출력을 반복적으로 처리하고 최종 상태를 얻음
        out = self.decoder(process_block_state)
        return out#디코더를 통해 최종 출력을 생성하는 데 사용


class PointerNetwork(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 n_encode_layers=None,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization=None,
                 **kwargs):
        super(PointerNetwork, self).__init__()

        self.problem = problem
        assert problem.NAME == "tsp", "Pointer Network only supported for TSP"
        self.input_dim = 2

        self.encoder = Encoder(
            embedding_dim,
            hidden_dim)

        self.decoder = Decoder(
            embedding_dim,
            hidden_dim,
            tanh_exploration=tanh_clipping,
            use_tanh=tanh_clipping > 0,
            n_glimpses=1,
            mask_glimpses=mask_inner,
            mask_logits=mask_logits
        )

        # Trainable initial hidden states # 첫번쨰 decoder의 state를 정의의
        std = 1. / math.sqrt(embedding_dim)
        self.decoder_in_0 = nn.Parameter(torch.FloatTensor(embedding_dim))
        self.decoder_in_0.data.uniform_(-std, std)

        self.embedding = nn.Parameter(torch.FloatTensor(self.input_dim, embedding_dim))
        self.embedding.data.uniform_(-std, std)

    def set_decode_type(self, decode_type): #greedy인지 sampling인지 설정
        self.decoder.decode_type = decode_type

    def forward(self, inputs, eval_tours=None, return_pi=False): #내부연산수행을 통해 비용 및 로그 확률 계산
        batch_size, graph_size, input_dim = inputs.size()

        embedded_inputs = torch.mm(
            inputs.transpose(0, 1).contiguous().view(-1, input_dim),
            self.embedding
        ).view(graph_size, batch_size, -1)

        # query the actor net for the input indices 
        # making up the output, and the pointer attn 
        _log_p, pi = self._inner(embedded_inputs, eval_tours)

        cost, mask = self.problem.get_costs(inputs, pi)
        # Log likelyhood is calculated within the model since returning it per action does not work well with
        # DataParallel since sequences can be of different lengths
        ll = self._calc_log_likelihood(_log_p, pi, mask)
        if return_pi: # 결과 반환
            return cost, ll, pi

        return cost, ll

    def _calc_log_likelihood(self, _log_p, a, mask): #

        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1)

    def _inner(self, inputs, eval_tours=None): # 입력 데이터를 인코딩하고 디코딩하여 포인터 확률과 선택된 인덱스를 반환

        encoder_hx = encoder_cx = Variable(
            torch.zeros(1, inputs.size(1), self.encoder.hidden_dim, out=inputs.data.new()),
            requires_grad=False
        )

        # encoder forward pass
        enc_h, (enc_h_t, enc_c_t) = self.encoder(inputs, (encoder_hx, encoder_cx))

        dec_init_state = (enc_h_t[-1], enc_c_t[-1])

        # repeat decoder_in_0 across batch
        decoder_input = self.decoder_in_0.unsqueeze(0).repeat(inputs.size(1), 1)

        (pointer_probs, input_idxs), dec_hidden_t = self.decoder(decoder_input,
                                                                 inputs,
                                                                 dec_init_state,
                                                                 enc_h,
                                                                 eval_tours)

        return pointer_probs, input_idxs
import os
import json
import glob
import base64
import argparse

from transformers import AutoTokenizer


class LlmExporter():
    '''
    Base class for all llm model export. Inherits from [`torch.nn.Module`].
    '''
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.stop_ids = []
        self.load_pretrained()

    def load_pretrained(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_path, trust_remote_code=True, use_fast=False)
        except:
            self.tokenizer = None
        if None == self.tokenizer:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_path, trust_remote_code=True)
            except:
                self.tokenizer = None
        if None == self.tokenizer:
            print("Default load tokenizer failed for ", self.args.tokenizer_path)
        
        if os.path.exists(os.path.join(self.args.tokenizer_path, 'generation_config.json')):
            self.generation_config = json.load(open(os.path.join(self.args.tokenizer_path, 'generation_config.json'), 'r'))
        else:
            self.generation_config = None
        
        
        self.stop_ids.append(self.tokenizer.eos_token_id)
        if hasattr(self.tokenizer, 'im_end_id'):
            self.stop_ids.append(self.tokenizer.im_end_id)
        try:
            eot_id = self.tokenizer.encode('<|eot_id|>')
            if len(eot_id) == 1:
                self.stop_ids.append(eot_id[0])
            # gemma/gemma-2
            eot_id = self.tokenizer.encode('<end_of_turn>')
            if len(eot_id) == 2 and eot_id[0] == 2:
                self.stop_ids.append(eot_id[1])
        except:
            pass
        # if hasattr(self.model, 'generation_config') and self.model.generation_config is not None:
        if self.generation_config is not None:
            eos_token_id = self.generation_config.get('eos_token_id', None)
            from collections.abc import Iterable
            if isinstance(eos_token_id, int):
                self.stop_ids.append(eos_token_id)
            elif isinstance(eos_token_id, Iterable):
                for id in eos_token_id:
                    self.stop_ids.append(id)
                
        self.stop_ids = [stop_id for stop_id in self.stop_ids if stop_id is not None]
        self.stop_ids = list(set(self.stop_ids))
        

    def export_tokenizer(self):
        # load tokenizer file
        tokenizer_model = os.path.join(self.args.tokenizer_path, 'tokenizer.model')
        ice_text_model = os.path.join(self.args.tokenizer_path, 'ice_text.model')
        try:
            import sentencepiece as spm
            if os.path.exists(tokenizer_model):
                self.sp_model = spm.SentencePieceProcessor(tokenizer_model)
            elif os.path.exists(ice_text_model):
                self.sp_model = spm.SentencePieceProcessor(ice_text_model)
            else:
                self.sp_model = None
        except:
            self.sp_model = None
        merge_file = os.path.join(self.args.tokenizer_path, 'merges.txt')
        if os.path.exists(merge_file):
            self.merge_txt = merge_file
        else:
            self.merge_txt = None
        # TOKENIZER MAGIC NUMBER
        MAGIC_NUMBER = 430
        # TOKENIZER TYPE
        SENTENCEPIECE = 0; TIKTOIKEN = 1; BERT = 2; HUGGINGFACE = 3
        def write_line(fp, *args):
            for arg in args:
                for token in arg:
                    fp.write(str(token) + ' ')
            fp.write('\n')
        def write_header(fp, type, speicals, prefix = []):
            fp.write(f'{MAGIC_NUMBER} {type}\n')
            fp.write(f'{len(speicals)} {len(self.stop_ids)} {len(prefix)}\n')
            write_line(fp, speicals, self.stop_ids, prefix)

        # if not os.path.exists(self.args.dst_path):
        #     os.makedirs(self.args.dst_path)
        
        file_path = self.args.dst_path
        special_list = list(self.tokenizer.added_tokens_decoder.keys())
        if hasattr(self.tokenizer, 'special_tokens'):
            for k, v in self.tokenizer.special_tokens.items():
                special_list.append(v)
        if hasattr(self.tokenizer, 'all_special_ids'): #gemma3
            special_list.extend(self.tokenizer.all_special_ids)
        if hasattr(self.tokenizer, 'gmask_token_id'):
            special_list.append(self.tokenizer.gmask_token_id)
            
        if self.generation_config is not None:
            generation_config = self.generation_config
            if hasattr(generation_config, 'user_token_id'):
                special_list.append(generation_config.user_token_id)
            if hasattr(generation_config, 'assistant_token_id'):
                special_list.append(generation_config.assistant_token_id)
                
        vocab_list = []
        prefix_list = []
        if hasattr(self.tokenizer, 'get_prefix_tokens'):
            prefix_list = self.tokenizer.get_prefix_tokens()
        if len(prefix_list) == 0:
            try:
                test_txt = 'A'
                ids = self.tokenizer.encode(test_txt)
                get_txt = self.tokenizer.decode(ids[-1])
                if len(ids) > 1 and get_txt == test_txt:
                    prefix_list += ids[:-1]
            except:
                pass

        if self.sp_model is not None:
            # senetencepiece
            NORMAL = 1; UNKNOWN = 2; CONTROL = 3
            USER_DEFINED = 4; UNUSED = 5; BYTE = 6
            for i in range(self.sp_model.GetPieceSize()):
                token = self.sp_model.IdToPiece(i)
                score = self.sp_model.GetScore(i)
                token_type = NORMAL
                if self.sp_model.IsUnknown(i):
                    token_type = UNKNOWN
                elif self.sp_model.IsControl(i):
                    token_type = CONTROL
                elif self.sp_model.IsUnused(i):
                    token_type = UNUSED
                elif self.sp_model.IsByte(i):
                    token_type = BYTE
                if self.args.path == 'Chatglm_6b':
                    if '<n>' in token: token = '\n'
                    if '<|tab|>' in token: token = '\t'
                    if '<|blank_' in token: token = ' ' * int(token[8:token.find('|>')])
                if '▁' in token: token = token.replace('▁', ' ')
                token_encode = base64.b64encode(token.encode("utf-8")).decode("utf8")
                vocab_list.append(f'{token_encode} {score} {token_type}\n')
            # add special tokens to vocab_list
            for index in special_list:
                if index >= len(vocab_list):
                    try:
                        token = self.tokenizer.decode(index)
                        token_encode = base64.b64encode(token.encode("utf-8")).decode("utf8")
                        vocab_list.append(f'{token_encode} {0} {NORMAL}\n')
                    except:
                        pass
            with open(file_path, "w", encoding="utf8") as fp:
                write_header(fp, SENTENCEPIECE, special_list, prefix_list)
                if self.model_type == "gemma3" or self.model_type == "gemma3-text":
                    fp.write(f'{len(vocab_list) + 1}\n') # len(vocab_list)==262144, self.tokenizer([262144])=='image_soft_token' is a special token
                else:
                    fp.write(f'{len(vocab_list)}\n')
                for vocab in vocab_list:
                    fp.write(vocab)
        elif hasattr(self.tokenizer, 'mergeable_ranks'):
            # tikton
            vocab_list = []
            for k, v in self.tokenizer.mergeable_ranks.items():
                line = base64.b64encode(k).decode("utf8") + "\n"
                vocab_list.append(line)
            if hasattr(self.tokenizer, 'special_tokens'):
                for k, v in self.tokenizer.special_tokens.items():
                    line = base64.b64encode(k.encode("utf-8")).decode("utf8") + "\n"
                    vocab_list.append(line)
            if hasattr(self.tokenizer, 'added_tokens_decoder'):
                for k, v in self.tokenizer.added_tokens_decoder.items():
                    line = base64.b64encode(v.__str__().encode("utf-8")).decode("utf8") + "\n"
                    vocab_list.append(line)
            with open(file_path, "w", encoding="utf8") as fp:
                write_header(fp, TIKTOIKEN, special_list, prefix_list)
                fp.write(f'{len(vocab_list)}\n')
                for vocab in vocab_list:
                    fp.write(vocab)
        elif self.merge_txt is not None:
            # huggingface tokenizer
            merge_list = []
            vocab = self.tokenizer.get_vocab()
            special_list = list(self.tokenizer.added_tokens_decoder.keys())
            vocab_list = ['<unk>' for i in range(len(vocab))]
            # load vocab
            for k, v in vocab.items():
                vocab_list[int(v)] = k
            # load merge
            with open(self.merge_txt, 'rt') as merge:
                for line in merge.readlines():
                    merge_list.append(line)
            # write to tokenizer.txt
            with open(file_path, "w", encoding="utf8") as fp:
                write_header(fp, HUGGINGFACE, special_list)
                fp.write(f'{len(vocab_list)} {len(merge_list)}\n')
                for v in vocab_list:
                    fp.write(v + '\n')
                for m in merge_list:
                    fp.write(m)
        else:
            # tiktoken or bert
            if 'bert' in type(self.tokenizer).__name__.lower():
                tokenizer_type = BERT
            else:
                tokenizer_type = TIKTOIKEN
            # bert tokenizer
            def unicode_to_byte(u: int):
                if u >= 256 and u <= 288:
                    return u - 256
                if u >= 289 and u <= 322:
                    return u - 162
                if u == 323:
                    return 173
                # need not convert using jinja
                # if u == 65372: # |
                #     return 124
                # if u == 9601:  # _
                #     return 95
                return u
            vocab = self.tokenizer.get_vocab()
            vocab_list = ['<unk>' for i in range(len(vocab))]
            for k, v in vocab.items():
                try:
                    vocab_list[int(v)] = bytes([unicode_to_byte(ord(c)) for c in k])
                except:
                    vocab_list[int(v)] = k.encode('utf-8')

            special_list = list(self.tokenizer.added_tokens_decoder.keys())
            with open(file_path, "w", encoding="utf8") as fp:
                write_header(fp, tokenizer_type, special_list)
                fp.write(f'{len(vocab_list)}\n')
                for v in vocab_list:
                    line = base64.b64encode(v).decode("utf8") + "\n"
                    fp.write(line)
        return file_path


def main():
    parser = argparse.ArgumentParser(description='llm_exporter', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--tokenizer_path', type=str, help='tokenizer path(hf path or local dir, for example: "Qwen/Qwen3-VL-2B-Instruct" "Qwen/Qwen3-1.7B")')
    parser.add_argument('--dst_path', type=str, default='./tokenizer', help='export tokenizer path, defaut is `./qwen3_tokenizer.txt`.')
    args = parser.parse_args()


    llm_exporter = LlmExporter(args)
    llm_exporter.export_tokenizer()

if __name__ == '__main__':
    main()
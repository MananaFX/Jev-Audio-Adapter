import unittest
import torch
from torch import nn
from jev_audio_adapter.model import prompt_parts, MARKER, LoRALinear, preprocess_audio
from jev_audio_adapter.loader import apply_release

class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return '\n'.join(m['content'] for m in messages)

class ContractTests(unittest.TestCase):
    def test_prompt_and_codes(self):
        before, after = prompt_parts(Tokenizer(), 'Which?', ['speech', 'music'], ['A','CC'])
        self.assertIn('State:', before)
        self.assertIn('CC: music', after)
        for choices in ([], [''], [5]):
            with self.assertRaises(ValueError): prompt_parts(Tokenizer(), 'q', choices, ['A'])
        with self.assertRaises(ValueError): prompt_parts(Tokenizer(), MARKER, ['a'], ['A'])

    def test_lora_exact_fp32(self):
        torch.manual_seed(7)
        base = nn.Linear(4,3)
        layer = LoRALinear(base, rank=2, alpha=4).eval()
        with torch.no_grad(): layer.b.normal_()
        x = torch.randn(2,4)
        wanted = base(x) + (x @ layer.a.T @ layer.b.T) * 2
        torch.testing.assert_close(layer(x), wanted, rtol=0, atol=0)

    def test_no_truncation(self):
        class Extractor:
            sampling_rate=16000; padding_side='right'; padding_value=0; dither=0; n_samples=4800000
            def __call__(self, data, **kw): return kw
        result = preprocess_audio(Extractor(), [torch.zeros(16000)])
        self.assertFalse(result['truncation'])
        self.assertTrue(result['return_attention_mask'])

    def test_release_rejects_inventory(self):
        with self.assertRaises(ValueError): apply_release(nn.Linear(4,3), {})

if __name__ == '__main__': unittest.main()
